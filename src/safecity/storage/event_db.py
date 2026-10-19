"""SafeCity AI: Near-Miss Event Database, Hysteresis & Evidence Extraction.

Manages persistent SQLite logging of safety conflicts with temporal hysteresis,
pair cooldown, deduplication, and automated MP4 evidence clip generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
import math
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import cv2

logger = logging.getLogger("safecity.storage")


@dataclass
class ActiveConflict:
    """Tracks an ongoing spatiotemporal conflict encounter between two agents."""
    pair_key: Tuple[int, int]
    agent1_id: int
    agent2_id: int
    agent1_class: str
    agent2_class: str
    start_frame: int
    last_seen_frame: int
    consecutive_conflict_frames: int = 0
    alert_triggered: bool = False
    alert_frame: Optional[int] = None
    peak_risk_frame: Optional[int] = None
    min_ttc: float = float("inf")
    min_pet: float = float("inf")
    max_drac: float = 0.0
    min_separation: float = float("inf")
    max_rel_speed: float = 0.0
    approach_angle_deg: float = 0.0
    motion_type: str = "unknown"
    highest_severity: str = "warning"
    prob_collision_2s: float = 0.0
    risk_score_prob: float = 0.0
    is_calibrated: bool = False


class EventDatabase:
    """Manages SQLite storage for near-miss conflict events."""

    def __init__(self, db_path: str = "outputs/events.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    video_source TEXT,
                    agent1_id INTEGER,
                    agent2_id INTEGER,
                    agent1_class TEXT,
                    agent2_class TEXT,
                    start_frame INTEGER,
                    alert_frame INTEGER,
                    peak_risk_frame INTEGER,
                    end_frame INTEGER,
                    min_ttc REAL,
                    min_pet REAL,
                    max_drac REAL,
                    min_separation REAL,
                    relative_speed REAL,
                    approach_angle_deg REAL,
                    motion_type TEXT,
                    severity TEXT,
                    is_calibrated INTEGER,
                    evidence_clip_path TEXT,
                    prob_collision_2s REAL,
                    risk_score_prob REAL
                )
            """)
            # Schema migration for existing databases
            cursor.execute("PRAGMA table_info(events)")
            existing_cols = [row[1] for row in cursor.fetchall()]
            if "prob_collision_2s" not in existing_cols:
                cursor.execute("ALTER TABLE events ADD COLUMN prob_collision_2s REAL")
            if "risk_score_prob" not in existing_cols:
                cursor.execute("ALTER TABLE events ADD COLUMN risk_score_prob REAL")
            conn.commit()

    def insert_event(self, event_data: Dict[str, Any]) -> None:
        """Inserts a finalized conflict event into SQLite."""
        query = """
            INSERT OR REPLACE INTO events (
                event_id, created_at, video_source, agent1_id, agent2_id,
                agent1_class, agent2_class, start_frame, alert_frame, peak_risk_frame,
                end_frame, min_ttc, min_pet, max_drac, min_separation,
                relative_speed, approach_angle_deg, motion_type, severity,
                is_calibrated, evidence_clip_path, prob_collision_2s, risk_score_prob
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            event_data["event_id"],
            event_data.get("created_at", datetime.now().isoformat()),
            event_data.get("video_source", ""),
            event_data["agent1_id"],
            event_data["agent2_id"],
            event_data.get("agent1_class", "car"),
            event_data.get("agent2_class", "car"),
            event_data["start_frame"],
            event_data.get("alert_frame", event_data["start_frame"]),
            event_data.get("peak_risk_frame", event_data["start_frame"]),
            event_data["end_frame"],
            event_data.get("min_ttc"),
            event_data.get("min_pet"),
            event_data.get("max_drac"),
            event_data.get("min_separation"),
            event_data.get("relative_speed"),
            event_data.get("approach_angle_deg"),
            event_data.get("motion_type", "crossing"),
            event_data.get("severity", "warning"),
            1 if event_data.get("is_calibrated", False) else 0,
            event_data.get("evidence_clip_path", ""),
            event_data.get("prob_collision_2s", 0.0),
            event_data.get("risk_score_prob", 0.0)
        )

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()

        logger.info(f"Recorded event {event_data['event_id']} in database (Agents {event_data['agent1_id']} & {event_data['agent2_id']})")


class NearMissEventManager:
    """Handles temporal hysteresis, cooldown windows, deduplication, and evidence export."""

    def __init__(
        self,
        db_path: str = "outputs/events.db",
        evidence_dir: str = "outputs/evidence",
        hysteresis_frames: int = 3,
        cooldown_frames: int = 30,
        clip_padding_sec: float = 2.0
    ) -> None:
        self.db = EventDatabase(db_path)
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.hysteresis_frames = hysteresis_frames
        self.cooldown_frames = cooldown_frames
        self.clip_padding_sec = clip_padding_sec

        self.active_conflicts: Dict[Tuple[int, int], ActiveConflict] = {}
        self.pair_cooldowns: Dict[Tuple[int, int], int] = {}  # pair_key -> frame when cooldown expires
        self.event_counter = 0

    def process_frame_conflicts(
        self,
        frame_idx: int,
        conflicts: List[Dict[str, Any]],
        video_source_path: Optional[str] = None,
        fps: float = 12.5
    ) -> List[Dict[str, Any]]:
        """Updates conflict states with new frame detections. Finalizes completed events."""
        current_frame_pairs = set()
        triggered_events = []

        for conf in conflicts:
            id1 = conf["agent1_id"]
            id2 = conf["agent2_id"]
            pair_key = (min(id1, id2), max(id1, id2))
            current_frame_pairs.add(pair_key)

            # Check if this pair is in cooldown
            if pair_key in self.pair_cooldowns and frame_idx < self.pair_cooldowns[pair_key]:
                continue

            if pair_key not in self.active_conflicts:
                self.active_conflicts[pair_key] = ActiveConflict(
                    pair_key=pair_key,
                    agent1_id=id1,
                    agent2_id=id2,
                    agent1_class=conf["agent1_class"],
                    agent2_class=conf["agent2_class"],
                    start_frame=frame_idx,
                    last_seen_frame=frame_idx,
                    is_calibrated=conf.get("is_calibrated", False)
                )

            act = self.active_conflicts[pair_key]
            act.last_seen_frame = frame_idx
            act.consecutive_conflict_frames += 1

            # Update peak metrics
            ttc = conf.get("ttc_sec")
            if ttc is not None and ttc < act.min_ttc:
                act.min_ttc = ttc
                act.peak_risk_frame = frame_idx

            pet = conf.get("predicted_pet_sec")
            if pet is not None and pet < act.min_pet:
                act.min_pet = pet

            drac = conf.get("drac_m_s2")
            if drac is not None and drac > act.max_drac:
                act.max_drac = drac

            sep = conf.get("current_distance")
            if sep is not None and sep < act.min_separation:
                act.min_separation = sep

            act.max_rel_speed = max(act.max_rel_speed, conf.get("relative_speed", 0.0))
            act.approach_angle_deg = conf.get("approach_angle_deg", act.approach_angle_deg)
            act.motion_type = conf.get("motion_type", act.motion_type)

            if conf.get("severity") == "critical":
                act.highest_severity = "critical"

            if "probabilistic_ttc" in conf:
                p_info = conf["probabilistic_ttc"]
                p_2s = p_info.get("prob_collision_2s", 0.0)
                p_score = p_info.get("risk_score_0_100", 0.0)
                act.prob_collision_2s = max(act.prob_collision_2s, p_2s)
                act.risk_score_prob = max(act.risk_score_prob, p_score)

            # Check Hysteresis: promote to official alert once threshold reached
            if not act.alert_triggered and act.consecutive_conflict_frames >= self.hysteresis_frames:
                act.alert_triggered = True
                act.alert_frame = frame_idx
                logger.warning(f"⚠️ [ALERT TRIGGERED] Near-miss between Agent {id1} and Agent {id2} at frame {frame_idx}!")

        # Finalize conflicts that are no longer active in this frame
        finished_keys = []
        for pair_key, act in self.active_conflicts.items():
            if pair_key not in current_frame_pairs or (frame_idx - act.last_seen_frame > 5):
                # The encounter has ended
                if act.alert_triggered:
                    event_dict = self._finalize_event(act, frame_idx, video_source_path, fps)
                    triggered_events.append(event_dict)
                finished_keys.append(pair_key)
                self.pair_cooldowns[pair_key] = frame_idx + self.cooldown_frames

        for k in finished_keys:
            del self.active_conflicts[k]

        return triggered_events

    def flush_remaining_events(
        self,
        final_frame_idx: int,
        video_source_path: Optional[str] = None,
        fps: float = 12.5
    ) -> List[Dict[str, Any]]:
        """Finalizes any open encounters at the end of the video."""
        finished_events = []
        for act in list(self.active_conflicts.values()):
            if act.alert_triggered:
                event_dict = self._finalize_event(act, final_frame_idx, video_source_path, fps)
                finished_events.append(event_dict)
        self.active_conflicts.clear()
        return finished_events

    def _finalize_event(
        self,
        act: ActiveConflict,
        end_frame: int,
        video_source_path: Optional[str],
        fps: float
    ) -> Dict[str, Any]:
        """Packages event record, extracts evidence video clip, and commits to SQLite."""
        self.event_counter += 1
        event_id = f"EVT_{self.event_counter:04d}"

        clip_path = ""
        if video_source_path and Path(video_source_path).exists():
            clip_path = self._extract_evidence_clip(
                video_source_path=video_source_path,
                event_id=event_id,
                start_frame=act.start_frame,
                end_frame=end_frame,
                fps=fps
            )

        event_data = {
            "event_id": event_id,
            "created_at": datetime.now().isoformat(),
            "video_source": str(video_source_path or ""),
            "agent1_id": act.agent1_id,
            "agent2_id": act.agent2_id,
            "agent1_class": act.agent1_class,
            "agent2_class": act.agent2_class,
            "start_frame": act.start_frame,
            "alert_frame": act.alert_frame or act.start_frame,
            "peak_risk_frame": act.peak_risk_frame or act.start_frame,
            "end_frame": end_frame,
            "min_ttc": None if math.isinf(act.min_ttc) else round(act.min_ttc, 2),
            "min_pet": None if math.isinf(act.min_pet) else round(act.min_pet, 2),
            "max_drac": round(act.max_drac, 2) if act.max_drac > 0 else None,
            "min_separation": None if math.isinf(act.min_separation) else round(act.min_separation, 2),
            "relative_speed": round(act.max_rel_speed, 2),
            "approach_angle_deg": round(act.approach_angle_deg, 1),
            "motion_type": act.motion_type,
            "severity": act.highest_severity,
            "is_calibrated": act.is_calibrated,
            "evidence_clip_path": clip_path,
            "prob_collision_2s": round(act.prob_collision_2s, 3),
            "risk_score_prob": round(act.risk_score_prob, 1)
        }

        self.db.insert_event(event_data)
        return event_data

    def _extract_evidence_clip(
        self,
        video_source_path: str,
        event_id: str,
        start_frame: int,
        end_frame: int,
        fps: float
    ) -> str:
        """Extracts short video snippet centered on conflict interval with padding."""
        padding_frames = int(round(self.clip_padding_sec * fps))
        clip_start = max(0, start_frame - padding_frames)
        clip_end = end_frame + padding_frames

        out_clip_path = self.evidence_dir / f"{event_id}_evidence.mp4"

        cap = cv2.VideoCapture(video_source_path)
        if not cap.isOpened():
            return ""

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_clip_path), fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        cur_f = clip_start

        while cap.isOpened() and cur_f <= clip_end:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            # Stamp evidence banner
            cv2.putText(
                frame,
                f"EVIDENCE CLIP: {event_id} (Frames {start_frame}-{end_frame})",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                lineType=cv2.LINE_AA
            )
            writer.write(frame)
            cur_f += 1

        cap.release()
        writer.release()
        return str(out_clip_path)
