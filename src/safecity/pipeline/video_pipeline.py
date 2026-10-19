"""SafeCity AI: Master Video Processing & Conflict Analysis Pipeline.

Orchestrates:
1. Video frame reading & preprocessing
2. YOLOv8 object detection (cars, buses, trucks, pedestrians, bikes)
3. ByteTrack multi-object tracking with persistent IDs
4. Camera calibration & planar homography (image pixels <-> world meters)
5. Constant-velocity trajectory forecasting
6. Spatiotemporal conflict engine (TTC, CPA, PET, DRAC)
7. Temporal hysteresis & SQLite event database logging
8. Visual rendering of predictions, hazard warnings, and video export
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

from safecity.calibration.homography import HomographyCalibrator
from safecity.prediction.constant_velocity import ConstantVelocityPredictor
from safecity.prediction.lstm_predictor import LSTMTrajectoryPredictor
from safecity.prediction.interaction_predictor import InteractionTrajectoryPredictor
from safecity.metrics.conflict_engine import ConflictAnalysisEngine
from safecity.metrics.probabilistic_ttc import ProbabilisticTTCEngine
from safecity.metrics.environmental_context import EnvironmentalContextEncoder
from safecity.metrics.risk_fusion import MultiMetricRiskFusion
from safecity.storage.event_db import NearMissEventManager
from safecity.tracking.tracker import TrafficTracker

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("safecity.pipeline")

# Aesthetic Color Palette (BGR)
CLASS_COLORS = {
    "car": (0, 165, 255),         # Vivid Orange
    "pedestrian": (255, 105, 180), # Neon Pink
    "bicycle": (50, 205, 50),     # Lime Green
    "motorcycle": (0, 215, 255),  # Gold
    "bus": (255, 0, 0),           # Bright Blue
    "truck": (180, 105, 255),     # Violet
    "conflict": (0, 0, 255),      # Bright Red (Emergency)
    "default": (0, 255, 127)      # Spring Green
}


def draw_styled_track(
    frame: np.ndarray,
    track: Dict[str, Any],
    predicted_points: Optional[np.ndarray] = None,
    speed_kmh: Optional[float] = None,
    is_conflict: bool = False,
    conflict_msg: str = ""
) -> None:
    """Renders vehicle bounding box, persistent ID, historical trail, and predicted future trajectory."""
    x1, y1, x2, y2 = [int(v) for v in track["bbox"]]
    track_id = track["track_id"]
    cls_name = track["class_name"]
    conf = track["conf"]

    color = CLASS_COLORS["conflict"] if is_conflict else CLASS_COLORS.get(cls_name, CLASS_COLORS["default"])

    # 1. Historical Ground Contact Trail (past trajectory)
    if "trail" in track and len(track["trail"]) > 1:
        points = np.array([[int(pt[0]), int(pt[1])] for pt in track["trail"]], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
        current_pt = (int(track["trail"][-1][0]), int(track["trail"][-1][1]))
        cv2.circle(frame, current_pt, 4, color, -1, lineType=cv2.LINE_AA)

    # 2. Predicted Future Trajectory (future trajectory forecast)
    if predicted_points is not None and len(predicted_points) > 1:
        pred_pts = np.array([[int(pt[0]), int(pt[1])] for pt in predicted_points], np.int32).reshape((-1, 1, 2))
        # Draw dotted future trajectory line
        for p in pred_pts:
            cv2.circle(frame, (int(p[0][0]), int(p[0][1])), 2, (255, 255, 0) if not is_conflict else (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [pred_pts], isClosed=False, color=(255, 255, 0) if not is_conflict else (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    # 3. Bounding box
    thickness = 3 if is_conflict else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

    # 4. Header Badge
    speed_str = f" {speed_kmh:.1f} km/h" if speed_kmh is not None else ""
    if is_conflict:
        label = f"⚠️ ID:{track_id} {conflict_msg}"
    else:
        label = f"ID:{track_id} {cls_name}{speed_str} ({conf:.2f})"

    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    tag_y1 = max(0, y1 - label_h - 6)
    tag_y2 = y1
    tag_x2 = min(frame.shape[1], x1 + label_w + 8)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, label, (x1 + 4, tag_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0) if not is_conflict else (255, 255, 255), 1, lineType=cv2.LINE_AA)


def run_pipeline(
    video_path: str,
    output_video_path: str = "outputs/annotated_traffic.mp4",
    summary_path: str = "outputs/tracking_summary.json",
    calibration_path: Optional[str] = "configs/camera_calibration.yaml",
    db_path: str = "outputs/events.db",
    model_name: str = "yolov8n.pt",
    predictor_type: str = "constant_velocity",
    lstm_checkpoint: str = "models/lstm_baseline.pt",
    stgat_checkpoint: str = "models/interaction_stgat.pt",
    conf_thresh: float = 0.35,
    max_frames: Optional[int] = None,
    device: str = "auto"
) -> Dict[str, Any]:
    """Runs complete SafeCity AI detection, tracking, prediction, and near-miss conflict analysis."""
    input_file = Path(video_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    out_dir = Path(output_video_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Modules
    tracker = TrafficTracker(model_name=model_name, conf_thresh=conf_thresh, device=device)
    if predictor_type == "stgat" and Path(stgat_checkpoint).exists():
        predictor = InteractionTrajectoryPredictor(checkpoint_path=stgat_checkpoint, device=device)
        logger.info(f"Loaded ST-GAT Interaction-Aware predictor from {stgat_checkpoint}")
    elif predictor_type == "lstm" and Path(lstm_checkpoint).exists():
        predictor = LSTMTrajectoryPredictor(checkpoint_path=lstm_checkpoint, device=device)
        logger.info(f"Loaded Seq2Seq LSTM predictor from {lstm_checkpoint}")
    else:
        predictor = ConstantVelocityPredictor(observation_window_sec=2.0, prediction_horizon_sec=2.5, step_size_sec=0.25)
        logger.info("Using analytical Constant-Velocity predictor")
    conflict_engine = ConflictAnalysisEngine(critical_ttc_sec=2.0, warning_ttc_sec=3.0, critical_pet_sec=1.5)
    prob_engine = ProbabilisticTTCEngine(num_particles=150, dt=0.1, max_horizon_sec=3.0)
    env_encoder = EnvironmentalContextEncoder()
    risk_fuser = MultiMetricRiskFusion()
    event_manager = NearMissEventManager(db_path=db_path, evidence_dir=str(out_dir / "evidence"))

    # Load Calibration if available
    calibrator = HomographyCalibrator()
    if calibration_path and Path(calibration_path).exists():
        try:
            calibrator = HomographyCalibrator.from_yaml(calibration_path)
            logger.info(f"Loaded camera calibration from {calibration_path} (is_calibrated={calibrator.is_calibrated})")
        except Exception as e:
            logger.warning(f"Could not load calibration from {calibration_path}: {e}")

    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dt = 1.0 / fps if fps > 0 else 0.04
    target_frames = min(total_frames, max_frames) if max_frames else total_frames

    logger.info(f"Processing '{input_file.name}' | {width}x{height} @ {fps:.1f} FPS | Calibrated: {calibrator.is_calibrated}")

    temp_output_path = str(out_dir / "temp_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    frame_idx = 0
    start_time = time.time()
    all_seen_ids = set()
    total_alerts_count = 0

    pbar = tqdm(total=target_frames, desc="Processing Video Pipeline")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None or (max_frames and frame_idx >= max_frames):
                break

            # 0. Novelty 3: Environmental Context Analysis
            env_info = env_encoder.analyze_frame_environment(frame)

            # 1. Detection + Tracking
            active_tracks = tracker.track_frame(frame, frame_idx=frame_idx, dt=dt)
            track_dict = {}

            for trk in active_tracks:
                tid = trk["track_id"]
                all_seen_ids.add(tid)
                track_dict[tid] = trk

            # 2. Trajectory Prediction & Speed Estimation
            predictions = {}
            speeds_kmh = {}
            active_ids = list(track_dict.keys())
            for tid, trk in track_dict.items():
                trail = trk["trail"]
                if predictor_type == "stgat":
                    other_trails = [track_dict[oid]["trail"] for oid in active_ids if oid != tid]
                    future_pts = predictor.predict(trail, neighbor_trails=other_trails, fps=fps)
                else:
                    future_pts = predictor.predict(trail, fps=fps)
                predictions[tid] = future_pts
                speeds_kmh[tid] = calibrator.estimate_speed_kmh(trail, fps=fps)

            # 3. Spatiotemporal Conflict Analysis Across Active Pairs
            frame_conflicts = []
            active_ids = list(track_dict.keys())
            conflict_flags = {}

            for i in range(len(active_ids)):
                for j in range(i + 1, len(active_ids)):
                    id1 = active_ids[i]
                    id2 = active_ids[j]
                    trk1 = track_dict[id1]
                    trk2 = track_dict[id2]

                    # Ground contact points
                    pos1 = trk1["trail"][-1]
                    pos2 = trk2["trail"][-1]
                    vel1 = trk1["velocity"]
                    vel2 = trk2["velocity"]

                    # If calibrated, convert points and velocities to world meters
                    is_calib = calibrator.is_calibrated
                    if is_calib:
                        w_pos1 = calibrator.image_to_world(pos1)
                        w_pos2 = calibrator.image_to_world(pos2)
                        if w_pos1 is not None and w_pos2 is not None:
                            pos1 = tuple(w_pos1)
                            pos2 = tuple(w_pos2)
                            # Convert vel from px/s to m/s
                            vel1 = (vel1[0] * 0.05, vel1[1] * 0.05)
                            vel2 = (vel2[0] * 0.05, vel2[1] * 0.05)

                    res = conflict_engine.analyze_pair(
                        pos1=pos1, vel1=vel1,
                        pos2=pos2, vel2=vel2,
                        agent1_id=id1, agent2_id=id2,
                        agent1_class=trk1["class_name"],
                        agent2_class=trk2["class_name"],
                        is_calibrated=is_calib
                    )

                    # Novelty 2: Probabilistic Monte Carlo TTC Evaluation
                    prob_res = prob_engine.analyze_probabilistic_conflict(
                        pos1=pos1, vel1=vel1,
                        pos2=pos2, vel2=vel2,
                        agent1_id=id1, agent2_id=id2,
                        is_calibrated=is_calib
                    )
                    res["probabilistic_ttc"] = prob_res

                    # Novelty 4: Multi-Metric Adaptive Risk Fusion
                    fused_res = risk_fuser.fuse_risk(
                        deterministic_metrics=res,
                        probabilistic_metrics=prob_res,
                        environmental_context=env_info
                    )
                    res["fused_risk"] = fused_res

                    # Conflict threshold: adaptive fused risk alert
                    is_hazard = fused_res["is_conflict"] or res["is_conflict"] or prob_res["risk_level"] in ["WARNING", "CRITICAL"]
                    if is_hazard:
                        res["is_conflict"] = True
                        res["severity"] = fused_res["severity"].lower()
                        frame_conflicts.append(res)
                        u_score = fused_res["unified_risk_score"]
                        ttc_val = res.get("ttc_sec")
                        if ttc_val is not None:
                            msg = f"{fused_res['severity'][:4]} R:{u_score:.0f}% T:{ttc_val:.1f}s"
                        else:
                            msg = f"{fused_res['severity'][:4]} R:{u_score:.0f}%"
                        conflict_flags[id1] = msg
                        conflict_flags[id2] = msg

                        # Draw hazard connecting line between conflicting agents
                        c1 = (int(trk1["trail"][-1][0]), int(trk1["trail"][-1][1]))
                        c2 = (int(trk2["trail"][-1][0]), int(trk2["trail"][-1][1]))
                        cv2.line(frame, c1, c2, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # 4. Hysteresis & Event Database Logging
            new_events = event_manager.process_frame_conflicts(
                frame_idx=frame_idx,
                conflicts=frame_conflicts,
                video_source_path=str(input_file),
                fps=fps
            )
            total_alerts_count += len(new_events)

            # 5. Render Visuals on Frame
            for tid, trk in track_dict.items():
                is_conf = tid in conflict_flags
                c_msg = conflict_flags.get(tid, "")
                draw_styled_track(
                    frame=frame,
                    track=trk,
                    predicted_points=predictions.get(tid),
                    speed_kmh=speeds_kmh.get(tid),
                    is_conflict=is_conf,
                    conflict_msg=c_msg
                )

            # 6. Top HUD Banner
            calib_str = "CALIB (km/h)" if calibrator.is_calibrated else "IMAGE (px)"
            env_str = f"Env: {env_info['lighting_condition'][:5]} x{env_info['alpha_env_expansion_factor']:.2f}"
            hud_text = f"Frame: {frame_idx:04d}/{target_frames} | Active: {len(active_tracks)} | {env_str} | Hazards: {len(frame_conflicts)} | {calib_str}"
            cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, 36), (20, 20, 20), -1)
            hud_color = (0, 0, 255) if frame_conflicts else (255, 255, 255)
            cv2.putText(frame, hud_text, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, hud_color, 1, lineType=cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

    finally:
        cap.release()
        writer.release()
        pbar.close()

    # Flush any trailing events
    event_manager.flush_remaining_events(final_frame_idx=frame_idx, video_source_path=str(input_file), fps=fps)

    elapsed = time.time() - start_time
    proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0

    # Universal H.264 Transcode with FFmpeg
    try:
        cmd = [
            "ffmpeg", "-y", "-i", temp_output_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            output_video_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        Path(temp_output_path).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"FFmpeg transcode fallback: {e}")
        Path(temp_output_path).replace(output_video_path)

    summary = {
        "input_video": str(video_path),
        "output_video": str(output_video_path),
        "total_frames_processed": int(frame_idx),
        "video_fps": float(fps),
        "processing_fps": float(proc_fps),
        "elapsed_seconds": float(elapsed),
        "total_unique_tracks": len(all_seen_ids),
        "unique_track_ids": [int(x) for x in sorted(list(all_seen_ids))],
        "total_near_miss_events_logged": total_alerts_count,
        "is_calibrated": calibrator.is_calibrated
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✅ Pipeline complete in {elapsed:.2f}s ({proc_fps:.1f} FPS). Logged {total_alerts_count} near-miss events.")
    logger.info(f"Output video: {output_video_path}")
    logger.info(f"Events database: {db_path}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SafeCity AI Pipeline with Calibration & Conflict Analysis")
    parser.add_argument("--source", type=str, default="data/samples/cctv/sample_traffic.mp4")
    parser.add_argument("--output", type=str, default="outputs/annotated_traffic.mp4")
    parser.add_argument("--summary", type=str, default="outputs/tracking_summary.json")
    parser.add_argument("--calibration", type=str, default="configs/camera_calibration.yaml")
    parser.add_argument("--db", type=str, default="outputs/events.db")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    run_pipeline(
        video_path=args.source,
        output_video_path=args.output,
        summary_path=args.summary,
        calibration_path=args.calibration,
        db_path=args.db,
        model_name=args.model,
        conf_thresh=args.conf,
        max_frames=args.max_frames
    )
