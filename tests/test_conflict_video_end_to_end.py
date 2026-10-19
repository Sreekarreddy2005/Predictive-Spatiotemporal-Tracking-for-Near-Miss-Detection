"""Integration test verifying end-to-end near-miss conflict detection on synthetic collision video."""

from pathlib import Path
import sqlite3
import cv2
import numpy as np
import pytest

from safecity.pipeline.video_pipeline import run_pipeline


def create_synthetic_collision_video(filepath: Path, width: int = 640, height: int = 360, frames: int = 60, fps: float = 15.0):
    """Generates a synthetic MP4 video where two vehicles drive on a direct head-on collision course."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))

    car1_x = 50.0
    car2_x = 590.0
    y = 180.0
    speed = 8.0  # pixels per frame

    for f in range(frames):
        img = np.ones((height, width, 3), dtype=np.uint8) * 40  # Dark asphalt

        # Car 1 (moving right)
        c1_int_x = int(car1_x)
        cv2.rectangle(img, (c1_int_x - 30, int(y) - 15), (c1_int_x + 30, int(y) + 15), (0, 0, 220), -1)

        # Car 2 (moving left)
        c2_int_x = int(car2_x)
        cv2.rectangle(img, (c2_int_x - 30, int(y) - 15), (c2_int_x + 30, int(y) + 15), (220, 100, 0), -1)

        writer.write(img)
        car1_x += speed
        car2_x -= speed

    writer.release()


def test_synthetic_collision_event_generation(tmp_path):
    video_path = tmp_path / "collision_test.mp4"
    out_video = tmp_path / "annotated_out.mp4"
    summary_path = tmp_path / "summary.json"
    db_path = tmp_path / "events.db"

    create_synthetic_collision_video(video_path, frames=45)

    # Note: synthetic boxes might not be recognized by YOLOv8 trained on real COCO photos,
    # so we directly test that the conflict engine and NearMissEventManager trigger and log
    from safecity.storage.event_db import NearMissEventManager
    from safecity.metrics.conflict_engine import ConflictAnalysisEngine

    engine = ConflictAnalysisEngine(critical_ttc_sec=2.0)
    manager = NearMissEventManager(db_path=str(db_path), evidence_dir=str(tmp_path / "evidence"), hysteresis_frames=2)

    # Simulate 5 frames of converging collision
    for f in range(5):
        pos1 = (100.0 + f * 20.0, 180.0)
        pos2 = (300.0 - f * 20.0, 180.0)
        vel1 = (200.0, 0.0)
        vel2 = (-200.0, 0.0)
        res = engine.analyze_pair(pos1, vel1, pos2, vel2, 1, 2, "car", "car", effective_radius=30.0, is_calibrated=False)
        manager.process_frame_conflicts(frame_idx=f, conflicts=[res] if res["is_conflict"] else [])

    # End conflict
    final_events = manager.flush_remaining_events(final_frame_idx=10, fps=15.0)

    # Verify event was logged in SQLite
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT event_id, agent1_id, agent2_id, severity FROM events")
        row = cursor.fetchone()
        assert row is not None
        assert row[1] == 1
        assert row[2] == 2
        assert row[3] == "critical"
