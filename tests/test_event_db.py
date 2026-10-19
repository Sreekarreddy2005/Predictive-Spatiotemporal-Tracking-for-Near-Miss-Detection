"""Unit tests for SQLite event storage, hysteresis thresholding, and pair cooldown."""

from pathlib import Path
import sqlite3
import pytest
from safecity.storage.event_db import EventDatabase, NearMissEventManager


def test_sqlite_schema_creation(tmp_path):
    db_file = tmp_path / "test_events.db"
    db = EventDatabase(str(db_file))

    with sqlite3.connect(str(db_file)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events'")
        table = cursor.fetchone()
        assert table is not None
        assert table[0] == "events"


def test_hysteresis_and_cooldown_logic(tmp_path):
    db_file = tmp_path / "test_events.db"
    ev_dir = tmp_path / "evidence"
    manager = NearMissEventManager(
        db_path=str(db_file),
        evidence_dir=str(ev_dir),
        hysteresis_frames=3,
        cooldown_frames=10
    )

    conflict_sample = {
        "agent1_id": 1,
        "agent2_id": 2,
        "agent1_class": "car",
        "agent2_class": "car",
        "is_conflict": True,
        "severity": "critical",
        "ttc_sec": 1.2,
        "current_distance": 2.5,
        "relative_speed": 15.0,
        "approach_angle_deg": 180.0,
        "motion_type": "head_on"
    }

    # Frame 1: Conflict begins (count = 1, hysteresis threshold = 3) -> NOT alerted
    events_f1 = manager.process_frame_conflicts(frame_idx=1, conflicts=[conflict_sample])
    assert len(events_f1) == 0
    assert not manager.active_conflicts[(1, 2)].alert_triggered

    # Frame 2: Conflict persists (count = 2) -> NOT alerted
    events_f2 = manager.process_frame_conflicts(frame_idx=2, conflicts=[conflict_sample])
    assert len(events_f2) == 0
    assert not manager.active_conflicts[(1, 2)].alert_triggered

    # Frame 3: Conflict persists (count = 3 >= hysteresis) -> Alert triggered!
    events_f3 = manager.process_frame_conflicts(frame_idx=3, conflicts=[conflict_sample])
    assert manager.active_conflicts[(1, 2)].alert_triggered

    # Frame 4: Conflict clears (no conflicts reported) -> Finalized and committed to DB
    events_f4 = manager.process_frame_conflicts(frame_idx=4, conflicts=[])
    assert len(events_f4) == 1
    assert events_f4[0]["event_id"] == "EVT_0001"
    assert events_f4[0]["min_ttc"] == 1.2

    # Frame 5: New conflict appears immediately, but pair (1, 2) is in cooldown until frame 14!
    events_f5 = manager.process_frame_conflicts(frame_idx=5, conflicts=[conflict_sample])
    assert (1, 2) not in manager.active_conflicts  # Suppressed by cooldown
