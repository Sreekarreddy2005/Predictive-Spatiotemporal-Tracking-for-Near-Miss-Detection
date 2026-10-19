"""Unit tests for spatiotemporal conflict analysis, TTC, CPA, and PET calculations."""

import numpy as np
import pytest
from safecity.metrics.conflict_engine import ConflictAnalysisEngine


def test_head_on_collision_course():
    engine = ConflictAnalysisEngine(critical_ttc_sec=2.0)

    # Agent 1 at x = -20m moving right at 10 m/s
    # Agent 2 at x = +20m moving left at 10 m/s
    # Relative speed = 20 m/s, distance = 40m
    # Time to collision = 40m / 20m/s = 2.0s
    res = engine.analyze_pair(
        pos1=(-20.0, 0.0), vel1=(10.0, 0.0),
        pos2=(20.0, 0.0), vel2=(-10.0, 0.0),
        agent1_id=1, agent2_id=2,
        effective_radius=2.0,
        is_calibrated=True
    )

    assert res["motion_type"] == "head_on"
    assert res["approach_angle_deg"] == pytest.approx(180.0, abs=1.0)
    assert res["time_to_cpa_sec"] == pytest.approx(2.0, abs=0.05)
    assert res["distance_at_cpa"] == pytest.approx(0.0, abs=0.05)
    assert res["ttc_sec"] is not None
    # Footprint collision occurs slightly before centroid collision
    assert res["ttc_sec"] < 2.0
    assert res["is_conflict"] is True
    assert res["severity"] == "critical"


def test_diverging_vehicles_no_collision():
    engine = ConflictAnalysisEngine()

    # Agent 1 at x = 0m moving left at -10 m/s
    # Agent 2 at x = 10m moving right at +10 m/s (moving apart)
    res = engine.analyze_pair(
        pos1=(0.0, 0.0), vel1=(-10.0, 0.0),
        pos2=(10.0, 0.0), vel2=(10.0, 0.0),
        agent1_id=1, agent2_id=2,
        is_calibrated=True
    )

    assert res["motion_type"] == "diverging"
    assert res["ttc_sec"] is None
    assert res["is_conflict"] is False
    assert res["severity"] == "safe"


def test_parallel_lanes_safe_offset():
    engine = ConflictAnalysisEngine()

    # Two cars driving in parallel lanes separated by 4.0 meters
    # Agent 1 at (0, 0) moving forward at (0, 15)
    # Agent 2 at (4.0, 10) moving forward at (0, 12)
    res = engine.analyze_pair(
        pos1=(0.0, 0.0), vel1=(0.0, 15.0),
        pos2=(4.0, 10.0), vel2=(0.0, 12.0),
        agent1_id=1, agent2_id=2,
        effective_radius=1.8,
        is_calibrated=True
    )

    assert res["motion_type"] == "following"
    assert res["distance_at_cpa"] == pytest.approx(4.0, abs=0.1)
    assert res["ttc_sec"] is None  # Parallel lanes maintain safe 4m lateral offset
    assert res["is_conflict"] is False


def test_crossing_trajectory_predicted_pet():
    engine = ConflictAnalysisEngine(critical_pet_sec=1.5)

    # Crossing at origin (0, 0)
    # Agent 1 at (-20, 0) moving east at 10 m/s -> reaches origin at t = 2.0s
    # Agent 2 at (0, -30) moving north at 10 m/s -> reaches origin at t = 3.0s
    # Predicted PET = |2.0 - 3.0| = 1.0s
    res = engine.analyze_pair(
        pos1=(-20.0, 0.0), vel1=(10.0, 0.0),
        pos2=(0.0, -30.0), vel2=(0.0, 10.0),
        agent1_id=1, agent2_id=2,
        effective_radius=1.5,
        is_calibrated=True
    )

    assert res["motion_type"] == "crossing"
    assert res["predicted_pet_sec"] is not None
    assert res["predicted_pet_sec"] == pytest.approx(1.0, abs=0.1)
    assert res["is_conflict"] is True
