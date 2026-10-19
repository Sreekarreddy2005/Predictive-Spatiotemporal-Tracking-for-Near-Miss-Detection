"""Unit tests for Phase 4 Novelty 2: Probabilistic Time-to-Collision (TTC) Engine."""

import pytest
import numpy as np

from safecity.metrics.probabilistic_ttc import ProbabilisticTTCEngine


def test_probabilistic_ttc_head_on_conflict():
    """Verify that two vehicles on direct collision course yield high collision probability."""
    engine = ProbabilisticTTCEngine(num_particles=200, dt=0.1, max_horizon_sec=3.0, random_seed=42)

    # Vehicles at distance 24m approaching at 6 m/s each -> expected collision in 2.0s
    pos1 = (0.0, 0.0)
    vel1 = (6.0, 0.0)
    pos2 = (24.0, 0.0)
    vel2 = (-6.0, 0.0)

    res = engine.analyze_probabilistic_conflict(pos1, vel1, pos2, vel2, is_calibrated=True)

    assert res["risk_level"] in ["WARNING", "CRITICAL"]
    assert res["prob_collision_2s"] > 0.40
    assert res["prob_collision_3s"] > 0.50
    assert res["expected_ttc_sec"] is not None
    assert 1.5 <= res["expected_ttc_sec"] <= 2.5
    assert res["risk_score_0_100"] > 40.0

    # Verify confidence intervals
    ci = res["ttc_confidence_interval"]
    assert ci["p10"] is not None
    assert ci["median"] is not None
    assert ci["p90"] is not None
    assert ci["p10"] <= ci["median"] <= ci["p90"]


def test_probabilistic_ttc_parallel_benign_motion():
    """Verify that two vehicles driving parallel in adjacent lanes do not trigger false alarms."""
    engine = ProbabilisticTTCEngine(num_particles=200, dt=0.1, max_horizon_sec=3.0, random_seed=42)

    # Parallel highway lanes with 3.5m lateral offset
    pos1 = (0.0, 0.0)
    vel1 = (18.0, 0.0)
    pos2 = (5.0, 3.5)
    vel2 = (18.0, 0.0)

    res = engine.analyze_probabilistic_conflict(pos1, vel1, pos2, vel2, is_calibrated=True)

    assert res["risk_level"] == "SAFE"
    assert res["prob_collision_2s"] < 0.15
    assert res["prob_collision_1s"] == 0.0
    assert res["risk_score_0_100"] < 35.0


def test_probabilistic_ttc_distance_percentiles():
    """Verify minimum distance percentiles are monotonically non-decreasing."""
    engine = ProbabilisticTTCEngine(num_particles=150, dt=0.1, max_horizon_sec=3.0, random_seed=123)

    pos1 = (0.0, 0.0)
    vel1 = (10.0, 2.0)
    pos2 = (15.0, 5.0)
    vel2 = (0.0, -5.0)

    res = engine.analyze_probabilistic_conflict(pos1, vel1, pos2, vel2, is_calibrated=True)

    dist_dist = res["min_distance_distribution"]
    assert dist_dist["p05"] <= dist_dist["median"] <= dist_dist["p95"]


def test_probabilistic_ttc_uncalibrated_pixels():
    """Verify that uncalibrated pixel coordinates use pixel thresholds and operate smoothly."""
    engine = ProbabilisticTTCEngine(num_particles=150, dt=0.1, max_horizon_sec=3.0, random_seed=42)

    pos1 = (100.0, 200.0)
    vel1 = (30.0, 0.0)
    pos2 = (180.0, 200.0)
    vel2 = (-30.0, 0.0)

    res = engine.analyze_probabilistic_conflict(pos1, vel1, pos2, vel2, is_calibrated=False)

    assert "risk_level" in res
    assert "prob_collision_horizons" in res
    assert res["critical_radius"] == engine.crit_radius_px
