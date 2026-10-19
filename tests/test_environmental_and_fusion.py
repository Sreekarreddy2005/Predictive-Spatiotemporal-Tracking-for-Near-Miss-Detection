"""Unit tests for Phase 4 Novelties 3 & 4: Environmental Context & Risk Fusion."""

import pytest
import numpy as np

from safecity.metrics.environmental_context import EnvironmentalContextEncoder
from safecity.metrics.risk_fusion import MultiMetricRiskFusion


def test_environmental_context_daylight():
    """Verify that bright, clear frames yield nominal dry physics parameters."""
    encoder = EnvironmentalContextEncoder()

    # Create synthetic bright daytime frame (mean luminance ~ 150)
    day_frame = np.full((120, 160, 3), 150, dtype=np.uint8)
    # Add texture/edges for sharpness
    day_frame[::10, :] = 50
    day_frame[:, ::10] = 220

    env = encoder.analyze_frame_environment(day_frame)

    assert env["lighting_condition"] == "DAYLIGHT"
    assert env["effective_friction_mu"] >= 0.75
    assert env["alpha_env_expansion_factor"] == pytest.approx(1.0, abs=0.15)
    assert env["adjusted_critical_ttc_sec"] >= 1.9


def test_environmental_context_night_low_light():
    """Verify that dark frames expand perception time and safety stopping margins."""
    encoder = EnvironmentalContextEncoder()

    # Dark nighttime frame (mean luminance < 30)
    night_frame = np.full((120, 160, 3), 25, dtype=np.uint8)

    env = encoder.analyze_frame_environment(night_frame)

    assert env["lighting_condition"] == "NIGHT_LOW_LIGHT"
    assert env["effective_reaction_time_sec"] > encoder.t_react_nominal
    assert env["alpha_env_expansion_factor"] > 1.20
    assert env["adjusted_critical_ttc_sec"] > 2.2


def test_risk_fusion_critical_encounter():
    """Verify that imminent collision course produces CRITICAL severity and high unified score."""
    fuser = MultiMetricRiskFusion()

    det = {
        "ttc_sec": 1.1,
        "predicted_pet_sec": 0.8,
        "drac_m_s2": 4.8,
        "cpa_distance_m": 0.5,
        "relative_speed": 16.0
    }
    prob = {
        "prob_collision_1s": 0.75,
        "prob_collision_2s": 0.90,
        "prob_collision_3s": 0.95,
        "expected_ttc_sec": 1.1
    }
    env = {
        "alpha_env_expansion_factor": 1.0
    }

    res = fuser.fuse_risk(det, prob, env)

    assert res["severity"] == "CRITICAL"
    assert res["is_conflict"] is True
    assert res["unified_risk_score"] > 70.0


def test_risk_fusion_benign_safe_encounter():
    """Verify that benign encounters yield SAFE severity and low unified score."""
    fuser = MultiMetricRiskFusion()

    det = {
        "ttc_sec": None,
        "predicted_pet_sec": None,
        "drac_m_s2": 0.0,
        "cpa_distance_m": 8.5,
        "relative_speed": 4.0
    }
    prob = {
        "prob_collision_1s": 0.0,
        "prob_collision_2s": 0.02,
        "prob_collision_3s": 0.05,
        "expected_ttc_sec": None
    }
    env = {
        "alpha_env_expansion_factor": 1.0
    }

    res = fuser.fuse_risk(det, prob, env)

    assert res["severity"] == "SAFE"
    assert res["is_conflict"] is False
    assert res["unified_risk_score"] < 30.0


def test_risk_fusion_environmental_adaptation():
    """Verify that adverse weather / night amplifies borderline near-miss risk."""
    fuser = MultiMetricRiskFusion()

    # Borderline encounter (TTC = 2.4s, moderate probability)
    det = {"ttc_sec": 2.4, "predicted_pet_sec": 2.0, "drac_m_s2": 2.2, "cpa_distance_m": 2.2, "relative_speed": 12.0}
    prob = {"prob_collision_1s": 0.10, "prob_collision_2s": 0.35, "prob_collision_3s": 0.50, "expected_ttc_sec": 2.4}

    dry_env = {"alpha_env_expansion_factor": 1.0}
    adverse_env = {"alpha_env_expansion_factor": 1.45} # Wet night

    res_dry = fuser.fuse_risk(det, prob, dry_env)
    res_adverse = fuser.fuse_risk(det, prob, adverse_env)

    # Adverse environmental conditions must yield strictly higher unified risk
    assert res_adverse["unified_risk_score"] > res_dry["unified_risk_score"]
