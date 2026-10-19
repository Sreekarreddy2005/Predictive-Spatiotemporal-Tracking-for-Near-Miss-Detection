"""SafeCity AI: Multi-Metric Adaptive Risk Fusion Network (Novelty 4).

Fuses:
1. Deterministic Kinematic Metrics (TTC, PET, CPA, DRAC)
2. Novelty 1: ST-GAT Multi-Agent Interaction Proximity
3. Novelty 2: Probabilistic Monte Carlo Collision Probabilities P_coll(1s, 2s, 3s)
4. Novelty 3: Visual Environmental Context Expansion Factor alpha_env

Produces a unified, continuous SafeCity Risk Score (0-100) and calibrated alert severity.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("safecity.metrics.risk_fusion")


class MultiMetricRiskFusion:
    """Fuses kinematic, probabilistic, and environmental risk indicators."""

    def __init__(
        self,
        critical_risk_threshold: float = 72.0,
        warning_risk_threshold: float = 48.0,
        watch_risk_threshold: float = 28.0
    ) -> None:
        self.crit_thresh = critical_risk_threshold
        self.warn_thresh = warning_risk_threshold
        self.watch_thresh = watch_risk_threshold

        # Calibrated fusion weights
        # [ttc_inv, pet_inv, drac, p_coll_1s, p_coll_2s, p_coll_3s, proximity, rel_speed, alpha_env]
        self.weights = np.array([
            0.18,  # TTC urgency
            0.10,  # PET temporal margin
            0.12,  # DRAC deceleration requirement
            0.20,  # P_coll(1.0s) immediate probability
            0.22,  # P_coll(2.0s) tactical horizon
            0.08,  # P_coll(3.0s) strategic anticipation
            0.10,  # Spatial proximity (inverse CPA)
        ], dtype=np.float32)

    def fuse_risk(
        self,
        deterministic_metrics: Dict[str, Any],
        probabilistic_metrics: Optional[Dict[str, Any]] = None,
        environmental_context: Optional[Dict[str, Any]] = None,
        stgat_min_dist: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculates unified 0-100 risk score and categorizes conflict severity."""
        ttc = deterministic_metrics.get("ttc_sec")
        pet = deterministic_metrics.get("predicted_pet_sec")
        drac = deterministic_metrics.get("drac_m_s2", 0.0) or 0.0
        cpa_d = deterministic_metrics.get("cpa_distance_m", 10.0) or 10.0
        rel_speed = deterministic_metrics.get("relative_speed", 0.0) or 0.0

        # Environmental context adjustment
        alpha_env = environmental_context.get("alpha_env_expansion_factor", 1.0) if environmental_context else 1.0

        # Probabilistic metrics
        if probabilistic_metrics:
            p_1s = probabilistic_metrics.get("prob_collision_1s", 0.0) or 0.0
            p_2s = probabilistic_metrics.get("prob_collision_2s", 0.0) or 0.0
            p_3s = probabilistic_metrics.get("prob_collision_3s", 0.0) or 0.0
            exp_ttc = probabilistic_metrics.get("expected_ttc_sec")
        else:
            p_1s = 0.0
            p_2s = 0.8 if (ttc is not None and ttc <= 2.0) else 0.0
            p_3s = 0.9 if (ttc is not None and ttc <= 3.0) else 0.0
            exp_ttc = ttc

        # ST-GAT proximity factor
        if stgat_min_dist is not None:
            eff_dist = min(cpa_d, stgat_min_dist)
        else:
            eff_dist = cpa_d

        # 1. Normalized Component Features in [0, 1]
        # Adjusted TTC urgency scaled by environmental expansion
        adj_crit_ttc = 2.0 * alpha_env
        if ttc is not None and ttc > 0:
            f_ttc = float(np.clip(1.0 - (ttc / (3.5 * alpha_env)), 0.0, 1.0))
        else:
            f_ttc = 0.0

        if pet is not None and pet > 0:
            f_pet = float(np.clip(1.0 - (pet / 2.5), 0.0, 1.0))
        else:
            f_pet = 0.0

        # DRAC: 3.4 m/s2 is comfortable braking threshold; >5.0 m/s2 is emergency
        f_drac = float(np.clip(drac / 6.0, 0.0, 1.0))

        # Collision probabilities
        f_p1 = float(np.clip(p_1s, 0.0, 1.0))
        f_p2 = float(np.clip(p_2s, 0.0, 1.0))
        f_p3 = float(np.clip(p_3s, 0.0, 1.0))

        # Proximity margin
        f_prox = float(np.clip(1.0 - (eff_dist / 6.0), 0.0, 1.0))

        # Feature vector
        x = np.array([f_ttc, f_pet, f_drac, f_p1, f_p2, f_p3, f_prox], dtype=np.float32)

        # 2. Linear combination + Non-linear Sigmoid Activation
        raw_score = float(np.dot(self.weights, x))

        # Dynamic environmental boost: in adverse weather/night, risk amplifies
        env_boost = (alpha_env - 1.0) * 0.15
        fused_prob = float(np.clip(raw_score + env_boost, 0.0, 1.0))

        # Map to calibrated 0-100 Unified Safety Index
        # Uses mild contrast sigmoid to accentuate extreme near-misses
        k = 6.0
        x0 = 0.40
        unified_score = float(1.0 / (1.0 + np.exp(-k * (fused_prob - x0)))) * 100.0
        unified_score = round(float(np.clip(unified_score, 0.0, 100.0)), 1)

        # 3. Categorized Severity Level
        if unified_score >= self.crit_thresh or p_1s >= 0.50 or (ttc is not None and ttc <= 1.2 * alpha_env):
            severity = "CRITICAL"
            is_conflict = True
        elif unified_score >= self.warn_thresh or p_2s >= 0.40 or (ttc is not None and ttc <= 2.2 * alpha_env):
            severity = "WARNING"
            is_conflict = True
        elif unified_score >= self.watch_thresh or p_3s >= 0.25:
            severity = "WATCH"
            is_conflict = False
        else:
            severity = "SAFE"
            is_conflict = False

        return {
            "unified_risk_score": unified_score,
            "severity": severity,
            "is_conflict": is_conflict,
            "features_normalized": {
                "f_ttc_urgency": round(f_ttc, 3),
                "f_pet_temporal": round(f_pet, 3),
                "f_drac_decel": round(f_drac, 3),
                "f_prob_collision_1s": round(f_p1, 3),
                "f_prob_collision_2s": round(f_p2, 3),
                "f_prob_collision_3s": round(f_p3, 3),
                "f_proximity": round(f_prox, 3)
            },
            "environmental_expansion_alpha": round(alpha_env, 2),
            "effective_ttc": round(ttc, 2) if ttc is not None else None,
            "expected_prob_ttc": round(exp_ttc, 2) if exp_ttc is not None else None
        }
