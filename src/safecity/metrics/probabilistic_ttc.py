"""SafeCity AI: Probabilistic Time-to-Collision (TTC) Engine.

Novelty Module 2:
Replaces rigid deterministic linear ray-casting with Monte Carlo Gaussian uncertainty
propagation, modeling continuous collision probability distributions P(collision <= tau)
over future time horizons tau in [1.0s, 3.0s].

Mathematical Formulation:
For agents 1 and 2 with positions p_i, velocities v_i, and covariance Sigma_i:
   p_i(t) ~ N(p_i + v_i * t + 0.5 * a_i * t^2,  Sigma_pos + t^2 * Sigma_vel)
Relative position:
   Delta p(t) ~ N(mu_rel(t), Sigma_rel(t))

Collision Probability:
   P_coll(tau) = E [ I( min_{t in [0, tau]} ||p_1(t) - p_2(t)|| <= R_crit ) ]
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("safecity.metrics.probabilistic_ttc")


class ProbabilisticTTCEngine:
    """Computes distribution-aware collision probabilities and expected TTC."""

    def __init__(
        self,
        num_particles: int = 250,
        dt: float = 0.1,
        max_horizon_sec: float = 3.0,
        default_pos_std_m: float = 0.35,
        default_vel_std_m_s: float = 0.85,
        critical_radius_m: float = 2.0,
        critical_radius_px: float = 40.0,
        random_seed: Optional[int] = 42
    ) -> None:
        self.num_particles = num_particles
        self.dt = dt
        self.max_horizon = max_horizon_sec
        self.timesteps = int(max_horizon_sec / dt)
        self.time_grid = np.linspace(dt, max_horizon_sec, self.timesteps)

        self.pos_std_m = default_pos_std_m
        self.vel_std_m_s = default_vel_std_m_s
        self.crit_radius_m = critical_radius_m
        self.crit_radius_px = critical_radius_px

        self.rng = np.random.default_rng(random_seed)

    def analyze_probabilistic_conflict(
        self,
        pos1: Tuple[float, float],
        vel1: Tuple[float, float],
        pos2: Tuple[float, float],
        vel2: Tuple[float, float],
        acc1: Optional[Tuple[float, float]] = None,
        acc2: Optional[Tuple[float, float]] = None,
        agent1_id: int = 1,
        agent2_id: int = 2,
        is_calibrated: bool = True,
        custom_pos_std: Optional[float] = None,
        custom_vel_std: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculates collision probability, expected TTC, and distance distributions.

        Args:
            pos1, pos2: (x, y) coordinates
            vel1, vel2: (vx, vy) velocity vectors
            acc1, acc2: Optional (ax, ay) acceleration vectors
            is_calibrated: True if coordinates are in real-world meters, False for image pixels
        """
        p1 = np.array(pos1, dtype=np.float64)
        p2 = np.array(pos2, dtype=np.float64)
        v1 = np.array(vel1, dtype=np.float64)
        v2 = np.array(vel2, dtype=np.float64)
        a1 = np.array(acc1, dtype=np.float64) if acc1 is not None else np.zeros(2)
        a2 = np.array(acc2, dtype=np.float64) if acc2 is not None else np.zeros(2)

        crit_r = self.crit_radius_m if is_calibrated else self.crit_radius_px
        pos_std = custom_pos_std or (self.pos_std_m if is_calibrated else 12.0)
        vel_std = custom_vel_std or (self.vel_std_m_s if is_calibrated else 25.0)

        # 1. Deterministic baseline CPA for reference
        rel_pos = p1 - p2
        rel_vel = v1 - v2
        rel_speed_sq = float(np.dot(rel_vel, rel_vel))

        if rel_speed_sq > 1e-6:
            det_t_cpa = -float(np.dot(rel_pos, rel_vel)) / rel_speed_sq
            det_d_cpa = float(np.linalg.norm(rel_pos + rel_vel * max(0.0, det_t_cpa)))
            det_ttc = det_t_cpa if (0.0 < det_t_cpa <= self.max_horizon and det_d_cpa <= crit_r) else None
        else:
            det_t_cpa = 0.0
            det_d_cpa = float(np.linalg.norm(rel_pos))
            det_ttc = None

        # 2. Vectorized Monte Carlo Propagation
        # Particles: shape (N, 2)
        n = self.num_particles
        noise_p1 = self.rng.normal(0.0, pos_std, size=(n, 2))
        noise_p2 = self.rng.normal(0.0, pos_std, size=(n, 2))
        noise_v1 = self.rng.normal(0.0, vel_std, size=(n, 2))
        noise_v2 = self.rng.normal(0.0, vel_std, size=(n, 2))

        # Sampled initial states
        p1_samples = p1 + noise_p1  # (N, 2)
        p2_samples = p2 + noise_p2  # (N, 2)
        v1_samples = v1 + noise_v1  # (N, 2)
        v2_samples = v2 + noise_v2  # (N, 2)

        # Time grid: (T, 1, 1)
        t_grid = self.time_grid.reshape(-1, 1, 1)  # (T, 1, 1)

        # Trajectories over time: (T, N, 2)
        traj1 = p1_samples.reshape(1, n, 2) + v1_samples.reshape(1, n, 2) * t_grid + 0.5 * a1.reshape(1, 1, 2) * (t_grid ** 2)
        traj2 = p2_samples.reshape(1, n, 2) + v2_samples.reshape(1, n, 2) * t_grid + 0.5 * a2.reshape(1, 1, 2) * (t_grid ** 2)

        # Pairwise euclidean distances at every timestep: (T, N)
        rel_diff = traj1 - traj2
        distances = np.linalg.norm(rel_diff, axis=2)  # (T, N)

        # Minimum distance achieved by each particle across the full horizon: (N,)
        min_dist_per_particle = np.min(distances, axis=0)
        min_time_idx_per_particle = np.argmin(distances, axis=0)
        time_to_min_dist = self.time_grid[min_time_idx_per_particle]  # (N,)

        # 3. Collision probability over horizons: 1.0s, 2.0s, 3.0s
        prob_coll = {}
        for tau in [1.0, 1.5, 2.0, 2.5, 3.0]:
            if tau <= self.max_horizon:
                tau_steps = int(tau / self.dt)
                # Min distance within tau
                min_d_tau = np.min(distances[:tau_steps, :], axis=0)
                p_val = float(np.mean(min_d_tau <= crit_r))
                prob_coll[f"P_coll_{tau:.1f}s"] = p_val

        p_overall = float(np.mean(min_dist_per_particle <= crit_r))

        # 4. Expected TTC conditioned on collision
        colliding_mask = min_dist_per_particle <= crit_r
        if np.any(colliding_mask):
            colliding_ttcs = time_to_min_dist[colliding_mask]
            expected_ttc = float(np.mean(colliding_ttcs))
            ttc_std = float(np.std(colliding_ttcs))
            ttc_p10 = float(np.percentile(colliding_ttcs, 10))
            ttc_p50 = float(np.percentile(colliding_ttcs, 50))
            ttc_p90 = float(np.percentile(colliding_ttcs, 90))
        else:
            expected_ttc = None
            ttc_std = None
            ttc_p10, ttc_p50, ttc_p90 = None, None, None

        # 5. Distance distribution quantiles (meters)
        d_mean = float(np.mean(min_dist_per_particle))
        d_p05 = float(np.percentile(min_dist_per_particle, 5))
        d_p50 = float(np.percentile(min_dist_per_particle, 50))
        d_p95 = float(np.percentile(min_dist_per_particle, 95))

        # 6. Risk Level Categorization
        p_2s = prob_coll.get("P_coll_2.0s", 0.0)
        p_1s = prob_coll.get("P_coll_1.0s", 0.0)

        if p_1s >= 0.40 or p_2s >= 0.65:
            risk_level = "CRITICAL"
        elif p_2s >= 0.35:
            risk_level = "WARNING"
        elif prob_coll.get("P_coll_3.0s", 0.0) >= 0.15:
            risk_level = "WATCH"
        else:
            risk_level = "SAFE"

        # Continuous 0-100 Probabilistic Risk Score
        # Combines probability, time urgency, and proximity margin
        urgency_factor = max(0.0, 1.0 - (expected_ttc / self.max_horizon)) if expected_ttc else 0.0
        proximity_factor = max(0.0, 1.0 - (d_p05 / (crit_r * 3.0)))
        risk_score = float(np.clip((p_2s * 0.55 + urgency_factor * 0.25 + proximity_factor * 0.20) * 100.0, 0.0, 100.0))

        return {
            "agent1_id": agent1_id,
            "agent2_id": agent2_id,
            "risk_level": risk_level,
            "risk_score_0_100": round(risk_score, 1),
            "prob_collision_overall": round(p_overall, 4),
            "prob_collision_1s": round(prob_coll.get("P_coll_1.0s", 0.0), 4),
            "prob_collision_2s": round(p_2s, 4),
            "prob_collision_3s": round(prob_coll.get("P_coll_3.0s", 0.0), 4),
            "prob_collision_horizons": prob_coll,
            "expected_ttc_sec": round(expected_ttc, 2) if expected_ttc is not None else None,
            "ttc_confidence_interval": {
                "p10": round(ttc_p10, 2) if ttc_p10 is not None else None,
                "median": round(ttc_p50, 2) if ttc_p50 is not None else None,
                "p90": round(ttc_p90, 2) if ttc_p90 is not None else None,
                "std": round(ttc_std, 2) if ttc_std is not None else None
            },
            "min_distance_distribution": {
                "mean": round(d_mean, 2),
                "p05": round(d_p05, 2),
                "median": round(d_p50, 2),
                "p95": round(d_p95, 2)
            },
            "deterministic_baseline": {
                "ttc_sec": round(det_ttc, 2) if det_ttc is not None else None,
                "d_cpa": round(det_d_cpa, 2),
                "t_cpa_sec": round(det_t_cpa, 2)
            },
            "critical_radius": crit_r,
            "num_particles": n
        }
