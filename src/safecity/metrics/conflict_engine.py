"""SafeCity AI: Spatiotemporal Conflict & Near-Miss Metrics Engine.

Implements mathematically rigorous definitions of:
- Time to Collision (TTC)
- Time & Distance at Closest Point of Approach (CPA)
- Predicted Post-Encroachment Time (PET)
- Deceleration Rate to Avoid Collision (DRAC)
- Relative Speed, Approach Angle, and Separation Distance
Handles crossing, following, head-on, parallel, and diverging geometries.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("safecity.metrics")


class ConflictAnalysisEngine:
    """Computes deterministic traffic conflict indicators between pairs of road users."""

    def __init__(
        self,
        critical_ttc_sec: float = 2.0,
        warning_ttc_sec: float = 3.0,
        critical_pet_sec: float = 1.5,
        critical_separation_meters: float = 1.8,
        critical_separation_pixels: float = 35.0,
        critical_drac_m_s2: float = 3.5
    ) -> None:
        self.critical_ttc = critical_ttc_sec
        self.warning_ttc = warning_ttc_sec
        self.critical_pet = critical_pet_sec
        self.critical_sep_m = critical_separation_meters
        self.critical_sep_px = critical_separation_pixels
        self.critical_drac = critical_drac_m_s2

    def analyze_pair(
        self,
        pos1: Tuple[float, float],
        vel1: Tuple[float, float],
        pos2: Tuple[float, float],
        vel2: Tuple[float, float],
        agent1_id: int,
        agent2_id: int,
        agent1_class: str = "car",
        agent2_class: str = "car",
        effective_radius: float = 1.5,
        is_calibrated: bool = False
    ) -> Dict[str, Any]:
        """Performs spatiotemporal conflict analysis for an agent pair at a single timestamp.

        pos1, pos2: (x, y) coordinates (meters if is_calibrated, else pixels)
        vel1, vel2: (vx, vy) velocities per second
        """
        p1 = np.array(pos1, dtype=np.float64)
        p2 = np.array(pos2, dtype=np.float64)
        v1 = np.array(vel1, dtype=np.float64)
        v2 = np.array(vel2, dtype=np.float64)

        # 1. Current spatial separation
        displacement = p1 - p2
        current_distance = float(np.linalg.norm(displacement))

        # 2. Relative velocity & speed
        rel_vel = v1 - v2
        rel_speed = float(np.linalg.norm(rel_vel))

        # 3. Approach Angle (degrees)
        speed1 = float(np.linalg.norm(v1))
        speed2 = float(np.linalg.norm(v2))
        approach_angle_deg = self._compute_approach_angle(v1, v2, speed1, speed2)
        motion_type = self._classify_motion(v1, v2, displacement, approach_angle_deg, speed1, speed2)

        # 4. Closest Point of Approach (CPA)
        t_cpa, d_cpa = self._compute_cpa(displacement, rel_vel, current_distance)

        # 5. Time to Collision (TTC)
        ttc = self._compute_ttc(displacement, rel_vel, effective_radius, t_cpa, d_cpa)

        # 6. Predicted PET (for crossing trajectories)
        predicted_pet = self._compute_predicted_pet(p1, v1, p2, v2, approach_angle_deg)

        # 7. DRAC (Deceleration Rate to Avoid Collision)
        drac = self._compute_drac(current_distance, displacement, rel_vel, effective_radius, is_calibrated)

        # 8. Determine Conflict Status & Severity
        crit_sep = self.critical_sep_m if is_calibrated else self.critical_sep_px
        is_conflict = False
        severity = "safe"

        if ttc is not None and ttc <= self.critical_ttc:
            is_conflict = True
            severity = "critical"
        elif ttc is not None and ttc <= self.warning_ttc:
            is_conflict = True
            severity = "warning"
        elif t_cpa > 0 and t_cpa <= self.critical_ttc and d_cpa <= crit_sep:
            is_conflict = True
            severity = "critical"
        elif predicted_pet is not None and predicted_pet <= self.critical_pet and (t_cpa > 0 and t_cpa <= 3.0):
            is_conflict = True
            severity = "warning"

        return {
            "agent1_id": agent1_id,
            "agent2_id": agent2_id,
            "agent1_class": agent1_class,
            "agent2_class": agent2_class,
            "is_calibrated": is_calibrated,
            "current_distance": current_distance,
            "relative_speed": rel_speed,
            "approach_angle_deg": approach_angle_deg,
            "motion_type": motion_type,
            "time_to_cpa_sec": t_cpa,
            "distance_at_cpa": d_cpa,
            "ttc_sec": ttc,
            "predicted_pet_sec": predicted_pet,
            "drac_m_s2": drac,
            "is_conflict": is_conflict,
            "severity": severity
        }

    @staticmethod
    def _compute_approach_angle(v1: np.ndarray, v2: np.ndarray, s1: float, s2: float) -> float:
        """Computes orientation angle between velocity vectors in [0, 180] degrees."""
        if s1 < 0.1 or s2 < 0.1:
            return 0.0
        cos_theta = np.dot(v1, v2) / (s1 * s2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    @staticmethod
    def _classify_motion(
        v1: np.ndarray,
        v2: np.ndarray,
        disp: np.ndarray,
        angle_deg: float,
        s1: float,
        s2: float
    ) -> str:
        """Classifies geometric encounter into distinct interaction regimes."""
        if s1 < 0.2 and s2 < 0.2:
            return "stationary"
        # Closing rate: negative derivative of distance indicates closing in
        closing_rate = -np.dot(disp, v1 - v2)
        if closing_rate < -0.1:
            return "diverging"
        if angle_deg > 135.0:
            return "head_on"
        elif angle_deg < 35.0:
            return "following"
        elif 45.0 <= angle_deg <= 135.0:
            return "crossing"
        return "parallel"

    @staticmethod
    def _compute_cpa(disp: np.ndarray, rel_vel: np.ndarray, curr_dist: float) -> Tuple[float, float]:
        """Calculates Time to CPA (seconds) and Distance at CPA."""
        v_norm_sq = float(np.dot(rel_vel, rel_vel))
        if v_norm_sq < 1e-6:
            # Relative speed is near zero: currently at CPA
            return 0.0, curr_dist

        # t_cpa = - (r . v_rel) / ||v_rel||^2
        t_cpa = - float(np.dot(disp, rel_vel)) / v_norm_sq

        if t_cpa <= 0.0:
            # Vehicles are diverging; closest distance was in past or right now
            return 0.0, curr_dist

        cpa_vector = disp + rel_vel * t_cpa
        d_cpa = float(np.linalg.norm(cpa_vector))
        return t_cpa, d_cpa

    @staticmethod
    def _compute_ttc(
        disp: np.ndarray,
        rel_vel: np.ndarray,
        r_eff: float,
        t_cpa: float,
        d_cpa: float
    ) -> Optional[float]:
        """Analytically computes Time to Collision (TTC) using footprint proximity radius.

        Returns None if paths do not collide within finite time or are diverging.
        """
        # If distance at closest approach is greater than effective collision boundary, no collision occurs
        if d_cpa > r_eff or t_cpa <= 0:
            return None

        # Solve: ||disp + rel_vel * t||^2 = r_eff^2
        # a*t^2 + b*t + c = 0
        a = float(np.dot(rel_vel, rel_vel))
        if a < 1e-6:
            return None

        b = 2.0 * float(np.dot(disp, rel_vel))
        c = float(np.dot(disp, disp)) - (r_eff ** 2)

        if c <= 0:
            # Footprints are already overlapping
            return 0.0

        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        # Return earliest positive root
        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        return None

    @staticmethod
    def _compute_predicted_pet(
        p1: np.ndarray,
        v1: np.ndarray,
        p2: np.ndarray,
        v2: np.ndarray,
        angle_deg: float
    ) -> Optional[float]:
        """Calculates predicted PET at geometric trajectory intersection point for crossing paths."""
        if angle_deg < 25.0 or angle_deg > 155.0:
            # Not a crossing geometry
            return None

        # Find 2D line intersection: p1 + v1*t1 = p2 + v2*t2
        # Matrix equation: [v1, -v2] [t1, t2]^T = p2 - p1
        M = np.column_stack((v1, -v2))
        if abs(np.linalg.det(M)) < 1e-4:
            return None

        rhs = p2 - p1
        try:
            t_vals = np.linalg.solve(M, rhs)
            t1, t2 = float(t_vals[0]), float(t_vals[1])
            if t1 > 0 and t2 > 0:
                # Both vehicles reach the conflict point in the future
                return abs(t1 - t2)
        except np.linalg.LinAlgError:
            pass

        return None

    @staticmethod
    def _compute_drac(
        curr_dist: float,
        disp: np.ndarray,
        rel_vel: np.ndarray,
        r_eff: float,
        is_calibrated: bool
    ) -> Optional[float]:
        """Computes Deceleration Rate to Avoid Collision (DRAC) in m/s^2.

        DRAC is only physically defined when world coordinates are calibrated in meters.
        """
        if not is_calibrated or curr_dist <= r_eff:
            return None

        # Closing speed: projection of relative velocity along line of sight
        closing_speed = - float(np.dot(disp, rel_vel)) / curr_dist
        if closing_speed <= 0.1:
            return 0.0  # Not closing in

        clearance = curr_dist - r_eff
        if clearance <= 0:
            return None

        drac = (closing_speed ** 2) / (2.0 * clearance)
        return float(np.clip(drac, 0.0, 20.0))
