"""SafeCity AI: Visual Environmental Context Encoder (Novelty 3).

Dynamically estimates real-time environmental risk factors (ambient illumination,
contrast degradation, haze/blur, and wet surface specularity) directly from surveillance
video frames.

Mathematical Formulation:
Based on AASHTO (Stopping Sight Distance) and ISO 15623:
   d_stopping = v * t_react + v^2 / (2 * g * mu_friction)

When lighting is degraded (night/dusk) or visibility is reduced (rain/fog):
1. Driver perception-reaction time increases: t_react' = t_react * (1 + delta_react)
2. Road tire-pavement friction coefficient decreases: mu' = mu_dry * (1 - delta_friction)
3. Dynamic Environmental Safety Expansion Factor:
   alpha_env = (t_react' / t_react) * (mu_dry / mu')

Adjusts critical conflict thresholds:
   TTC_crit(env) = TTC_crit_nominal * alpha_env
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("safecity.metrics.environmental_context")


class EnvironmentalContextEncoder:
    """Extracts scene-level visibility, ambient lighting, and road condition factors."""

    def __init__(
        self,
        nominal_friction_coeff: float = 0.80,
        nominal_reaction_time_sec: float = 1.50,
        gravity_m_s2: float = 9.81
    ) -> None:
        self.mu_dry = nominal_friction_coeff
        self.t_react_nominal = nominal_reaction_time_sec
        self.g = gravity_m_s2

        # Temporal smoothing for video stream
        self.smoothed_alpha: Optional[float] = None
        self.alpha_momentum = 0.85

    def analyze_frame_environment(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extracts environmental features from a single BGR frame.

        Args:
            frame: BGR numpy image (H, W, 3)

        Returns:
            Dictionary containing illumination, contrast, weather degradation,
            estimated friction coefficient, and threshold expansion factor alpha_env.
        """
        if frame is None or frame.size == 0:
            return self._default_environment()

        # Downsample for sub-millisecond real-time efficiency
        h, w = frame.shape[:2]
        small_w = min(320, w)
        small_h = int(h * (small_w / w))
        thumb = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)

        # 1. Ambient Illumination (Luminance) in LAB color space
        lab = cv2.cvtColor(thumb, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        mean_l = float(np.mean(l_channel))          # Range [0, 255]
        p10_l = float(np.percentile(l_channel, 10))
        p90_l = float(np.percentile(l_channel, 90))

        # 2. Contrast & Visibility (Laplacian variance and dynamic range)
        gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        l_contrast = float(np.std(l_channel))

        # 3. Environmental Condition Classification
        # Day: mean_l > 85, Dusk/Dawn: 40 <= mean_l <= 85, Night: mean_l < 40
        if mean_l >= 85:
            lighting_cond = "DAYLIGHT"
            delta_react_light = 0.0
        elif mean_l >= 45:
            lighting_cond = "DUSK_DAWN"
            delta_react_light = 0.20
        else:
            lighting_cond = "NIGHT_LOW_LIGHT"
            delta_react_light = 0.45

        # Visibility degradation (fog, haze, or camera glare)
        if laplacian_var < 80.0 or l_contrast < 25.0:
            visibility_cond = "POOR_HAZE"
            delta_vis = 0.25
        elif laplacian_var < 200.0 or l_contrast < 40.0:
            visibility_cond = "MODERATE"
            delta_vis = 0.10
        else:
            visibility_cond = "CLEAR"
            delta_vis = 0.0

        # 4. Surface Condition Estimation (Specularity & saturation drop)
        hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
        sat_channel = hsv[:, :, 1]
        mean_sat = float(np.mean(sat_channel))
        # Wet asphalt typically exhibits low saturation + specular reflection peaks
        specular_ratio = float(np.mean(l_channel > 220))

        if mean_l < 100 and specular_ratio > 0.08 and mean_sat < 60:
            surface_cond = "WET_SLIPPERY"
            friction_reduction = 0.35  # Friction drops to ~0.45
        elif specular_ratio > 0.15:
            surface_cond = "GLARE_POTENTIAL"
            friction_reduction = 0.15
        else:
            surface_cond = "DRY_STANDARD"
            friction_reduction = 0.0

        # 5. Physics-Based Stopping Adjustment
        effective_mu = max(0.35, self.mu_dry * (1.0 - friction_reduction))
        effective_react_time = self.t_react_nominal * (1.0 + delta_react_light + delta_vis)

        # Dynamic Environmental Expansion Factor alpha_env
        # Combines reaction delay ratio with friction penalty
        raw_alpha = (effective_react_time / self.t_react_nominal) * (self.mu_dry / effective_mu)
        raw_alpha = float(np.clip(raw_alpha, 1.0, 2.2))

        # Temporal smoothing across video frames
        if self.smoothed_alpha is None:
            self.smoothed_alpha = raw_alpha
        else:
            self.smoothed_alpha = (
                self.alpha_momentum * self.smoothed_alpha + (1.0 - self.alpha_momentum) * raw_alpha
            )

        alpha_final = float(np.clip(self.smoothed_alpha, 1.0, 2.2))

        return {
            "mean_luminance": round(mean_l, 1),
            "contrast_std": round(l_contrast, 1),
            "sharpness_laplacian_var": round(laplacian_var, 1),
            "lighting_condition": lighting_cond,
            "visibility_condition": visibility_cond,
            "surface_condition": surface_cond,
            "effective_friction_mu": round(effective_mu, 2),
            "effective_reaction_time_sec": round(effective_react_time, 2),
            "alpha_env_expansion_factor": round(alpha_final, 3),
            "adjusted_critical_ttc_sec": round(2.0 * alpha_final, 2),
            "adjusted_warning_ttc_sec": round(3.0 * alpha_final, 2)
        }

    def _default_environment(self) -> Dict[str, Any]:
        """Fallback neutral daytime conditions."""
        return {
            "mean_luminance": 128.0,
            "contrast_std": 45.0,
            "sharpness_laplacian_var": 300.0,
            "lighting_condition": "DAYLIGHT",
            "visibility_condition": "CLEAR",
            "surface_condition": "DRY_STANDARD",
            "effective_friction_mu": self.mu_dry,
            "effective_reaction_time_sec": self.t_react_nominal,
            "alpha_env_expansion_factor": 1.0,
            "adjusted_critical_ttc_sec": 2.0,
            "adjusted_warning_ttc_sec": 3.0
        }
