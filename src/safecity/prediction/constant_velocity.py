"""SafeCity AI: Constant-Velocity Trajectory Predictor.

Analytical physics baseline forecasting future spatiotemporal coordinates
from causally filtered track history over configurable prediction horizons.
Operates seamlessly in both image pixel space and metric world space.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("safecity.prediction.cv")


class ConstantVelocityPredictor:
    """Analytical constant-velocity trajectory forecasting baseline."""

    def __init__(
        self,
        observation_window_sec: float = 2.0,
        prediction_horizon_sec: float = 3.0,
        step_size_sec: float = 0.2,
        min_history_points: int = 3,
        stationary_threshold: float = 0.2  # m/s or px/frame
    ) -> None:
        self.obs_window_sec = observation_window_sec
        self.pred_horizon_sec = prediction_horizon_sec
        self.step_size_sec = step_size_sec
        self.min_history_points = min_history_points
        self.stationary_threshold = stationary_threshold

    def estimate_velocity(
        self,
        history_points: List[Tuple[float, float]],
        fps: float = 12.5
    ) -> Tuple[float, float]:
        """Causally estimates velocity (vx, vy) per second using finite differences with recency weighting."""
        if len(history_points) < 2 or fps <= 0:
            return (0.0, 0.0)

        dt = 1.0 / fps
        points = np.array(history_points[-int(self.obs_window_sec * fps):], dtype=np.float64)
        if len(points) < 2:
            return (0.0, 0.0)

        # Compute displacements between consecutive frames
        diffs = np.diff(points, axis=0) / dt  # velocities per frame step in units/sec

        # Exponential recency weights
        n = len(diffs)
        weights = np.exp(np.linspace(-1.0, 0.0, n))
        weights /= np.sum(weights)

        vx = float(np.sum(diffs[:, 0] * weights))
        vy = float(np.sum(diffs[:, 1] * weights))

        speed = np.hypot(vx, vy)
        if speed < self.stationary_threshold:
            return (0.0, 0.0)

        return (vx, vy)

    def predict(
        self,
        history_points: List[Tuple[float, float]],
        fps: float = 12.5,
        horizon_sec: Optional[float] = None
    ) -> np.ndarray:
        """Forecasts future points from the most recent point at step_size_sec intervals.

        Returns array of shape (N_steps, 2) where N_steps = int(horizon_sec / step_size_sec).
        """
        horizon = horizon_sec or self.pred_horizon_sec
        num_steps = max(1, int(round(horizon / self.step_size_sec)))

        if not history_points:
            return np.zeros((num_steps, 2), dtype=np.float64)

        current_pos = np.array(history_points[-1], dtype=np.float64)

        if len(history_points) < self.min_history_points:
            # Fallback: predict stationary if insufficient history
            return np.tile(current_pos, (num_steps, 1))

        vx, vy = self.estimate_velocity(history_points, fps=fps)
        time_offsets = np.arange(1, num_steps + 1) * self.step_size_sec

        future_x = current_pos[0] + vx * time_offsets
        future_y = current_pos[1] + vy * time_offsets

        return np.column_stack((future_x, future_y))

    def predict_footprints(
        self,
        predicted_points: np.ndarray,
        footprint_dims: Tuple[float, float],
        heading_angle_rad: Optional[float] = None
    ) -> List[np.ndarray]:
        """Projects a 4-corner bounding rectangle along each predicted future trajectory step."""
        length, width = footprint_dims
        half_l = length / 2.0
        half_w = width / 2.0

        footprints: List[np.ndarray] = []
        for i, center in enumerate(predicted_points):
            # Compute heading from trajectory segment if not provided
            if heading_angle_rad is None:
                if i < len(predicted_points) - 1:
                    dx = predicted_points[i + 1, 0] - center[0]
                    dy = predicted_points[i + 1, 1] - center[1]
                    theta = np.arctan2(dy, dx)
                elif i > 0:
                    dx = center[0] - predicted_points[i - 1, 0]
                    dy = center[1] - predicted_points[i - 1, 1]
                    theta = np.arctan2(dy, dx)
                else:
                    theta = 0.0
            else:
                theta = heading_angle_rad

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # Local box corners relative to center
            local_corners = np.array([
                [-half_l, -half_w],
                [half_l, -half_w],
                [half_l, half_w],
                [-half_l, half_w]
            ])

            R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            world_corners = (local_corners @ R.T) + center
            footprints.append(world_corners)

        return footprints
