"""SafeCity AI: Trajectory Prediction Metrics.

Implements Average Displacement Error (ADE) and Final Displacement Error (FDE)
across multi-step prediction horizons (e.g., 1.0s, 2.0s, 3.0s).
"""

from __future__ import annotations

from typing import Dict
import numpy as np


def compute_ade_fde(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    horizons: Dict[str, int]
) -> Dict[str, float]:
    """Computes ADE and FDE at specified timestep horizons.

    Args:
        pred_coords: Array (N, T_pred, 2)
        gt_coords: Array (N, T_pred, 2)
        horizons: Dict of label -> timestep_index (1-indexed)
    """
    if pred_coords.shape != gt_coords.shape:
        raise ValueError(f"Shape mismatch: {pred_coords.shape} vs {gt_coords.shape}")

    # Euclidean error at each step: (N, T_pred)
    errors = np.linalg.norm(pred_coords - gt_coords, axis=2)

    metrics = {}
    for label, step in horizons.items():
        step_idx = min(step, pred_coords.shape[1])
        # ADE up to step_idx
        metrics[f"ADE_{label}"] = float(np.mean(errors[:, :step_idx]))
        # FDE at step_idx
        metrics[f"FDE_{label}"] = float(np.mean(errors[:, step_idx - 1]))

    metrics["ADE_overall"] = float(np.mean(errors))
    metrics["FDE_overall"] = float(np.mean(errors[:, -1]))
    return metrics
