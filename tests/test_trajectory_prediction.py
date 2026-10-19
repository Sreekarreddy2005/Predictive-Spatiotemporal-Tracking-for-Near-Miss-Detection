"""Unit tests for Phase 3: Trajectory dataset loading, PyTorch LSTM modeling, and ADE/FDE metrics."""

from pathlib import Path
import numpy as np
import pytest
import torch

from safecity.prediction.dataset import TrajectoryDataset, extract_trajectory_windows
from safecity.prediction.lstm_predictor import Seq2SeqLSTMModel, LSTMTrajectoryPredictor
from safecity.metrics.trajectory_metrics import compute_ade_fde


def test_seq2seq_lstm_forward_pass():
    """Verify PyTorch Seq2Seq LSTM forward pass on tensor batches."""
    batch_size = 8
    obs_len = 10
    pred_len = 15
    model = Seq2SeqLSTMModel(input_dim=2, embedding_dim=32, hidden_dim=64, num_layers=1, pred_len=pred_len)

    # Input: (B, T_obs, 2)
    obs_rel = torch.randn(batch_size, obs_len, 2)
    out = model(obs_rel, pred_len=pred_len)

    assert out.shape == (batch_size, pred_len, 2), f"Expected shape ({batch_size}, {pred_len}, 2), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN values"


def test_lstm_predictor_runtime_wrapper(tmp_path):
    """Verify LSTMTrajectoryPredictor runtime trail inference and checkpoint saving."""
    ckpt_path = tmp_path / "test_lstm.pt"
    predictor = LSTMTrajectoryPredictor(checkpoint_path=None, obs_len=5, pred_len=10, hidden_dim=32, device="cpu")

    # Synthetic trail of points
    trail = [(float(i), float(i * 0.5)) for i in range(10)]
    pred = predictor.predict(trail)

    assert pred is not None
    assert pred.shape == (10, 2)
    assert not np.isnan(pred).any()

    # Save and reload
    predictor.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()

    reloaded = LSTMTrajectoryPredictor(checkpoint_path=str(ckpt_path), device="cpu")
    assert reloaded.is_trained is True


def test_trajectory_dataset_slicing():
    """Verify sliding window trajectory extractor with synthetic DataFrame."""
    import pandas as pd
    # Create synthetic linear motion tracks
    rows = []
    for agent_id in [1, 2]:
        for frame in range(100):
            rows.append({
                "frame": frame,
                "id": agent_id,
                "x": frame * 1.5,
                "y": frame * 0.2
            })
    df = pd.DataFrame(rows)

    obs_len = 8
    pred_len = 12
    obs, pred, obs_rel, pred_rel, ids, last_pos = extract_trajectory_windows(
        df, obs_len=obs_len, pred_len=pred_len, downsample_factor=2, sliding_stride=5
    )

    assert len(obs) > 0
    assert obs.shape[1:] == (obs_len, 2)
    assert pred.shape[1:] == (pred_len, 2)
    assert obs_rel.shape[1:] == (obs_len, 2)
    assert pred_rel.shape[1:] == (pred_len, 2)


def test_ade_fde_metrics_computation():
    """Verify that compute_ade_fde computes correct euclidean displacement errors."""
    n_samples = 10
    pred_len = 15
    horizons = {"1.0s": 5, "2.0s": 10, "3.0s": 15}

    # Ground truth: constant motion (t, t)
    gt = np.zeros((n_samples, pred_len, 2), dtype=np.float32)
    for t in range(pred_len):
        gt[:, t, :] = t

    # Prediction 1: Perfect prediction -> ADE = 0, FDE = 0
    perfect_metrics = compute_ade_fde(gt, gt, horizons)
    assert perfect_metrics["ADE_1.0s"] == pytest.approx(0.0)
    assert perfect_metrics["FDE_1.0s"] == pytest.approx(0.0)
    assert perfect_metrics["ADE_overall"] == pytest.approx(0.0)

    # Prediction 2: Constant offset of (3, 4) -> Euclidean error is exactly 5.0
    offset_pred = gt.copy()
    offset_pred[:, :, 0] += 3.0
    offset_pred[:, :, 1] += 4.0
    offset_metrics = compute_ade_fde(offset_pred, gt, horizons)
    assert offset_metrics["ADE_1.0s"] == pytest.approx(5.0)
    assert offset_metrics["FDE_3.0s"] == pytest.approx(5.0)
    assert offset_metrics["ADE_overall"] == pytest.approx(5.0)
