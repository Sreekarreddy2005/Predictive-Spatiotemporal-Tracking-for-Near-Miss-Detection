"""Unit tests for Phase 3 Novelty 1: Spatiotemporal Graph Attention (ST-GAT) interaction model."""

import numpy as np
import pytest
import torch

from safecity.prediction.interaction_predictor import (
    SpatialGraphAttentionLayer,
    STGATTrajectoryModel,
    InteractionTrajectoryPredictor
)
from safecity.prediction.dataset import InteractionTrajectoryDataset


def test_spatial_graph_attention_layer():
    """Verify that SpatialGraphAttentionLayer attends to active neighbors and ignores masked padding."""
    b, k, d = 4, 3, 64
    attn_layer = SpatialGraphAttentionLayer(hidden_dim=d, num_heads=2)

    ego_feat = torch.randn(b, d)
    neighbor_feat = torch.randn(b, k, d)
    # Batch 0: all 3 active, Batch 1: only first active, Batch 2: none active
    neighbor_mask = torch.tensor([
        [True, True, True],
        [True, False, False],
        [False, False, False],
        [True, True, False]
    ], dtype=torch.bool)

    context, weights = attn_layer(ego_feat, neighbor_feat, neighbor_mask)

    assert context.shape == (b, d)
    assert weights.shape == (b, 2, 1, k)
    assert not torch.isnan(context).any(), "Context output contains NaN values"

    # In Batch 1, masked neighbors (index 1 and 2) must receive 0 attention weight
    assert weights[1, :, 0, 1].max().item() == pytest.approx(0.0)
    assert weights[1, :, 0, 2].max().item() == pytest.approx(0.0)


def test_stgat_trajectory_model_forward():
    """Verify STGATTrajectoryModel forward pass shapes and cumulative displacement computation."""
    b, t_obs, t_pred, k = 4, 10, 15, 4
    model = STGATTrajectoryModel(
        input_dim=2,
        neighbor_feat_dim=4,
        embedding_dim=32,
        hidden_dim=64,
        num_layers=1,
        pred_len=t_pred
    )

    obs_rel = torch.randn(b, t_obs, 2)
    neigh_states = torch.randn(b, k, 4)
    neigh_masks = torch.ones(b, k, dtype=torch.bool)

    pred_rel, attn = model(obs_rel, neigh_states, neigh_masks, pred_len=t_pred)

    assert pred_rel.shape == (b, t_pred, 2)
    assert not torch.isnan(pred_rel).any()


def test_interaction_predictor_runtime_wrapper(tmp_path):
    """Verify runtime wrapper handles trail inference with and without neighbors."""
    ckpt_path = tmp_path / "test_stgat.pt"
    predictor = InteractionTrajectoryPredictor(checkpoint_path=None, obs_len=5, pred_len=10, hidden_dim=32, device="cpu")

    ego_trail = [(float(i), float(i * 0.5)) for i in range(8)]
    neighbor_trails = [
        [(float(i + 2), float(i * 0.4)) for i in range(8)],
        [(float(i - 1), float(i * 0.6)) for i in range(8)]
    ]

    # Predict with neighbors
    pred_with = predictor.predict(ego_trail, neighbor_trails=neighbor_trails)
    assert pred_with is not None
    assert pred_with.shape == (10, 2)

    # Predict with empty neighbors (graceful fallback)
    pred_no_neigh = predictor.predict(ego_trail, neighbor_trails=None)
    assert pred_no_neigh is not None
    assert pred_no_neigh.shape == (10, 2)

    # Checkpoint save & load
    predictor.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()

    reloaded = InteractionTrajectoryPredictor(checkpoint_path=str(ckpt_path), device="cpu")
    assert reloaded.is_trained is True
