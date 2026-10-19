"""SafeCity AI: Trajectory Model Training & ADE/FDE Evaluation Benchmark.

Trains and evaluates 3 models on the identical held-out test split:
1. Baseline 1: Analytical Constant-Velocity Physics Model
2. Baseline 2: Per-Agent Sequence-to-Sequence PyTorch LSTM
3. Novelty Module 1: Spatiotemporal Graph Attention (ST-GAT) Interaction-Aware Model

Metrics computed:
- Average Displacement Error (ADE) at 1.0s, 2.0s, 3.0s horizons.
- Final Displacement Error (FDE) at 1.0s, 2.0s, 3.0s horizons.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from safecity.metrics.trajectory_metrics import compute_ade_fde
from safecity.prediction.dataset import (
    create_trajectory_splits,
    create_interaction_splits,
    TrajectoryDataset,
    InteractionTrajectoryDataset
)
from safecity.prediction.lstm_predictor import Seq2SeqLSTMModel, LSTMTrajectoryPredictor
from safecity.prediction.interaction_predictor import STGATTrajectoryModel, InteractionTrajectoryPredictor

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("safecity.train_eval")


def evaluate_constant_velocity_baseline(
    test_dataset: TrajectoryDataset,
    horizons: Dict[str, int],
    dt: float = 0.2
) -> Dict[str, float]:
    """Evaluates the analytical constant-velocity physics baseline on the test split."""
    obs_all = test_dataset.obs_traj.numpy()
    gt_all = test_dataset.pred_traj.numpy()
    n_samples = len(obs_all)
    pred_len = gt_all.shape[1]

    cv_predictions = np.zeros_like(gt_all)
    for i in range(n_samples):
        obs = obs_all[i]
        last_pos = obs[-1]
        vel = (obs[-1] - obs[0]) / (len(obs) * dt)
        steps = np.arange(1, pred_len + 1).reshape(-1, 1)
        cv_predictions[i] = last_pos + vel * (steps * dt)

    metrics = compute_ade_fde(cv_predictions, gt_all, horizons)
    return metrics


def train_lstm_model(
    train_dataset: TrajectoryDataset,
    val_dataset: TrajectoryDataset,
    checkpoint_path: str = "models/lstm_baseline.pt",
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    device: str = "auto"
) -> Seq2SeqLSTMModel:
    """Trains the Seq2Seq LSTM model with validation tracking."""
    resolved_device = LSTMTrajectoryPredictor._resolve_device(device)
    logger.info(f"Training Seq2Seq LSTM on device: {resolved_device}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    pred_len = train_dataset.pred_traj.shape[1]
    model = Seq2SeqLSTMModel(pred_len=pred_len, hidden_dim=hidden_dim).to(resolved_device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            obs_rel = batch["obs_rel"].to(resolved_device)
            target_rel = batch["pred_rel"].to(resolved_device)

            optimizer.zero_grad()
            pred_rel = model(obs_rel, pred_len=pred_len)
            loss = criterion(pred_rel, target_rel)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                obs_rel = batch["obs_rel"].to(resolved_device)
                target_rel = batch["pred_rel"].to(resolved_device)
                pred_rel = model(obs_rel, pred_len=pred_len)
                val_losses.append(criterion(pred_rel, target_rel).item())

        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))
        scheduler.step(mean_val)
        logger.info(f"[LSTM] Epoch {epoch:02d}/{epochs:02d} | Train: {mean_train:.4f} | Val: {mean_val:.4f}")

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "obs_len": int(train_dataset.obs_traj.shape[1]),
                    "pred_len": int(pred_len),
                    "hidden_dim": int(model.hidden_dim),
                    "embedding_dim": int(model.embedding_dim),
                    "num_layers": int(model.num_layers),
                    "val_loss": float(best_val_loss),
                    "epoch": int(epoch)
                },
                checkpoint_path
            )

    logger.info(f"Saved best LSTM checkpoint to {checkpoint_path} (Val: {best_val_loss:.4f})")
    best_ckpt = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    return model


def evaluate_lstm_model(
    model: Seq2SeqLSTMModel,
    test_dataset: TrajectoryDataset,
    horizons: Dict[str, int],
    device: str = "auto"
) -> Dict[str, float]:
    """Evaluates the trained LSTM model on the test split."""
    resolved_device = LSTMTrajectoryPredictor._resolve_device(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    all_preds, all_gts = [], []

    with torch.no_grad():
        for batch in test_loader:
            obs_rel = batch["obs_rel"].to(resolved_device)
            last_pos = batch["last_obs_pos"].numpy()
            pred_gt = batch["pred_traj"].numpy()

            pred_rel = model(obs_rel).cpu().numpy()
            pred_abs = last_pos[:, np.newaxis, :] + pred_rel
            all_preds.append(pred_abs)
            all_gts.append(pred_gt)

    pred_coords = np.concatenate(all_preds, axis=0)
    gt_coords = np.concatenate(all_gts, axis=0)
    metrics = compute_ade_fde(pred_coords, gt_coords, horizons)
    return metrics


def train_stgat_model(
    train_dataset: InteractionTrajectoryDataset,
    val_dataset: InteractionTrajectoryDataset,
    checkpoint_path: str = "models/interaction_stgat.pt",
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    device: str = "auto"
) -> STGATTrajectoryModel:
    """Trains the Novelty 1 Spatiotemporal Graph Attention (ST-GAT) model."""
    resolved_device = InteractionTrajectoryPredictor._resolve_device(device)
    logger.info(f"Training ST-GAT Interaction model on device: {resolved_device}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    pred_len = train_dataset.pred_traj.shape[1]
    model = STGATTrajectoryModel(pred_len=pred_len, hidden_dim=hidden_dim).to(resolved_device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            obs_rel = batch["obs_rel"].to(resolved_device)
            neigh_states = batch["neighbor_states"].to(resolved_device)
            neigh_masks = batch["neighbor_masks"].to(resolved_device)
            target_rel = batch["pred_rel"].to(resolved_device)

            optimizer.zero_grad()
            pred_rel, _ = model(obs_rel, neigh_states, neigh_masks, pred_len=pred_len)
            loss = criterion(pred_rel, target_rel)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                obs_rel = batch["obs_rel"].to(resolved_device)
                neigh_states = batch["neighbor_states"].to(resolved_device)
                neigh_masks = batch["neighbor_masks"].to(resolved_device)
                target_rel = batch["pred_rel"].to(resolved_device)
                pred_rel, _ = model(obs_rel, neigh_states, neigh_masks, pred_len=pred_len)
                val_losses.append(criterion(pred_rel, target_rel).item())

        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))
        scheduler.step(mean_val)
        logger.info(f"[ST-GAT] Epoch {epoch:02d}/{epochs:02d} | Train: {mean_train:.4f} | Val: {mean_val:.4f}")

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "obs_len": int(train_dataset.obs_traj.shape[1]),
                    "pred_len": int(pred_len),
                    "hidden_dim": int(model.hidden_dim),
                    "embedding_dim": int(model.embedding_dim),
                    "num_layers": int(model.num_layers),
                    "val_loss": float(best_val_loss),
                    "epoch": int(epoch)
                },
                checkpoint_path
            )

    logger.info(f"Saved best ST-GAT checkpoint to {checkpoint_path} (Val: {best_val_loss:.4f})")
    best_ckpt = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    return model


def evaluate_stgat_model(
    model: STGATTrajectoryModel,
    test_dataset: InteractionTrajectoryDataset,
    horizons: Dict[str, int],
    device: str = "auto"
) -> Dict[str, float]:
    """Evaluates the trained ST-GAT model on the held-out test split."""
    resolved_device = InteractionTrajectoryPredictor._resolve_device(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    all_preds, all_gts = [], []

    with torch.no_grad():
        for batch in test_loader:
            obs_rel = batch["obs_rel"].to(resolved_device)
            neigh_states = batch["neighbor_states"].to(resolved_device)
            neigh_masks = batch["neighbor_masks"].to(resolved_device)
            last_pos = batch["last_obs_pos"].numpy()
            pred_gt = batch["pred_traj"].numpy()

            pred_rel, _ = model(obs_rel, neigh_states, neigh_masks)
            pred_rel_np = pred_rel.cpu().numpy()
            pred_abs = last_pos[:, np.newaxis, :] + pred_rel_np
            all_preds.append(pred_abs)
            all_gts.append(pred_gt)

    pred_coords = np.concatenate(all_preds, axis=0)
    gt_coords = np.concatenate(all_gts, axis=0)
    metrics = compute_ade_fde(pred_coords, gt_coords, horizons)
    return metrics


def run_benchmark(
    data_path: str = "data/raw/trajectories/01_tracks.csv",
    lstm_checkpoint: str = "models/lstm_baseline.pt",
    stgat_checkpoint: str = "models/interaction_stgat.pt",
    output_report_path: str = "outputs/prediction_benchmark.json",
    epochs: int = 5,
    max_agents: int = 200,
    device: str = "auto"
) -> Dict[str, Any]:
    """Runs data preparation, baseline evaluation, LSTM training, ST-GAT training, and 3-way benchmarking."""
    start_time = time.time()
    horizons = {"1.0s": 5, "2.0s": 10, "3.0s": 15}

    logger.info("=== Phase 3: Multi-Agent Trajectory Forecasting Benchmark ===")
    train_ds, val_ds, test_ds = create_interaction_splits(
        csv_path=data_path,
        obs_len=10,
        pred_len=15,
        downsample_factor=5,
        max_agents=max_agents
    )

    logger.info("1. Evaluating Baseline 1: Constant-Velocity Physics Model...")
    cv_metrics = evaluate_constant_velocity_baseline(test_ds, horizons, dt=0.2)

    logger.info("2. Training Baseline 2: Per-Agent Sequence-to-Sequence PyTorch LSTM...")
    lstm_model = train_lstm_model(
        train_dataset=train_ds,
        val_dataset=val_ds,
        checkpoint_path=lstm_checkpoint,
        epochs=epochs,
        batch_size=128,
        lr=1e-3,
        device=device
    )
    lstm_metrics = evaluate_lstm_model(lstm_model, test_ds, horizons, device=device)

    logger.info("3. Training Novelty 1: Spatiotemporal Graph Attention (ST-GAT)...")
    stgat_model = train_stgat_model(
        train_dataset=train_ds,
        val_dataset=val_ds,
        checkpoint_path=stgat_checkpoint,
        epochs=epochs,
        batch_size=128,
        lr=1e-3,
        device=device
    )
    stgat_metrics = evaluate_stgat_model(stgat_model, test_ds, horizons, device=device)

    total_time = time.time() - start_time

    # Generate Comparative Report
    report = {
        "benchmark_dataset": str(data_path),
        "test_samples_count": len(test_ds),
        "observation_horizon_sec": 2.0,
        "prediction_horizon_sec": 3.0,
        "step_frequency_hz": 5.0,
        "training_epochs": epochs,
        "elapsed_seconds": total_time,
        "models": {
            "constant_velocity_baseline": cv_metrics,
            "per_agent_lstm": lstm_metrics,
            "stgat_interaction_aware": stgat_metrics
        },
        "comparison_table": {
            "ADE_1.0s": {"Constant_Velocity": cv_metrics["ADE_1.0s"], "LSTM": lstm_metrics["ADE_1.0s"], "ST_GAT": stgat_metrics["ADE_1.0s"]},
            "ADE_2.0s": {"Constant_Velocity": cv_metrics["ADE_2.0s"], "LSTM": lstm_metrics["ADE_2.0s"], "ST_GAT": stgat_metrics["ADE_2.0s"]},
            "ADE_3.0s": {"Constant_Velocity": cv_metrics["ADE_3.0s"], "LSTM": lstm_metrics["ADE_3.0s"], "ST_GAT": stgat_metrics["ADE_3.0s"]},
            "FDE_1.0s": {"Constant_Velocity": cv_metrics["FDE_1.0s"], "LSTM": lstm_metrics["FDE_1.0s"], "ST_GAT": stgat_metrics["FDE_1.0s"]},
            "FDE_2.0s": {"Constant_Velocity": cv_metrics["FDE_2.0s"], "LSTM": lstm_metrics["FDE_2.0s"], "ST_GAT": stgat_metrics["FDE_2.0s"]},
            "FDE_3.0s": {"Constant_Velocity": cv_metrics["FDE_3.0s"], "LSTM": lstm_metrics["FDE_3.0s"], "ST_GAT": stgat_metrics["FDE_3.0s"]},
        }
    }

    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print Formatted Results Table
    print("\n" + "=" * 80)
    print("  SAFECITY AI: 3-WAY TRAJECTORY PREDICTION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Metric Horizon':<15} | {'Constant Velocity (m)':<22} | {'Per-Agent LSTM (m)':<20} | {'ST-GAT (Novelty 1) (m)'}")
    print("-" * 80)
    for k in ["1.0s", "2.0s", "3.0s"]:
        ade_cv = cv_metrics[f"ADE_{k}"]
        ade_lstm = lstm_metrics[f"ADE_{k}"]
        ade_gat = stgat_metrics[f"ADE_{k}"]
        print(f"ADE @ {k:<9} | {ade_cv:<22.4f} | {ade_lstm:<20.4f} | {ade_gat:<20.4f}")
    print("-" * 80)
    for k in ["1.0s", "2.0s", "3.0s"]:
        fde_cv = cv_metrics[f"FDE_{k}"]
        fde_lstm = lstm_metrics[f"FDE_{k}"]
        fde_gat = stgat_metrics[f"FDE_{k}"]
        print(f"FDE @ {k:<9} | {fde_cv:<22.4f} | {fde_lstm:<20.4f} | {fde_gat:<20.4f}")
    print("=" * 80)
    print(f"✅ Report saved to {output_report_path}\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate 3-Way Trajectory Benchmark")
    parser.add_argument("--data", type=str, default="data/raw/trajectories/01_tracks.csv")
    parser.add_argument("--lstm_checkpoint", type=str, default="models/lstm_baseline.pt")
    parser.add_argument("--stgat_checkpoint", type=str, default="models/interaction_stgat.pt")
    parser.add_argument("--output", type=str, default="outputs/prediction_benchmark.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_agents", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_benchmark(
        data_path=args.data,
        lstm_checkpoint=args.lstm_checkpoint,
        stgat_checkpoint=args.stgat_checkpoint,
        output_report_path=args.output,
        epochs=args.epochs,
        max_agents=args.max_agents,
        device=args.device
    )
