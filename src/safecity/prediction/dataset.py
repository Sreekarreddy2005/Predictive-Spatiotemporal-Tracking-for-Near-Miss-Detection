"""SafeCity AI: Trajectory Dataset & Windowing Pipeline.

Loads naturalistic vehicle and pedestrian trajectory CSVs (inD, highD, levelXdata format),
performs downsampling, sliding window extraction (T_obs = 2.0s, T_pred = 3.0s),
and creates leak-free PyTorch datasets with displacement normalization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("safecity.prediction.dataset")


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for trajectory sequence forecasting."""

    def __init__(
        self,
        obs_traj: np.ndarray,
        pred_traj: np.ndarray,
        obs_rel: np.ndarray,
        pred_rel: np.ndarray,
        agent_ids: np.ndarray,
        last_obs_pos: np.ndarray,
    ) -> None:
        """
        Args:
            obs_traj: Absolute observed coordinates (N, T_obs, 2)
            pred_traj: Absolute ground-truth future coordinates (N, T_pred, 2)
            obs_rel: Incremental observed displacements (N, T_obs, 2)
            pred_rel: Incremental future displacements from last observed point (N, T_pred, 2)
            agent_ids: Agent track ID for each sample (N,)
            last_obs_pos: Position at the last observed step (N, 2)
        """
        self.obs_traj = torch.from_numpy(obs_traj).float()
        self.pred_traj = torch.from_numpy(pred_traj).float()
        self.obs_rel = torch.from_numpy(obs_rel).float()
        self.pred_rel = torch.from_numpy(pred_rel).float()
        self.agent_ids = torch.from_numpy(agent_ids).long()
        self.last_obs_pos = torch.from_numpy(last_obs_pos).float()

    def __len__(self) -> int:
        return len(self.obs_traj)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "obs_traj": self.obs_traj[idx],
            "pred_traj": self.pred_traj[idx],
            "obs_rel": self.obs_rel[idx],
            "pred_rel": self.pred_rel[idx],
            "agent_id": self.agent_ids[idx],
            "last_obs_pos": self.last_obs_pos[idx],
        }


def extract_trajectory_windows(
    df: pd.DataFrame,
    obs_len: int = 10,
    pred_len: int = 15,
    downsample_factor: int = 5,
    sliding_stride: int = 2,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slices continuous vehicle tracks into sliding observation and prediction windows.

    At 25 Hz original frequency with downsample_factor=5 (5 Hz effective):
    - obs_len = 10 steps -> 2.0 seconds observation
    - pred_len = 15 steps -> 3.0 seconds prediction horizon
    """
    total_len = obs_len + pred_len
    obs_list = []
    pred_list = []
    obs_rel_list = []
    pred_rel_list = []
    id_list = []
    last_pos_list = []

    # Ensure required columns exist
    df_cols = [c.lower() for c in df.columns]
    frame_col = "frame" if "frame" in df_cols else df.columns[0]
    id_col = "id" if "id" in df_cols else ("track_id" if "track_id" in df_cols else df.columns[1])
    x_col = "x" if "x" in df_cols else df.columns[2]
    y_col = "y" if "y" in df_cols else df.columns[3]

    grouped = df.groupby(id_col)
    logger.info(f"Extracting trajectories from {len(grouped)} unique agents...")

    for agent_id, group in grouped:
        group_sorted = group.sort_values(by=frame_col)
        # Subsample to target frequency
        subsampled = group_sorted.iloc[::downsample_factor]
        coords = subsampled[[x_col, y_col]].to_numpy(dtype=np.float32)

        if len(coords) < total_len:
            continue

        # Slide window across track
        for start_idx in range(0, len(coords) - total_len + 1, sliding_stride):
            window = coords[start_idx : start_idx + total_len]
            obs = window[:obs_len]
            pred = window[obs_len:]

            # Compute relative displacements
            # obs_rel: step-to-step deltas
            obs_rel = np.zeros_like(obs)
            obs_rel[1:] = obs[1:] - obs[:-1]

            # pred_rel: offset relative to the final observed point
            last_pos = obs[-1]
            pred_rel = pred - last_pos

            obs_list.append(obs)
            pred_list.append(pred)
            obs_rel_list.append(obs_rel)
            pred_rel_list.append(pred_rel)
            id_list.append(int(agent_id))
            last_pos_list.append(last_pos)

            if max_samples and len(obs_list) >= max_samples:
                break
        if max_samples and len(obs_list) >= max_samples:
            break

    if not obs_list:
        raise ValueError("No valid trajectory sequences found meeting the length requirement.")

    return (
        np.array(obs_list, dtype=np.float32),
        np.array(pred_list, dtype=np.float32),
        np.array(obs_rel_list, dtype=np.float32),
        np.array(pred_rel_list, dtype=np.float32),
        np.array(id_list, dtype=np.int64),
        np.array(last_pos_list, dtype=np.float32),
    )


def create_trajectory_splits(
    csv_path: Union[str, Path],
    obs_len: int = 10,
    pred_len: int = 15,
    downsample_factor: int = 5,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_agents: Optional[int] = None,
    random_seed: int = 42
) -> Tuple[TrajectoryDataset, TrajectoryDataset, TrajectoryDataset]:
    """Splits trajectory data by Agent ID to strictly prevent temporal leakage across splits."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {csv_path}")

    logger.info(f"Loading trajectory table from {csv_path}...")
    df = pd.read_csv(csv_path)

    id_col = "id" if "id" in df.columns else ("track_id" if "track_id" in df.columns else df.columns[1])
    unique_ids = df[id_col].unique()

    if max_agents:
        unique_ids = unique_ids[:max_agents]
        df = df[df[id_col].isin(unique_ids)]

    # Deterministic split by ID
    np.random.seed(random_seed)
    shuffled_ids = np.random.permutation(unique_ids)

    n_total = len(shuffled_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = set(shuffled_ids[:n_train])
    val_ids = set(shuffled_ids[n_train : n_train + n_val])
    test_ids = set(shuffled_ids[n_train + n_val :])

    df_train = df[df[id_col].isin(train_ids)]
    df_val = df[df[id_col].isin(val_ids)]
    df_test = df[df[id_col].isin(test_ids)]

    logger.info(f"Agent split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    train_arrays = extract_trajectory_windows(df_train, obs_len, pred_len, downsample_factor)
    val_arrays = extract_trajectory_windows(df_val, obs_len, pred_len, downsample_factor)
    test_arrays = extract_trajectory_windows(df_test, obs_len, pred_len, downsample_factor)

    train_dataset = TrajectoryDataset(*train_arrays)
    val_dataset = TrajectoryDataset(*val_arrays)
    test_dataset = TrajectoryDataset(*test_arrays)

    logger.info(
        f"Generated windows: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test"
    )
    return train_dataset, val_dataset, test_dataset


class InteractionTrajectoryDataset(Dataset):
    """PyTorch Dataset for multi-agent interaction-aware trajectory forecasting."""

    def __init__(
        self,
        obs_traj: np.ndarray,
        pred_traj: np.ndarray,
        obs_rel: np.ndarray,
        pred_rel: np.ndarray,
        agent_ids: np.ndarray,
        last_obs_pos: np.ndarray,
        neighbor_states: np.ndarray,
        neighbor_masks: np.ndarray,
    ) -> None:
        """
        Args:
            obs_traj: Absolute observed coordinates (N, T_obs, 2)
            pred_traj: Absolute ground-truth future coordinates (N, T_pred, 2)
            obs_rel: Incremental observed displacements (N, T_obs, 2)
            pred_rel: Incremental future displacements from last observed point (N, T_pred, 2)
            agent_ids: Agent track ID for each sample (N,)
            last_obs_pos: Position at the last observed step (N, 2)
            neighbor_states: Relative states (pos_x, pos_y, vel_x, vel_y) of K closest neighbors (N, K, 4)
            neighbor_masks: Valid neighbor indicators (N, K) boolean
        """
        self.obs_traj = torch.from_numpy(obs_traj).float()
        self.pred_traj = torch.from_numpy(pred_traj).float()
        self.obs_rel = torch.from_numpy(obs_rel).float()
        self.pred_rel = torch.from_numpy(pred_rel).float()
        self.agent_ids = torch.from_numpy(agent_ids).long()
        self.last_obs_pos = torch.from_numpy(last_obs_pos).float()
        self.neighbor_states = torch.from_numpy(neighbor_states).float()
        self.neighbor_masks = torch.from_numpy(neighbor_masks).bool()

    def __len__(self) -> int:
        return len(self.obs_traj)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "obs_traj": self.obs_traj[idx],
            "pred_traj": self.pred_traj[idx],
            "obs_rel": self.obs_rel[idx],
            "pred_rel": self.pred_rel[idx],
            "agent_id": self.agent_ids[idx],
            "last_obs_pos": self.last_obs_pos[idx],
            "neighbor_states": self.neighbor_states[idx],
            "neighbor_masks": self.neighbor_masks[idx],
        }


def extract_interaction_windows(
    df: pd.DataFrame,
    obs_len: int = 10,
    pred_len: int = 15,
    downsample_factor: int = 5,
    sliding_stride: int = 2,
    max_neighbors: int = 4,
    interaction_radius_meters: float = 60.0,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts trajectory windows along with K nearest neighbor spatial interaction states."""
    total_len = obs_len + pred_len
    obs_list = []
    pred_list = []
    obs_rel_list = []
    pred_rel_list = []
    id_list = []
    last_pos_list = []
    neighbor_state_list = []
    neighbor_mask_list = []

    # Map column names
    df_cols = {c.lower(): c for c in df.columns}
    frame_col = df_cols.get("frame", df.columns[0])
    id_col = df_cols.get("id", df_cols.get("track_id", df.columns[1]))
    x_col = df_cols.get("x", df.columns[2])
    y_col = df_cols.get("y", df.columns[3])
    vx_col = df_cols.get("xvelocity", df_cols.get("vx", None))
    vy_col = df_cols.get("yvelocity", df_cols.get("vy", None))

    # Fast multi-agent frame index
    use_cols = [id_col, x_col, y_col]
    has_vel = vx_col is not None and vy_col is not None
    if has_vel:
        use_cols.extend([vx_col, vy_col])

    frame_dict = {}
    for frame_val, group in df.groupby(frame_col):
        frame_dict[frame_val] = group[use_cols].to_numpy(dtype=np.float32)

    grouped = df.groupby(id_col)
    logger.info(f"Extracting interaction windows across {len(grouped)} unique agents...")

    for agent_id, group in grouped:
        group_sorted = group.sort_values(by=frame_col)
        subsampled = group_sorted.iloc[::downsample_factor]
        coords = subsampled[[x_col, y_col]].to_numpy(dtype=np.float32)
        frame_ids = subsampled[frame_col].to_numpy()

        if len(coords) < total_len:
            continue

        for start_idx in range(0, len(coords) - total_len + 1, sliding_stride):
            window = coords[start_idx : start_idx + total_len]
            obs = window[:obs_len]
            pred = window[obs_len:]

            obs_rel = np.zeros_like(obs)
            obs_rel[1:] = obs[1:] - obs[:-1]

            last_pos = obs[-1]
            pred_rel = pred - last_pos

            # Extract interaction neighbors at the final observation timestep
            obs_frame = frame_ids[start_idx + obs_len - 1]
            frame_agents = frame_dict.get(obs_frame)

            neigh_state = np.zeros((max_neighbors, 4), dtype=np.float32)
            neigh_mask = np.zeros(max_neighbors, dtype=bool)

            if frame_agents is not None and len(frame_agents) > 1:
                other_agents = frame_agents[frame_agents[:, 0] != agent_id]
                if len(other_agents) > 0:
                    rel_p = other_agents[:, 1:3] - last_pos
                    dists = np.linalg.norm(rel_p, axis=1)
                    within_radius = dists <= interaction_radius_meters
                    valid_others = other_agents[within_radius]
                    valid_dists = dists[within_radius]

                    if len(valid_others) > 0:
                        order = np.argsort(valid_dists)[:max_neighbors]
                        selected = valid_others[order]
                        for k, row in enumerate(selected):
                            rp = row[1:3] - last_pos
                            rv = (row[3:5] - (obs_rel[-1] * 5.0)) if has_vel else np.zeros(2, dtype=np.float32)
                            neigh_state[k, :2] = rp
                            neigh_state[k, 2:] = rv
                            neigh_mask[k] = True

            obs_list.append(obs)
            pred_list.append(pred)
            obs_rel_list.append(obs_rel)
            pred_rel_list.append(pred_rel)
            id_list.append(int(agent_id))
            last_pos_list.append(last_pos)
            neighbor_state_list.append(neigh_state)
            neighbor_mask_list.append(neigh_mask)

            if max_samples and len(obs_list) >= max_samples:
                break
        if max_samples and len(obs_list) >= max_samples:
            break

    return (
        np.array(obs_list, dtype=np.float32),
        np.array(pred_list, dtype=np.float32),
        np.array(obs_rel_list, dtype=np.float32),
        np.array(pred_rel_list, dtype=np.float32),
        np.array(id_list, dtype=np.int64),
        np.array(last_pos_list, dtype=np.float32),
        np.array(neighbor_state_list, dtype=np.float32),
        np.array(neighbor_mask_list, dtype=bool),
    )


def create_interaction_splits(
    csv_path: Union[str, Path],
    obs_len: int = 10,
    pred_len: int = 15,
    downsample_factor: int = 5,
    max_neighbors: int = 4,
    interaction_radius_meters: float = 60.0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_agents: Optional[int] = None,
    random_seed: int = 42
) -> Tuple[InteractionTrajectoryDataset, InteractionTrajectoryDataset, InteractionTrajectoryDataset]:
    """Creates train, val, and test splits with multi-agent interaction graph states."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {csv_path}")

    logger.info(f"Loading interaction trajectory table from {csv_path}...")
    df = pd.read_csv(csv_path)

    id_col = "id" if "id" in df.columns else ("track_id" if "track_id" in df.columns else df.columns[1])
    unique_ids = df[id_col].unique()

    if max_agents:
        unique_ids = unique_ids[:max_agents]
        df = df[df[id_col].isin(unique_ids)]

    np.random.seed(random_seed)
    shuffled_ids = np.random.permutation(unique_ids)

    n_total = len(shuffled_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = set(shuffled_ids[:n_train])
    val_ids = set(shuffled_ids[n_train : n_train + n_val])
    test_ids = set(shuffled_ids[n_train + n_val :])

    df_train = df[df[id_col].isin(train_ids)]
    df_val = df[df[id_col].isin(val_ids)]
    df_test = df[df[id_col].isin(test_ids)]

    logger.info(f"Interaction agent split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    train_arrays = extract_interaction_windows(df_train, obs_len, pred_len, downsample_factor, max_neighbors=max_neighbors, interaction_radius_meters=interaction_radius_meters)
    val_arrays = extract_interaction_windows(df_val, obs_len, pred_len, downsample_factor, max_neighbors=max_neighbors, interaction_radius_meters=interaction_radius_meters)
    test_arrays = extract_interaction_windows(df_test, obs_len, pred_len, downsample_factor, max_neighbors=max_neighbors, interaction_radius_meters=interaction_radius_meters)

    train_ds = InteractionTrajectoryDataset(*train_arrays)
    val_ds = InteractionTrajectoryDataset(*val_arrays)
    test_ds = InteractionTrajectoryDataset(*test_arrays)

    logger.info(f"Interaction windows generated: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
    return train_ds, val_ds, test_ds
