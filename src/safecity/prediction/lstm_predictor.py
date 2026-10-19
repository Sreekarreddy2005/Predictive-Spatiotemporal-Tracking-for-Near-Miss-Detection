"""SafeCity AI: Per-Agent Sequence-to-Sequence PyTorch LSTM Trajectory Predictor.

Features:
1. Encoder-Decoder LSTM architecture modeling causal temporal motion dynamics.
2. Invariance to global coordinate space via relative displacement encoding.
3. Checkpoint save/load methods and seamless inference wrapper for runtime pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("safecity.prediction.lstm")


class Seq2SeqLSTMModel(nn.Module):
    """Sequence-to-sequence LSTM predicting future vehicle displacement trajectories."""

    def __init__(
        self,
        input_dim: int = 2,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        pred_len: int = 15,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len

        # Encoder
        self.enc_embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.encoder_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Decoder
        self.dec_embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.decoder_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        obs_rel: torch.Tensor,
        pred_len: Optional[int] = None,
        teacher_forcing_target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            obs_rel: Tensor (B, T_obs, 2) of past relative displacements
            pred_len: Number of future timesteps to forecast
            teacher_forcing_target: Optional ground truth future deltas for training

        Returns:
            pred_rel: Tensor (B, T_pred, 2) of cumulative displacements relative to last obs point
        """
        batch_size = obs_rel.size(0)
        p_len = pred_len or self.pred_len
        device = obs_rel.device

        # 1. Encode past trajectory
        enc_emb = self.enc_embedding(obs_rel)
        _, (hidden, cell) = self.encoder_lstm(enc_emb)

        # 2. Decode future trajectory autoregressively
        # Start decoder with the last observed displacement step
        current_input = obs_rel[:, -1:, :]  # (B, 1, 2)
        step_outputs = []

        for t in range(p_len):
            dec_emb = self.dec_embedding(current_input)
            dec_out, (hidden, cell) = self.decoder_lstm(dec_emb, (hidden, cell))
            step_pred = self.out_proj(dec_out)  # (B, 1, 2)
            step_outputs.append(step_pred)

            if teacher_forcing_target is not None and t < teacher_forcing_target.size(1) - 1:
                # Use ground truth input for next step during teacher forcing
                current_input = teacher_forcing_target[:, t : t + 1, :]
            else:
                # Autoregressive feedback
                current_input = step_pred

        # Concatenate steps: (B, T_pred, 2)
        delta_steps = torch.cat(step_outputs, dim=1)

        # Cumulative displacement from origin (last obs point)
        pred_rel = torch.cumsum(delta_steps, dim=1)
        return pred_rel


class LSTMTrajectoryPredictor:
    """Production wrapper for loading and running the Seq2Seq LSTM predictor."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = "models/lstm_baseline.pt",
        obs_len: int = 10,
        pred_len: int = 15,
        hidden_dim: int = 128,
        device: str = "auto"
    ) -> None:
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.device = self._resolve_device(device)
        self.model = Seq2SeqLSTMModel(pred_len=pred_len, hidden_dim=hidden_dim).to(self.device)
        self.is_trained = False

        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "obs_len": self.obs_len,
                "pred_len": self.pred_len,
                "hidden_dim": self.model.hidden_dim,
                "embedding_dim": self.model.embedding_dim,
                "num_layers": self.model.num_layers,
                "is_trained": True
            },
            path
        )
        logger.info(f"Saved LSTM checkpoint to {path}")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.obs_len = checkpoint.get("obs_len", self.obs_len)
        self.pred_len = checkpoint.get("pred_len", self.pred_len)
        hidden_dim = checkpoint.get("hidden_dim", self.model.hidden_dim)
        embedding_dim = checkpoint.get("embedding_dim", self.model.embedding_dim)
        num_layers = checkpoint.get("num_layers", self.model.num_layers)

        self.model = Seq2SeqLSTMModel(
            pred_len=self.pred_len,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.is_trained = True
        logger.info(f"Loaded trained LSTM model from {path} (device={self.device})")

    def predict(self, trail: List[Tuple[float, float]], fps: Optional[float] = None) -> Optional[np.ndarray]:
        """Predicts future points given an agent's historical trail.

        Args:
            trail: List of (x, y) past centroid points
            fps: Optional video frame rate for interface polymorphism

        Returns:
            np.ndarray of shape (pred_len, 2) containing absolute future coordinates, or None
        """
        if len(trail) < 2:
            return None

        # Sample or interpolate trail to obs_len
        coords = np.array(trail, dtype=np.float32)
        if len(coords) < self.obs_len:
            # Linear extrapolation / padding if history is shorter than obs_len
            pad_len = self.obs_len - len(coords)
            first_pt = coords[0]
            padded = np.vstack([np.repeat([first_pt], pad_len, axis=0), coords])
            obs_traj = padded
        else:
            obs_traj = coords[-self.obs_len :]

        # Calculate incremental displacement steps
        obs_rel = np.zeros_like(obs_traj)
        obs_rel[1:] = obs_traj[1:] - obs_traj[:-1]

        inp_tensor = torch.from_numpy(obs_rel).unsqueeze(0).to(self.device)  # (1, T_obs, 2)

        self.model.eval()
        with torch.no_grad():
            pred_rel_tensor = self.model(inp_tensor, pred_len=self.pred_len)  # (1, T_pred, 2)

        pred_rel = pred_rel_tensor.squeeze(0).cpu().numpy()
        last_pos = obs_traj[-1]

        # Reconstruct absolute coordinates
        future_coords = last_pos + pred_rel
        return future_coords
