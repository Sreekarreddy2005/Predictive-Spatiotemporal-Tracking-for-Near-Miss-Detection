"""SafeCity AI: Spatiotemporal Graph Attention (ST-GAT) Interaction-Aware Trajectory Predictor.

Novelty Module 1:
Models multi-agent spatial-temporal interaction graphs so that road users
dynamically account for neighboring vehicle positions, headings, and approaching trajectories.

Mathematical Formulation:
1. Ego-temporal motion encoder: h_i = LSTM(Delta p_i(1:T_obs))
2. Neighbor spatial embedding: e_j = MLP(Delta p_ij, Delta v_ij) for j in N(i)
3. Multi-Head Spatial Graph Attention:
     alpha_ij = softmax( (W_q h_i) (W_k e_j)^T / sqrt(d) + mask_ij )
     s_i = sum_j alpha_ij (W_v e_j)
4. Interaction Gated Fusion: z_i = LayerNorm( Gate(h_i, s_i) )
5. Autoregressive Displacement Decoder: Delta hat{p}_i(T_obs+1:T_obs+T_pred) = Decoder(z_i)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("safecity.prediction.interaction")


class SpatialGraphAttentionLayer(nn.Module):
    """Calculates spatial attention weights over dynamic surrounding agent graphs."""

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        ego_feat: torch.Tensor,
        neighbor_feat: torch.Tensor,
        neighbor_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ego_feat: Tensor (B, hidden_dim)
            neighbor_feat: Tensor (B, K, hidden_dim)
            neighbor_mask: Boolean tensor (B, K) where True indicates an active neighbor

        Returns:
            context: Aggregated social context vector (B, hidden_dim)
            attn_weights: Attention distribution over neighbors (B, num_heads, 1, K)
        """
        b, k, _ = neighbor_feat.shape
        q = self.w_q(ego_feat).view(b, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, D_h)
        k_proj = self.w_k(neighbor_feat).view(b, k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, K, D_h)
        v_proj = self.w_v(neighbor_feat).view(b, k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, K, D_h)

        # Scaled dot-product
        scores = torch.matmul(q, k_proj.transpose(-2, -1)) / np.sqrt(self.head_dim)  # (B, H, 1, K)

        # Mask padding neighbors
        # neighbor_mask: (B, K) -> (B, 1, 1, K)
        mask = neighbor_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~mask, -1e9)

        # Softmax over neighbors
        attn = F.softmax(scores, dim=-1)
        # If an agent has 0 neighbors, replace all-NaN softmax with 0
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v_proj)  # (B, H, 1, D_h)
        context = context.transpose(1, 2).contiguous().view(b, self.hidden_dim)  # (B, hidden_dim)
        out = self.out_proj(context)

        return out, attn


class STGATTrajectoryModel(nn.Module):
    """Spatiotemporal Graph Attention (ST-GAT) Multi-Agent Trajectory Predictor."""

    def __init__(
        self,
        input_dim: int = 2,
        neighbor_feat_dim: int = 4,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pred_len: int = 15,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.neighbor_feat_dim = neighbor_feat_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len

        # 1. Ego-Temporal Motion Encoder
        self.ego_embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.ego_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 2. Neighbor Spatial State Encoder
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(neighbor_feat_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 3. Spatial Graph Attention Layer
        self.spatial_attention = SpatialGraphAttentionLayer(hidden_dim=hidden_dim, num_heads=4, dropout=dropout)

        # 4. Gated Interaction Fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.norm_layer = nn.LayerNorm(hidden_dim)

        # 5. Autoregressive Trajectory Decoder
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
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        obs_rel: torch.Tensor,
        neighbor_states: torch.Tensor,
        neighbor_masks: torch.Tensor,
        pred_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_rel: Tensor (B, T_obs, 2)
            neighbor_states: Tensor (B, K, 4) with (rel_x, rel_y, rel_vx, rel_vy)
            neighbor_masks: Boolean Tensor (B, K)
            pred_len: Forecast horizon steps

        Returns:
            pred_rel: (B, T_pred, 2) cumulative predicted displacements
            attn_weights: (B, num_heads, 1, K) spatial attention weights
        """
        b = obs_rel.size(0)
        p_len = pred_len or self.pred_len

        # 1. Encode ego trajectory
        ego_emb = self.ego_embedding(obs_rel)
        _, (h_enc, c_enc) = self.ego_lstm(ego_emb)
        h_ego_last = h_enc[-1]  # Top layer hidden state (B, hidden_dim)

        # 2. Encode surrounding neighbor graph (normalize scale: ~50m -> ~1.0)
        scale = torch.tensor([0.02, 0.02, 0.05, 0.05], device=neighbor_states.device).view(1, 1, 4)
        scaled_neigh = neighbor_states * scale
        neigh_emb = self.neighbor_encoder(scaled_neigh)  # (B, K, hidden_dim)

        # 3. Spatial Graph Attention Pooling
        social_context, attn_weights = self.spatial_attention(h_ego_last, neigh_emb, neighbor_masks)

        # 4. Gated Interaction Modulation of Encoder States
        gate_input = torch.cat([h_ego_last, social_context], dim=-1)
        gate = self.fusion_gate(gate_input)
        soc_mod = gate * self.fusion_proj(social_context)  # (B, hidden_dim)

        # 5. Initialize Decoder with Interaction-Modulated States
        # Preserve full encoder velocity momentum in cell state (c_enc)
        h_dec = h_enc + soc_mod.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_dec = c_enc

        # 6. Autoregressive Decoding
        current_input = obs_rel[:, -1:, :]
        step_outputs = []

        for _ in range(p_len):
            d_emb = self.dec_embedding(current_input)
            d_out, (h_dec, c_dec) = self.decoder_lstm(d_emb, (h_dec, c_dec))
            step_delta = self.out_proj(d_out)
            step_outputs.append(step_delta)
            current_input = step_delta

        delta_steps = torch.cat(step_outputs, dim=1)  # (B, T_pred, 2)
        pred_rel = torch.cumsum(delta_steps, dim=1)

        return pred_rel, attn_weights


class InteractionTrajectoryPredictor:
    """Production wrapper for loading and running the Interaction-Aware ST-GAT predictor."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = "models/interaction_stgat.pt",
        obs_len: int = 10,
        pred_len: int = 15,
        hidden_dim: int = 128,
        device: str = "auto"
    ) -> None:
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.device = self._resolve_device(device)
        self.model = STGATTrajectoryModel(pred_len=pred_len, hidden_dim=hidden_dim).to(self.device)
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
                "obs_len": int(self.obs_len),
                "pred_len": int(self.pred_len),
                "hidden_dim": int(self.model.hidden_dim),
                "embedding_dim": int(self.model.embedding_dim),
                "num_layers": int(self.model.num_layers),
                "is_trained": True
            },
            path
        )
        logger.info(f"Saved ST-GAT Interaction checkpoint to {path}")

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

        self.model = STGATTrajectoryModel(
            pred_len=self.pred_len,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.is_trained = True
        logger.info(f"Loaded trained ST-GAT Interaction model from {path} (device={self.device})")

    def predict(
        self,
        trail: List[Tuple[float, float]],
        neighbor_trails: Optional[List[List[Tuple[float, float]]]] = None,
        fps: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """Predicts future points accounting for neighboring interacting vehicles.

        Args:
            trail: List of (x, y) coordinates of ego agent
            neighbor_trails: Optional list of surrounding agent trails
            fps: Optional video frame rate
        """
        if len(trail) < 2:
            return None

        coords = np.array(trail, dtype=np.float32)
        if len(coords) < self.obs_len:
            pad_len = self.obs_len - len(coords)
            first_pt = coords[0]
            obs_traj = np.vstack([np.repeat([first_pt], pad_len, axis=0), coords])
        else:
            obs_traj = coords[-self.obs_len :]

        obs_rel = np.zeros_like(obs_traj)
        obs_rel[1:] = obs_traj[1:] - obs_traj[:-1]
        last_pos = obs_traj[-1]

        # Neighbor context
        k_max = 4
        neigh_states = np.zeros((k_max, 4), dtype=np.float32)
        neigh_masks = np.zeros(k_max, dtype=bool)

        if neighbor_trails:
            valid_k = 0
            for n_tr in neighbor_trails:
                if len(n_tr) > 0 and valid_k < k_max:
                    n_last = np.array(n_tr[-1], dtype=np.float32)
                    rel_p = n_last - last_pos
                    rel_v = (np.array(n_tr[-1]) - np.array(n_tr[-2])) if len(n_tr) > 1 else np.zeros(2)
                    neigh_states[valid_k, :2] = rel_p
                    neigh_states[valid_k, 2:] = rel_v
                    neigh_masks[valid_k] = True
                    valid_k += 1

        inp_obs = torch.from_numpy(obs_rel).unsqueeze(0).to(self.device)
        inp_neigh = torch.from_numpy(neigh_states).unsqueeze(0).to(self.device)
        inp_mask = torch.from_numpy(neigh_masks).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred_rel_tensor, _ = self.model(inp_obs, inp_neigh, inp_mask, pred_len=self.pred_len)

        pred_rel = pred_rel_tensor.squeeze(0).cpu().numpy()
        future_coords = last_pos + pred_rel
        return future_coords
