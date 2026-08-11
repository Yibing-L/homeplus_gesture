#!/usr/bin/env python3
"""Model definitions and structured adapters for Home+ pipeline v2."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_pipeline_v2 import HAND_PARENTS, flatten_sequence


def prepare_flat(data: Dict[str, np.ndarray], feature_set: str) -> np.ndarray:
    return flatten_sequence(data, feature_set)[0].astype(np.float32)


def prepare_graph(data: Dict[str, np.ndarray], feature_set: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return node features (C,T,V) and global features (T,G)."""
    node = [data["hand_pos"], data["hand_vel"]]
    if feature_set in {"bones", "full"}:
        node.append(data["hand_bone"])
    node.append(data["joint_valid"][:, :, None])
    node_ctv = np.concatenate(node, axis=2).transpose(2, 0, 1).astype(np.float32)

    glob = [
        data["arm_pos"].reshape(len(data["arm_pos"]), -1),
        data["arm_vel"].reshape(len(data["arm_vel"]), -1),
        data["global_features"][:, :15],
    ]
    if feature_set in {"angles", "full"}:
        glob.append(data["hand_angles"])
    glob.extend((data["arm_valid"], data["frame_valid"][:, None]))
    if feature_set in {"angles", "full"}:
        glob.append(data["angle_valid"])
    return node_ctv, np.concatenate(glob, axis=1).astype(np.float32)


def hand_adjacency() -> torch.Tensor:
    a = np.eye(21, dtype=np.float32)
    for j in range(1, 21):
        p = int(HAND_PARENTS[j])
        a[j, p] = a[p, j] = 1.0
    deg = a.sum(axis=1)
    d = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-6)))
    return torch.from_numpy(d @ a @ d)


class TemporalAttentionPool(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(x.size(0), -1, -1)
        pooled, _ = self.attn(q, x, x, key_padding_mask=~mask.bool())
        return pooled[:, 0]


class AttentionBiLSTM(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128,
                 layers: int = 2, proj: int = 128, heads: int = 4, dropout: float = 0.4):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, proj), nn.LayerNorm(proj), nn.GELU())
        self.rnn = nn.LSTM(proj, hidden, layers, batch_first=True, bidirectional=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.residual = nn.Linear(hidden * 2, proj)
        self.norm = nn.LayerNorm(proj)
        self.pool = TemporalAttentionPool(proj, heads)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(proj, n_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, global_x=None) -> torch.Tensor:
        # Do not use pack_padded_sequence: validity holes can occur inside clips.
        h0 = self.input_proj(x)
        h, _ = self.rnn(h0)
        h = self.norm(self.residual(h) + h0)
        return self.head(self.pool(h, mask))


class TCNBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        pad = 2 * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(width, width, 3, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, width)
        self.norm2 = nn.GroupNorm(1, width)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(-1)
        h = self.drop(self.act(self.norm1(self.conv1(x)[..., :t])))
        h = self.drop(self.norm2(self.conv2(h)[..., :t]))
        return self.act(x + h)


class GestureTCN(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, width: int = 128,
                 blocks: int = 5, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, width), nn.LayerNorm(width), nn.GELU())
        self.blocks = nn.ModuleList([TCNBlock(width, 2 ** i, dropout) for i in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, n_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, global_x=None) -> torch.Tensor:
        h = self.input_proj(x).transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)
        m = mask[:, :h.size(1), None].float()
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        return self.head(pooled)


class STGCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, adjacency: torch.Tensor,
                 stride: int = 1, dropout: float = 0.2):
        super().__init__()
        self.register_buffer("adjacency", adjacency.clone())
        self.spatial = nn.Conv2d(c_in, c_out, 1)
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(c_out), nn.GELU(),
            nn.Conv2d(c_out, c_out, (9, 1), stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(c_out), nn.Dropout(dropout),
        )
        if c_in == c_out and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(c_in, c_out, 1, stride=(stride, 1))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Aggregate physical neighbours, then learn channel mixing.
        graph_x = torch.einsum("bctv,vw->bctw", x, self.adjacency)
        return self.act(self.temporal(self.spatial(graph_x)) + self.residual(x))


class LightweightSTGCN(nn.Module):
    def __init__(self, node_channels: int, global_dim: int, n_classes: int,
                 width: int = 64, dropout: float = 0.3):
        super().__init__()
        a = hand_adjacency()
        self.blocks = nn.ModuleList([
            STGCNBlock(node_channels, width, a, dropout=dropout),
            STGCNBlock(width, width, a, dropout=dropout),
            STGCNBlock(width, width * 2, a, stride=2, dropout=dropout),
            STGCNBlock(width * 2, width * 2, a, dropout=dropout),
            STGCNBlock(width * 2, width * 2, a, stride=2, dropout=dropout),
        ])
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(width * 3), nn.Dropout(dropout), nn.Linear(width * 3, n_classes)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, global_x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h)
        # Temporal strides reduce T, so resize the real-frame mask before pooling.
        graph_mask = F.interpolate(mask[:, None].float(), size=h.size(2), mode="nearest")[:, 0]
        graph_mask = graph_mask[:, None, :, None]
        graph_pooled = (h * graph_mask).sum(dim=(2, 3))
        graph_pooled /= (graph_mask.sum(dim=2).squeeze(-1) * h.size(3)).clamp_min(1.0)
        m = mask[:, :, None].float()
        g = self.global_encoder(global_x)
        global_pooled = (g * m).sum(1) / m.sum(1).clamp_min(1.0)
        return self.head(torch.cat((graph_pooled, global_pooled), dim=1))


def build_model(model_type: str, n_classes: int, config: Dict) -> nn.Module:
    if model_type == "tcn":
        return GestureTCN(config["in_dim"], n_classes, config.get("width", 128),
                          config.get("blocks", 5), config.get("dropout", 0.3))
    if model_type == "bilstm":
        return AttentionBiLSTM(
            config["in_dim"], n_classes, config.get("hidden", 128), config.get("layers", 2),
            config.get("proj", 128), config.get("heads", 4), config.get("dropout", 0.4),
        )
    if model_type == "stgcn":
        return LightweightSTGCN(config["node_channels"], config["global_dim"], n_classes,
                                config.get("width", 64), config.get("dropout", 0.3))
    raise ValueError(f"Unsupported model_type={model_type!r}")
