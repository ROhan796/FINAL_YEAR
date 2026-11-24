"""GCN layer — PyTorch Graph Convolutional Network layer."""

from __future__ import annotations

import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """Graph Convolutional Network layer: H' = ReLU(A_norm @ H @ W + b)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, H: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(A_norm @ H))


__all__ = ["GCNLayer"]
