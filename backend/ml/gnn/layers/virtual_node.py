"""Virtual global node — PyTorch module for 1-hop global graph context."""

from __future__ import annotations

import torch
import torch.nn as nn


class VirtualNode(nn.Module):
    """Virtual global node with learnable gated residual broadcast."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
        )
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        if H.size(0) == 0:
            return H
        global_ctx = self.mlp(H.mean(dim=0, keepdim=True))  # (1, dim)
        return H + self.gate * global_ctx.expand_as(H)


__all__ = ["VirtualNode"]
