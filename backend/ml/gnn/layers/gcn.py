"""GCN layer — Graph Convolutional Network.

Standard GCN propagation rule:
    H^(l+1) = σ(Â · H^(l) · W^(l))

Where Â = D̃^(-1/2) · Ã · D̃^(-1/2), Ã = A + I_N
"""

from __future__ import annotations

import numpy as np


class GCNLayer:
    """Graph Convolutional Network layer (numpy implementation)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.W = np.random.randn(in_dim, out_dim).astype(np.float32) * 0.01
        self.bias = np.zeros(out_dim, dtype=np.float32)

    def forward(self, H: np.ndarray, A_norm: np.ndarray) -> np.ndarray:
        """Forward pass: H' = σ(A_norm @ H @ W + b)."""
        out = A_norm @ H @ self.W + self.bias
        return np.maximum(out, 0)  # ReLU
