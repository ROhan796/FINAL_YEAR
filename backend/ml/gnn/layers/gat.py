"""GAT layer — Graph Attention Network.

Attention coefficient:
    e_vu = LeakyReLU(a^T [W·h_v || W·h_u])

Normalized attention:
    α_vu = exp(e_vu) / Σ_k exp(e_vk)

Multi-head update:
    h_v' = ||_k σ(Σ_u α_vu^(k) W^(k) h_u)
"""

from __future__ import annotations

import numpy as np


class GATLayer:
    """Graph Attention Network layer with multi-head attention (numpy)."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 8) -> None:
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0

        # Shared projection
        self.W = np.random.randn(in_dim, out_dim).astype(np.float32) * 0.01
        # Attention vector per head
        self.a = np.random.randn(num_heads, 2 * self.head_dim).astype(np.float32) * 0.01
        self.leaky_relu_slope = 0.2

    def forward(self, H: np.ndarray, adj_list: list[list[int]], num_nodes: int) -> np.ndarray:
        """Forward pass with neighbor lists."""
        N = num_nodes
        K = self.num_heads
        d = self.head_dim

        # Project: (N, in_dim) -> (N, K, d)
        projected = (H @ self.W).reshape(N, K, d)
        out = np.zeros((N, K, d), dtype=np.float32)

        for v in range(N):
            neighbors = adj_list[v] if v < len(adj_list) else []
            if not neighbors:
                continue

            # Compute attention scores
            h_v = projected[v]  # (K, d)
            scores = []
            for u in neighbors:
                h_u = projected[u]  # (K, d)
                # a^T [W·h_v || W·h_u] for each head
                concat = np.concatenate([h_v, h_u], axis=-1)  # (K, 2d)
                e = np.sum(self.a * concat, axis=-1)  # (K,)
                e = np.maximum(e * self.leaky_relu_slope, 0)  # LeakyReLU
                scores.append(e)

            scores = np.stack(scores, axis=-1)  # (K, num_neighbors)
            # Softmax over neighbors
            scores_max = scores.max(axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            alpha = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-9)

            # Weighted sum
            neighbor_stack = np.stack([projected[u] for u in neighbors], axis=0)  # (num_neighbors, K, d)
            alpha_expanded = alpha[:, :, np.newaxis]  # (K, num_neighbors, 1)
            out[v] = (alpha_expanded * neighbor_stack).sum(axis=0)  # (K, d)

        out = np.maximum(out, 0)  # ReLU
        return out.reshape(N, -1)  # (N, K*d)
