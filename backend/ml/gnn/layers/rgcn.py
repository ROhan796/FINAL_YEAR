"""R-GCN layer — Relational Graph Convolutional Network.

For typed edges in the knowledge graph:
    h_v^(l+1) = σ(W_0 h_v + Σ_r Σ_u∈N_r(v) (1/|N_r(v)|) W_r h_u)

Basis decomposition (parameter sharing):
    W_r = Σ_b a_rb · V_b
"""

from __future__ import annotations

import numpy as np


class RGCNLayer:
    """Relational GCN layer with basis decomposition (numpy)."""

    def __init__(self, in_dim: int, out_dim: int, num_relations: int,
                 num_bases: int = 30) -> None:
        self.num_relations = num_relations
        self.num_bases = num_bases

        # Basis matrices
        self.V = np.random.randn(num_bases, in_dim, out_dim).astype(np.float32) * 0.01
        # Relation-specific coefficients
        self.a = np.random.randn(num_relations, num_bases).astype(np.float32) * 0.01
        # Self-connection weight
        self.W_self = np.random.randn(in_dim, out_dim).astype(np.float32) * 0.01
        self.bias = np.zeros(out_dim, dtype=np.float32)

    def get_weight(self, relation_idx: int) -> np.ndarray:
        """Get weight matrix for a relation type via basis decomposition."""
        # W_r = Σ_b a_rb · V_b
        coeffs = self.a[relation_idx]  # (num_bases,)
        W_r = np.einsum("b,bij->ij", coeffs, self.V)  # (in_dim, out_dim)
        return W_r

    def forward(self, H: np.ndarray, adj_by_relation: dict[int, list[tuple[int, int]]],
                num_nodes: int) -> np.ndarray:
        """Forward pass with relation-specific adjacency lists.

        Args:
            H: Node features (N, in_dim)
            adj_by_relation: {relation_idx: [(src, dst), ...]}
            num_nodes: Number of nodes
        """
        N = num_nodes
        out = H @ self.W_self + self.bias  # Self-connection

        # Aggregate per relation
        for rel_idx, edges in adj_by_relation.items():
            if not edges:
                continue
            W_r = self.get_weight(rel_idx)

            # Count neighbors per node for this relation
            neighbor_count = np.zeros(N, dtype=np.float32)
            agg = np.zeros((N, H.shape[1]), dtype=np.float32)
            for src, dst in edges:
                if src < N and dst < N:
                    agg[dst] += H[src]
                    neighbor_count[dst] += 1

            # Normalize and apply weight
            mask = neighbor_count > 0
            agg[mask] /= neighbor_count[mask, np.newaxis]
            out += agg @ W_r

        return np.maximum(out, 0)  # ReLU
