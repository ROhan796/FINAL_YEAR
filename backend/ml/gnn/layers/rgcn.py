"""R-GCN layer — PyTorch Relational Graph Convolutional Network with basis decomposition.

For typed edges in the knowledge graph:
    h_v^(l+1) = ReLU(W_0 h_v + sum_r sum_{u in N_r(v)} (1/|N_r(v)|) W_r h_u + b)
Basis decomposition:
    W_r = sum_b a_{rb} * V_b
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RGCNLayer(nn.Module):
    """Relational GCN layer with basis decomposition (PyTorch)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        num_bases: int = 30,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = min(num_bases, max(1, num_relations * 2))

        self.V = nn.Parameter(torch.empty(self.num_bases, in_dim, out_dim))
        self.a = nn.Parameter(torch.empty(num_relations, self.num_bases))
        self.W_self = nn.Linear(in_dim, out_dim)

        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.W_self.weight)

    def get_weight(self, relation_idx: int) -> torch.Tensor:
        idx = relation_idx % self.num_relations
        coeffs = self.a[idx]  # (num_bases,)
        return torch.einsum("b,bij->ij", coeffs, self.V)

    def forward(
        self,
        H: torch.Tensor,
        adj_by_relation: dict[int, list[tuple[int, int]]],
        num_nodes: int,
    ) -> torch.Tensor:
        out = self.W_self(H)
        device = H.device

        for rel_idx, edges in adj_by_relation.items():
            if not edges:
                continue
            valid = [(s, d) for s, d in edges if 0 <= s < num_nodes and 0 <= d < num_nodes]
            if not valid:
                continue
            src = torch.tensor([e[0] for e in valid], dtype=torch.long, device=device)
            dst = torch.tensor([e[1] for e in valid], dtype=torch.long, device=device)

            agg = torch.zeros((num_nodes, self.in_dim), dtype=H.dtype, device=device)
            agg.index_add_(0, dst, H[src])

            deg = torch.zeros(num_nodes, dtype=H.dtype, device=device)
            deg.index_add_(0, dst, torch.ones(len(valid), dtype=H.dtype, device=device))
            agg = agg / deg.clamp(min=1.0).unsqueeze(-1)

            W_r = self.get_weight(rel_idx)
            out = out + agg @ W_r

        return torch.relu(out)


__all__ = ["RGCNLayer"]
