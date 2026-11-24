"""GAT layer — PyTorch Multi-Head Graph Attention Network layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Multi-head Graph Attention Network layer (PyTorch)."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.num_heads = num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.head_dim = out_dim // num_heads

        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn)

    def forward(
        self,
        H: torch.Tensor,
        adj_list: list[list[int]] | None,
        num_nodes: int,
    ) -> torch.Tensor:
        if adj_list is None or num_nodes == 0:
            return F.elu(self.proj(H))

        K, d = self.num_heads, self.head_dim
        projected = self.proj(H).view(num_nodes, K, d)  # (N, K, d)

        edges_src: list[int] = []
        edges_dst: list[int] = []
        for v in range(min(num_nodes, len(adj_list))):
            for u in adj_list[v]:
                if 0 <= u < num_nodes:
                    edges_src.append(u)
                    edges_dst.append(v)

        if not edges_src:
            return F.elu(projected.view(num_nodes, K * d))

        device = H.device
        src = torch.tensor(edges_src, dtype=torch.long, device=device)
        dst = torch.tensor(edges_dst, dtype=torch.long, device=device)

        h_src = projected[src]  # (E, K, d)
        h_dst = projected[dst]  # (E, K, d)
        cat = torch.cat([h_dst, h_src], dim=-1)  # (E, K, 2d)
        e = self.leaky_relu((cat * self.attn.unsqueeze(0)).sum(dim=-1))  # (E, K)

        # Numerically stable segment softmax per destination node & head
        e_exp = torch.exp(e - e.max(dim=0, keepdim=True).values)
        denom = torch.zeros((num_nodes, K), dtype=H.dtype, device=device)
        denom.index_add_(0, dst, e_exp)
        alpha = e_exp / denom[dst].clamp(min=1e-9)  # (E, K)

        weighted = h_src * alpha.unsqueeze(-1)  # (E, K, d)
        out = torch.zeros((num_nodes, K, d), dtype=H.dtype, device=device)
        out.index_add_(0, dst, weighted)
        return F.elu(out.view(num_nodes, K * d))


__all__ = ["GATLayer"]
