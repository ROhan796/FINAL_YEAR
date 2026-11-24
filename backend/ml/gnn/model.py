"""CKGAI GNN Model — PyTorch R-GCN + GAT + Virtual Node.

Spec:
- L = 3 GNN layers
- Hidden dim = 256
- R-GCN with basis decomposition (B = 30)
- GAT with K = 8 heads
- Dropout p = 0.1 between layers
- Optional virtual global node
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.gnn.layers import GATLayer, RGCNLayer, VirtualNode

log = logging.getLogger(__name__)


@dataclass
class GNNOutput:
    """Output from GNN forward pass."""

    node_embeddings: np.ndarray  # (N, out_dim)
    layer_outputs: list[np.ndarray]


class CKGAI_GNN(nn.Module):
    """PyTorch CKGAI Graph Neural Network (R-GCN + GAT + VirtualNode)."""

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        out_dim: int = 256,
        num_layers: int = 3,
        num_relations: int = 9,
        rgcn_bases: int = 30,
        gat_heads: int = 8,
        dropout: float = 0.1,
        use_virtual_node: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.dropout_p = dropout
        self.use_virtual_node = use_virtual_node

        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.rgcn_layers = nn.ModuleList()
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.rgcn_layers.append(RGCNLayer(hidden_dim, hidden_dim, num_relations, rgcn_bases))
            self.gat_layers.append(GATLayer(hidden_dim, hidden_dim, gat_heads))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.virtual_node = VirtualNode(hidden_dim) if use_virtual_node else None
        self.out_proj = nn.Linear(hidden_dim, out_dim) if out_dim != hidden_dim else nn.Identity()
        self.rel_embeddings = nn.Parameter(torch.randn(max(1, num_relations), out_dim) * 0.1)

    @staticmethod
    def _build_adj_list(
        adj_by_relation: dict[int, list[tuple[int, int]]],
        num_nodes: int,
    ) -> list[list[int]]:
        adj: list[list[int]] = [[] for _ in range(num_nodes)]
        for edges in adj_by_relation.values():
            for s, d in edges:
                if 0 <= s < num_nodes and 0 <= d < num_nodes:
                    adj[d].append(s)
                    adj[s].append(d)
        return adj

    def forward_tensor(
        self,
        node_features: torch.Tensor,
        adj_by_relation: dict[int, list[tuple[int, int]]],
        adj_list: list[list[int]] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        N = node_features.size(0)
        if N == 0:
            empty = torch.zeros((0, self.out_dim), dtype=node_features.dtype, device=node_features.device)
            return empty, [empty]

        if adj_list is None:
            adj_list = self._build_adj_list(adj_by_relation, N)

        H = F.relu(self.input_proj(node_features))
        layer_outputs = [H]

        for i in range(self.num_layers):
            H_rgcn = self.rgcn_layers[i](H, adj_by_relation, N)
            H_gat = self.gat_layers[i](H_rgcn, adj_list, N)
            H_next = self.norms[i](H + 0.5 * (H_rgcn + H_gat))
            if self.virtual_node is not None:
                H_next = self.virtual_node(H_next)
            if self.training and self.dropout_p > 0:
                H_next = F.dropout(H_next, p=self.dropout_p, training=True)
            H = H_next
            layer_outputs.append(H)

        Z = F.normalize(self.out_proj(H), p=2, dim=-1)
        return Z, layer_outputs

    def forward(
        self,
        node_features: np.ndarray | torch.Tensor,
        adj_by_relation: dict[int, list[tuple[int, int]]],
        adj_list: list[list[int]] | None = None,
    ) -> GNNOutput:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x = (
                node_features
                if isinstance(node_features, torch.Tensor)
                else torch.as_tensor(node_features, dtype=torch.float32)
            )
            x = x.to(self.rel_embeddings.device)
            Z, layers = self.forward_tensor(x, adj_by_relation, adj_list)
        if was_training:
            self.train()
        return GNNOutput(
            node_embeddings=Z.cpu().numpy().astype(np.float32),
            layer_outputs=[t.detach().cpu().numpy().astype(np.float32) for t in layers],
        )

    def predict_embeddings(
        self,
        node_features: np.ndarray | torch.Tensor,
        adj_by_relation: dict[int, list[tuple[int, int]]],
        adj_list: list[list[int]] | None = None,
    ) -> np.ndarray:
        return self.forward(node_features, adj_by_relation, adj_list).node_embeddings

    def relation_embedding_matrix(self) -> np.ndarray:
        return self.rel_embeddings.detach().cpu().numpy().astype(np.float32)

    def save_checkpoint(self, path: str | Path, metadata: dict | None = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": {
                    "in_dim": self.in_dim,
                    "hidden_dim": self.hidden_dim,
                    "out_dim": self.out_dim,
                    "num_layers": self.num_layers,
                    "num_relations": self.num_relations,
                    "dropout": self.dropout_p,
                    "use_virtual_node": self.use_virtual_node,
                },
                "state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            p,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str = "cpu") -> CKGAI_GNN:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        cfg = ckpt.get("config", {})
        model = cls(**cfg).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    def warmup(self) -> None:
        dummy_features = np.random.randn(10, self.in_dim).astype(np.float32)
        dummy_adj = {0: [(0, 1), (1, 2), (2, 3)]}
        _ = self.forward(dummy_features, dummy_adj)
        log.info("gnn_warmup_complete (pytorch)")


__all__ = ["CKGAI_GNN", "GNNOutput"]
