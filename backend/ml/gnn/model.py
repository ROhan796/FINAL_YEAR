"""CKGAI GNN Model — R-GCN + GAT + optional virtual node.

Per spec:
- L = 3 GNN layers
- Hidden dim = 256
- R-GCN with basis decomposition (B = 30)
- GAT with K = 8 heads
- Dropout p = 0.1 between layers
- Optional virtual node for global context

TRAINING STATUS: Architecture defined. Random weights.
Without trained weights, produces random embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ml.gnn.layers import GCNLayer, GATLayer, RGCNLayer, VirtualNode

log = logging.getLogger(__name__)


@dataclass
class GNNOutput:
    """Output from GNN forward pass."""
    node_embeddings: np.ndarray  # (N, hidden_dim)
    layer_outputs: list[np.ndarray]  # embeddings at each layer


class CKGAI_GNN:
    """CKGAI Graph Neural Network model.

    Architecture: R-GCN + GAT with optional virtual node.
    All layers use numpy for CPU-only environments.
    """

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
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_virtual_node = use_virtual_node

        # Build layers
        self.rgcn_layers: list[RGCNLayer] = []
        self.gat_layers: list[GATLayer] = []
        self.virtual_node: VirtualNode | None = None

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            self.rgcn_layers.append(
                RGCNLayer(dims[i], hidden_dim, num_relations, rgcn_bases)
            )
            self.gat_layers.append(
                GATLayer(hidden_dim, hidden_dim, gat_heads)
            )

        if use_virtual_node:
            self.virtual_node = VirtualNode(hidden_dim)

    def forward(
        self,
        node_features: np.ndarray,
        adj_by_relation: dict[int, list[tuple[int, int]]],
        adj_list: list[list[int]] | None = None,
    ) -> GNNOutput:
        """Forward pass through all GNN layers.

        Args:
            node_features: (N, in_dim) initial node features
            adj_by_relation: {relation_idx: [(src, dst), ...]}
            adj_list: Adjacency list for GAT (optional)

        Returns:
            GNNOutput with final embeddings and per-layer outputs.
        """
        N = node_features.shape[0]
        H = node_features.copy()
        layer_outputs = [H.copy()]

        for layer_idx in range(self.num_layers):
            # R-GCN message passing
            H_rgcn = self.rgcn_layers[layer_idx].forward(
                H, adj_by_relation, N
            )

            # GAT attention (if adjacency list provided)
            if adj_list is not None:
                H_gat = self.gat_layers[layer_idx].forward(
                    H_rgcn, adj_list, N
                )
                H = H_gat
            else:
                H = H_rgcn

            # Virtual node aggregation
            if self.virtual_node is not None:
                _ = self.virtual_node.aggregate(H)
                H = H + self.virtual_node.broadcast(N) * 0.1  # residual

            # Dropout
            if self.dropout > 0:
                mask = np.random.binomial(1, 1 - self.dropout, size=H.shape)
                H = H * mask / (1 - self.dropout)

            layer_outputs.append(H.copy())

        return GNNOutput(node_embeddings=H, layer_outputs=layer_outputs)

    def predict_embeddings(
        self,
        node_features: np.ndarray,
        adj_by_relation: dict[int, list[tuple[int, int]]],
        adj_list: list[list[int]] | None = None,
    ) -> np.ndarray:
        """Convenience method: return only final embeddings."""
        output = self.forward(node_features, adj_by_relation, adj_list)
        return output.node_embeddings

    def warmup(self) -> None:
        """Test forward pass with dummy data."""
        dummy_features = np.random.randn(10, self.in_dim).astype(np.float32)
        dummy_adj = {0: [(0, 1), (1, 2), (2, 3)]}
        _ = self.forward(dummy_features, dummy_adj)
        log.info("gnn_warmup_complete")
