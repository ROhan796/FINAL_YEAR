"""GNN dataset — load graph data from Postgres for training."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class GraphData:
    """Graph data for GNN training."""
    node_features: np.ndarray  # (N, feature_dim)
    node_ids: list[str]
    adj_by_relation: dict[int, list[tuple[int, int]]]
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_types: np.ndarray
    labels: np.ndarray | None = None


class GraphDataset:
    """Load and prepare graph data from PostgreSQL for GNN training."""

    def __init__(self, db_session=None):
        self.db = db_session

    def load_graph(self, feature_dim: int = 768) -> GraphData:
        """Load the full knowledge graph from Postgres."""
        if self.db is None:
            return self._mock_graph(feature_dim)

        from app.models.knowledge import KgNode, KgEdge
        from sqlalchemy import select

        # Load nodes
        nodes = list(self.db.execute(select(KgNode)).scalars().all())
        node_id_to_idx = {n.id: i for i, n in enumerate(nodes)}
        N = len(nodes)

        node_features = np.zeros((N, feature_dim), dtype=np.float32)
        node_ids = []
        for i, node in enumerate(nodes):
            node_ids.append(node.id)
            if node.embedding is not None:
                try:
                    emb = np.array(node.embedding, dtype=np.float32)
                    if emb.shape[0] == feature_dim:
                        node_features[i] = emb
                except Exception:
                    pass

        # Load edges
        edges = list(self.db.execute(select(KgEdge)).scalars().all())
        adj_by_relation: dict[int, list[tuple[int, int]]] = {}
        edge_srcs = []
        edge_dsts = []
        edge_types = []

        relation_map = {}
        for edge in edges:
            if edge.relation not in relation_map:
                relation_map[edge.relation] = len(relation_map)
            rel_idx = relation_map[edge.relation]

            src_idx = node_id_to_idx.get(edge.src_id)
            dst_idx = node_id_to_idx.get(edge.dst_id)
            if src_idx is not None and dst_idx is not None:
                adj_by_relation.setdefault(rel_idx, []).append((src_idx, dst_idx))
                edge_srcs.append(src_idx)
                edge_dsts.append(dst_idx)
                edge_types.append(rel_idx)

        return GraphData(
            node_features=node_features,
            node_ids=node_ids,
            adj_by_relation=adj_by_relation,
            edge_src=np.array(edge_srcs, dtype=np.int64),
            edge_dst=np.array(edge_dsts, dtype=np.int64),
            edge_types=np.array(edge_types, dtype=np.int64),
        )

    def _mock_graph(self, feature_dim: int = 768) -> GraphData:
        """Create a small mock graph for testing."""
        N = 20
        node_features = np.random.randn(N, feature_dim).astype(np.float32)
        node_ids = [f"mock-node-{i}" for i in range(N)]
        adj_by_relation = {
            0: [(i, (i + 1) % N) for i in range(0, N, 2)],
            1: [(i, (i + 2) % N) for i in range(0, N, 3)],
        }
        edge_srcs = [s for edges in adj_by_relation.values() for s, _ in edges]
        edge_dsts = [d for edges in adj_by_relation.values() for _, d in edges]
        edge_types = [r for r, edges in adj_by_relation.items() for _ in edges]

        return GraphData(
            node_features=node_features,
            node_ids=node_ids,
            adj_by_relation=adj_by_relation,
            edge_src=np.array(edge_srcs, dtype=np.int64),
            edge_dst=np.array(edge_dsts, dtype=np.int64),
            edge_types=np.array(edge_types, dtype=np.int64),
        )


__all__ = ["GraphDataset", "GraphData"]
