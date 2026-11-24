"""GNN dataset — load knowledge graph from PostgreSQL/SQLite for PyTorch training."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class GraphData:
    """Graph data for GNN training and discovery."""

    node_features: np.ndarray  # (N, feature_dim)
    node_ids: list[str]
    adj_by_relation: dict[int, list[tuple[int, int]]]
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_types: np.ndarray
    node_names: list[str] = field(default_factory=list)
    node_domains: list[str] = field(default_factory=list)
    node_types: list[str] = field(default_factory=list)
    relation_names: list[str] = field(default_factory=list)
    labels: np.ndarray | None = None


class GraphDataset:
    """Load and prepare knowledge graph data from SQLAlchemy session."""

    def __init__(self, db_session=None):
        self.db = db_session

    def load_graph(self, feature_dim: int = 768) -> GraphData:
        """Load the full knowledge graph from DB (or mock graph when db is None)."""
        if self.db is None:
            return self._mock_graph(feature_dim)

        from sqlalchemy import select

        from app.models.knowledge import KgEdge, KgNode

        nodes = list(self.db.execute(select(KgNode)).scalars().all())
        if not nodes:
            return self._mock_graph(feature_dim)

        node_id_to_idx = {n.id: i for i, n in enumerate(nodes)}
        N = len(nodes)

        node_features = np.zeros((N, feature_dim), dtype=np.float32)
        node_ids: list[str] = []
        node_names: list[str] = []
        node_domains: list[str] = []
        node_types: list[str] = []

        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for i, node in enumerate(nodes):
            node_ids.append(node.id)
            node_names.append(node.canonical_name)
            node_domains.append(node.domain or "General")
            node_types.append(node.entity_type or "METHOD")
            loaded = False
            if node.embedding is not None:
                try:
                    emb = np.asarray(node.embedding, dtype=np.float32)
                    if emb.shape == (feature_dim,) and np.linalg.norm(emb) > 0:
                        node_features[i] = emb
                        loaded = True
                except Exception:
                    pass
            if not loaded:
                missing_indices.append(i)
                missing_texts.append(f"{node.canonical_name} ({node.entity_type}, {node.domain})")

        if missing_texts:
            try:
                from ml.serving.embed_store import embed_entities

                embs = embed_entities(missing_texts)
                for idx, vec in zip(missing_indices, embs, strict=False):
                    node_features[idx] = vec
            except Exception:
                rng = np.random.default_rng(42)
                for idx in missing_indices:
                    v = rng.standard_normal(feature_dim).astype(np.float32)
                    node_features[idx] = v / (np.linalg.norm(v) + 1e-9)

        edges = list(self.db.execute(select(KgEdge)).scalars().all())
        adj_by_relation: dict[int, list[tuple[int, int]]] = {}
        edge_srcs: list[int] = []
        edge_dsts: list[int] = []
        edge_types: list[int] = []

        relation_map: dict[str, int] = {}
        relation_names: list[str] = []
        for edge in edges:
            if edge.relation not in relation_map:
                relation_map[edge.relation] = len(relation_map)
                relation_names.append(edge.relation)
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
            node_names=node_names,
            node_domains=node_domains,
            node_types=node_types,
            relation_names=relation_names,
        )

    def _mock_graph(self, feature_dim: int = 768) -> GraphData:
        N = 20
        rng = np.random.default_rng(7)
        node_features = rng.standard_normal((N, feature_dim)).astype(np.float32)
        node_ids = [f"mock-node-{i}" for i in range(N)]
        node_names = [f"Entity-{i}" for i in range(N)]
        node_domains = ["Computer Science" if i % 2 == 0 else "Medicine" for i in range(N)]
        node_types = ["METHOD" if i % 2 == 0 else "DISEASE" for i in range(N)]
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
            node_names=node_names,
            node_domains=node_domains,
            node_types=node_types,
            relation_names=["used_in", "applied_to"],
        )


__all__ = ["GraphDataset", "GraphData"]
