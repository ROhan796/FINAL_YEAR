"""Knowledge graph repository — Postgres reads/writes.

PostgreSQL is the source of truth for all KG data.
Neo4j is an optional projected read model for deep exploration.
"""

from __future__ import annotations

import logging
from typing import Any

from ml.graph.schema import KGEdgeData, KGNodeData

log = logging.getLogger(__name__)


class KGRepository:
    """Repository for knowledge graph operations on PostgreSQL.

    Reads/writes kg_nodes and kg_edges tables.
    """

    def __init__(self, db_session=None):
        self.db = db_session

    def upsert_node(self, node: KGNodeData, node_id: str | None = None) -> str:
        """Insert or update a KG node. Returns the node ID."""
        if self.db is None:
            log.debug("kg_repo_no_db: action=%s name=%s", "upsert_node", node.canonical_name)
            return node_id or "mock-node-id"

        from app.models.knowledge import KgNode
        from app.db.base import new_uuid
        from sqlalchemy import select

        # Check if exists by normalised name + entity type
        existing = self.db.execute(
            select(KgNode).where(
                KgNode.normalised_name == node.normalised_name,
                KgNode.entity_type == node.entity_type,
            )
        ).scalar_one_or_none()

        if existing:
            existing.mention_count += 1
            if node.canonical_name not in (existing.aliases or []):
                existing.aliases = list(existing.aliases or []) + [node.canonical_name]
            self.db.flush()
            return existing.id

        new_id = node_id or new_uuid()
        kg_node = KgNode(
            id=new_id,
            canonical_name=node.canonical_name,
            normalised_name=node.normalised_name,
            entity_type=node.entity_type,
            domain=node.domain,
            embedding=node.embedding,
            paper_id=node.paper_id,
            confidence=node.confidence,
            mention_count=node.mention_count,
            aliases=node.aliases,
        )
        self.db.add(kg_node)
        self.db.flush()
        return new_id

    def upsert_edge(self, edge: KGEdgeData) -> str:
        """Insert or update a KG edge. Returns the edge ID."""
        if self.db is None:
            log.debug("kg_repo_no_db: action=%s src=%s", "upsert_edge", edge.src_id)
            return "mock-edge-id"

        from app.models.knowledge import KgEdge
        from app.db.base import new_uuid
        from sqlalchemy import select

        existing = self.db.execute(
            select(KgEdge).where(
                KgEdge.src_id == edge.src_id,
                KgEdge.dst_id == edge.dst_id,
                KgEdge.relation == edge.relation,
            )
        ).scalar_one_or_none()

        if existing:
            existing.confidence = max(existing.confidence, edge.confidence)
            self.db.flush()
            return existing.id

        new_id = new_uuid()
        kg_edge = KgEdge(
            id=new_id,
            src_id=edge.src_id,
            dst_id=edge.dst_id,
            relation=edge.relation,
            confidence=edge.confidence,
            provenance_paper_id=edge.provenance_paper_id,
            provenance_sentence=edge.provenance_sentence,
            src_domain=edge.src_domain,
            dst_domain=edge.dst_domain,
            domain_cross=edge.domain_cross,
        )
        self.db.add(kg_edge)
        self.db.flush()
        return new_id

    def get_node_neighbors(self, node_id: str, max_hops: int = 1) -> dict[str, Any]:
        """Get neighbors of a node up to max_hops."""
        if self.db is None:
            return {"nodes": [], "edges": []}

        from app.models.knowledge import KgEdge, KgNode
        from sqlalchemy import or_

        # Get direct neighbors
        edges = self.db.execute(
            select(KgEdge).where(
                or_(KgEdge.src_id == node_id, KgEdge.dst_id == node_id)
            )
        ).scalars().all()

        neighbor_ids = set()
        for edge in edges:
            neighbor_ids.add(edge.src_id)
            neighbor_ids.add(edge.dst_id)
        neighbor_ids.discard(node_id)

        # Load neighbor nodes
        nodes = []
        if neighbor_ids:
            from sqlalchemy import select
            nodes = self.db.execute(
                select(KgNode).where(KgNode.id.in_(neighbor_ids))
            ).scalars().all()

        return {
            "nodes": nodes,
            "edges": edges,
        }

    def get_cross_domain_edges(self, limit: int = 100) -> list:
        """Get edges that cross domain boundaries."""
        if self.db is None:
            return []

        from app.models.knowledge import KgEdge
        from sqlalchemy import select

        return list(self.db.execute(
            select(KgEdge)
            .where(KgEdge.domain_cross.is_(True))
            .order_by(KgEdge.confidence.desc())
            .limit(limit)
        ).scalars().all())

    def get_all_node_embeddings(self) -> list[tuple[str, list[float]]]:
        """Get all node IDs and their embeddings for GNN training."""
        if self.db is None:
            return []

        from app.models.knowledge import KgNode
        from sqlalchemy import select

        nodes = self.db.execute(
            select(KgNode.id, KgNode.embedding).where(KgNode.embedding.is_not(None))
        ).all()

        return [(n[0], n[1]) for n in nodes if n[1] is not None]
