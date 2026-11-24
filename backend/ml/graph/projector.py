"""Incremental Neo4j graph projector (Postgres -> Neo4j read model).

Follows the dual-database rule (docs/01 §3.2):
- Postgres `kg_nodes` / `kg_edges` is the ONLY source of truth.
- This projector tails `kg_edges.updated_at` using `graph_projection_state`
  watermark and performs idempotent Cypher MERGE upserts into Neo4j.
- Safe to run repeatedly, replayable from zero (`full=True`), and degrades
  cleanly when Neo4j is disabled or unreachable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.knowledge import GraphProjectionState, KgEdge, KgNode

log = logging.getLogger(__name__)


@dataclass
class ProjectionResult:
    status: str  # "ok" | "skipped" | "error"
    nodes_projected: int = 0
    edges_projected: int = 0
    watermark: str | None = None
    error: str | None = None


class Neo4jProjector:
    """Watermark-based incremental projector from Postgres KG to Neo4j."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def state(self) -> dict[str, Any]:
        row = self.db.get(GraphProjectionState, 1)
        return {
            "neo4jEnabled": settings.neo4j_enabled,
            "lastProjectedAt": row.last_projected_at.isoformat() if row and row.last_projected_at else None,
            "lastEdgeUpdatedAt": row.last_edge_updated_at.isoformat() if row and row.last_edge_updated_at else None,
            "nodesProjected": row.nodes_projected if row else 0,
            "edgesProjected": row.edges_projected if row else 0,
            "error": row.error if row else None,
        }

    def project(self, *, full: bool = False) -> ProjectionResult:
        state_row = self.db.get(GraphProjectionState, 1)
        if state_row is None:
            state_row = GraphProjectionState(id=1, nodes_projected=0, edges_projected=0)
            self.db.add(state_row)
            self.db.flush()

        watermark = None if full else state_row.last_edge_updated_at

        edge_stmt = select(KgEdge).order_by(KgEdge.updated_at)
        if watermark is not None:
            edge_stmt = edge_stmt.where(KgEdge.updated_at > watermark)
        edges = list(self.db.scalars(edge_stmt.limit(settings.graph_projector_batch)).all())

        node_ids = {e.src_id for e in edges} | {e.dst_id for e in edges}
        if full:
            nodes = list(self.db.scalars(select(KgNode).limit(settings.graph_projector_batch * 2)).all())
        elif node_ids:
            nodes = list(self.db.scalars(select(KgNode).where(KgNode.id.in_(node_ids))).all())
        else:
            nodes = []

        if not settings.neo4j_enabled:
            # Record dry-run / standby watermark without failing
            state_row.error = "neo4j_enabled=false (standby)"
            self.db.commit()
            return ProjectionResult(
                status="skipped",
                nodes_projected=0,
                edges_projected=0,
                watermark=watermark.isoformat() if watermark else None,
                error="Neo4j disabled in settings",
            )

        if not edges and not nodes:
            return ProjectionResult(
                status="ok",
                nodes_projected=0,
                edges_projected=0,
                watermark=watermark.isoformat() if watermark else None,
            )

        try:
            from app.modules.graph.service import _get_neo4j_driver

            driver = _get_neo4j_driver()
            node_params = [
                {
                    "id": n.id,
                    "label": n.canonical_name,
                    "entityType": n.entity_type,
                    "domain": n.domain,
                    "confidence": float(n.confidence),
                    "mentionCount": int(n.mention_count),
                }
                for n in nodes
            ]
            edge_params = [
                {
                    "id": e.id,
                    "srcId": e.src_id,
                    "dstId": e.dst_id,
                    "relation": e.relation,
                    "confidence": float(e.confidence),
                    "domainCross": bool(e.domain_cross),
                    "paperId": e.provenance_paper_id or "",
                }
                for e in edges
            ]

            with driver.session(database=settings.neo4j_database) as session:
                if node_params:
                    session.run(
                        """
                        UNWIND $rows AS row
                        MERGE (n:Entity {id: row.id})
                        SET n.label = row.label,
                            n.entityType = row.entityType,
                            n.domain = row.domain,
                            n.confidence = row.confidence,
                            n.mentionCount = row.mentionCount
                        """,
                        rows=node_params,
                    )
                if edge_params:
                    session.run(
                        """
                        UNWIND $rows AS row
                        MATCH (a:Entity {id: row.srcId})
                        MATCH (b:Entity {id: row.dstId})
                        MERGE (a)-[r:RELATES {id: row.id}]->(b)
                        SET r.relation = row.relation,
                            r.confidence = row.confidence,
                            r.domainCross = row.domainCross,
                            r.paperId = row.paperId
                        """,
                        rows=edge_params,
                    )

            latest_edge_ts = max((e.updated_at for e in edges if e.updated_at), default=watermark)
            state_row.last_projected_at = datetime.now(UTC)
            state_row.last_edge_updated_at = latest_edge_ts
            state_row.nodes_projected = int(state_row.nodes_projected or 0) + len(node_params)
            state_row.edges_projected = int(state_row.edges_projected or 0) + len(edge_params)
            state_row.error = None
            self.db.commit()

            log.info("neo4j_projected: nodes=%d edges=%d", len(node_params), len(edge_params))
            return ProjectionResult(
                status="ok",
                nodes_projected=len(node_params),
                edges_projected=len(edge_params),
                watermark=latest_edge_ts.isoformat() if latest_edge_ts else None,
            )
        except Exception as exc:  # noqa: BLE001
            state_row.error = str(exc)[:500]
            self.db.commit()
            log.warning("neo4j_projection_failed: %s", exc)
            return ProjectionResult(status="error", error=str(exc))


__all__ = ["Neo4jProjector", "ProjectionResult"]
