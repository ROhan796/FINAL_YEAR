"""Knowledge-graph exploration and link-prediction discoveries.

Storage decision (see docs/01): **Postgres is the source of truth** for
`kg_nodes` / `kg_edges` / `discoveries`. Neo4j, when enabled, is a *projected
read model* kept in sync by the projector — reads try Neo4j first and fall back
to Postgres so the feature never depends on Neo4j being up.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import NotFoundError
from app.models.knowledge import Discovery, KgEdge, KgNode
from app.schemas import DiscoveryOut, GraphEdgeOut, GraphExplorationOut, GraphNodeOut

MAX_NODES = 500
DEFAULT_HOPS = 2


class GraphService:
    def __init__(self, db: Session) -> None:
        self.db = db

    # ── exploration ─────────────────────────────────────────────────────

    def exploration(
        self, node_id: str | None, hops: int = DEFAULT_HOPS, relation: str | None = None
    ) -> GraphExplorationOut:
        hops = max(1, min(hops, 3))
        if settings.neo4j_enabled:
            try:
                return self._neo4j_exploration(node_id, hops, relation)
            except Exception:  # noqa: BLE001 - read model must not break the page
                pass
        return self._postgres_exploration(node_id, hops, relation)

    def _postgres_exploration(
        self, node_id: str | None, hops: int, relation: str | None
    ) -> GraphExplorationOut:
        edges = list(self.db.scalars(select(KgEdge)).all())
        degrees: dict[str, int] = {}
        for e in edges:
            degrees[e.src_id] = degrees.get(e.src_id, 0) + 1
            degrees[e.dst_id] = degrees.get(e.dst_id, 0) + 1

        if relation:
            edges = [e for e in edges if e.relation == relation]

        if node_id:
            # Breadth-first expansion so the response is the local neighbourhood.
            adjacency: dict[str, list[KgEdge]] = {}
            for e in edges:
                adjacency.setdefault(e.src_id, []).append(e)
                adjacency.setdefault(e.dst_id, []).append(e)
            seen_nodes = {node_id}
            seen_edges: dict[str, KgEdge] = {}
            frontier: deque[tuple[str, int]] = deque([(node_id, 0)])
            truncated = False
            while frontier:
                current, depth = frontier.popleft()
                if depth >= hops:
                    continue
                for edge in adjacency.get(current, []):
                    if len(seen_nodes) >= MAX_NODES:
                        truncated = True
                        break
                    seen_edges.setdefault(edge.id, edge)
                    for other in (edge.src_id, edge.dst_id):
                        if other not in seen_nodes:
                            seen_nodes.add(other)
                            frontier.append((other, depth + 1))
                if truncated:
                    break
            chosen = list(seen_edges.values())
        else:
            # No centre: return the highest-degree core of the graph.
            ranked = sorted(
                edges,
                key=lambda e: degrees.get(e.src_id, 0) + degrees.get(e.dst_id, 0),
                reverse=True,
            )
            chosen = ranked[:MAX_NODES]
            truncated = len(edges) > MAX_NODES
            seen_nodes = {n for e in chosen for n in (e.src_id, e.dst_id)}

        ids = {e.src_id for e in chosen} | {e.dst_id for e in chosen}
        rows = self.db.scalars(select(KgNode).where(KgNode.id.in_(ids))).all() if ids else []
        by_id = {n.id: n for n in rows}
        nodes = [
            GraphNodeOut(
                id=n.id,
                label=n.canonical_name,
                entityType=n.entity_type,
                domain=n.domain,
                degree=degrees.get(n.id, 0),
            )
            for n in rows
        ]
        edges_out = [
            GraphEdgeOut(
                id=e.id,
                source=e.src_id,
                target=e.dst_id,
                relation=e.relation,
                confidence=round(e.confidence, 3),
            )
            for e in chosen
            if e.src_id in by_id and e.dst_id in by_id
        ]
        return GraphExplorationOut(
            nodes=nodes,
            edges=edges_out,
            centerId=node_id,
            truncated=truncated if node_id else False,
        )

    def _neo4j_exploration(
        self, node_id: str | None, hops: int, relation: str | None
    ) -> GraphExplorationOut:
        driver = _get_neo4j_driver()
        rel_filter = f"WHERE r.relation = $relation" if relation else ""
        centre = node_id or ""
        cypher = f"""
            MATCH (a)-[r]->(b)
            WHERE ($start = '' OR a.id = $start OR b.id = $start) {rel_filter}
            RETURN a.id AS aid, a.label AS alabel, a.entityType AS aetype, a.domain AS adomain,
                   b.id AS bid, b.label AS blabel, b.entityType AS betype, b.domain AS bdomain,
                   r.relation AS relation, r.confidence AS confidence
            LIMIT $limit
        """
        nodes: dict[str, GraphNodeOut] = {}
        edges: list[GraphEdgeOut] = []
        with driver.session(database=settings.neo4j_database) as session:
            for rec in session.run(cypher, start=centre, relation=relation, limit=MAX_NODES):
                a = GraphNodeOut(
                    id=rec["aid"], label=rec["alabel"],
                    entityType=rec["aetype"] or "concept", domain=rec["adomain"] or "",
                )
                b = GraphNodeOut(
                    id=rec["bid"], label=rec["blabel"],
                    entityType=rec["betype"] or "concept", domain=rec["bdomain"] or "",
                )
                nodes.setdefault(a.id, a)
                nodes.setdefault(b.id, b)
                edges.append(
                    GraphEdgeOut(
                        id=f"{rec['aid']}->{rec['bid']}",
                        source=a.id, target=b.id,
                        relation=rec["relation"],
                        confidence=round(float(rec["confidence"] or 1.0), 3),
                    )
                )
        return GraphExplorationOut(
            nodes=list(nodes.values()),
            edges=edges,
            centerId=node_id,
            truncated=len(edges) >= MAX_NODES,
        )

    # ── discoveries ─────────────────────────────────────────────────────

    def discoveries(self, limit: int = 50) -> list[DiscoveryOut]:
        rows = list(
            self.db.scalars(
                select(Discovery)
                .where(Discovery.dismissed.is_(False))
                .order_by(Discovery.score.desc())
                .limit(limit)
            ).all()
        )
        if not rows:
            rows = self._cold_start_rank()
        return [self._out(d) for d in rows]

    def _cold_start_rank(self) -> list[Discovery]:
        """Seed plausible cross-domain pairs when the GNN batch has not run yet."""
        edges = list(self.db.scalars(select(KgEdge).where(KgEdge.domain_cross.is_(True))).all())
        if not edges:
            edges = list(self.db.scalars(select(KgEdge).limit(100)).all())
        if not edges:
            return []
        by_node = self._nodes({e.src_id for e in edges} | {e.dst_id for e in edges})
        created: list[Discovery] = []
        for e in edges[:25]:
            src = by_node.get(e.src_id)
            dst = by_node.get(e.dst_id)
            if src is None or dst is None:
                continue
            created.append(
                Discovery(
                    src_id=e.src_id,
                    dst_id=e.dst_id,
                    relation=e.relation,
                    score=round(min(0.99, e.confidence * settings.discovery_cross_domain_boost), 4),
                    scorer="coldstart",
                    evidence_paths=[[src.canonical_name, e.relation, dst.canonical_name]],
                    evidence_paper_ids=[x for x in (e.provenance_paper_id,) if x],
                )
            )
        self.db.add_all(created)
        self.db.commit()
        for d in created:
            self.db.refresh(d)
        return created

    def feedback(self, discovery_id: str, useful: bool) -> DiscoveryOut:
        row = self.db.get(Discovery, discovery_id)
        if row is None:
            raise NotFoundError(f"Discovery '{discovery_id}' not found")
        if useful:
            row.useful_votes = int(row.useful_votes or 0) + 1
        else:
            row.dismissed = True
        self.db.commit()

        # Recalibrate Platt scaling weights from stored feedback
        try:
            from ml.discovery.calibration import recalibrate_from_db

            recalibrate_from_db(self.db, checkpoint_dir=settings.ml_checkpoint_dir)
        except Exception:  # noqa: BLE001
            pass

        self.db.refresh(row)
        return self._out(row)

    def retrain_discoveries(self, train_gnn: bool = True, epochs: int = 25) -> dict[str, Any]:
        from ml.discovery.calibration import recalibrate_from_db
        from ml.pipelines.discovery import DiscoveryPipeline

        res = DiscoveryPipeline(self.db).run(train=train_gnn, epochs=epochs, persist=True)
        cal = recalibrate_from_db(self.db, checkpoint_dir=settings.ml_checkpoint_dir)
        return {
            "modelTrained": res.model_trained,
            "discoveriesCount": len(res.discoveries),
            "persisted": res.persisted_count,
            "hitsAt10": round(res.hits_at_10, 4),
            "mrr": round(res.mrr, 4),
            "calibration": cal,
        }

    def project_neo4j(self, full: bool = False) -> dict[str, Any]:
        from ml.graph.projector import Neo4jProjector

        res = Neo4jProjector(self.db).project(full=full)
        return {
            "status": res.status,
            "nodesProjected": res.nodes_projected,
            "edgesProjected": res.edges_projected,
            "watermark": res.watermark,
            "error": res.error,
        }

    def projection_state(self) -> dict[str, Any]:
        from ml.graph.projector import Neo4jProjector

        return Neo4jProjector(self.db).state()

    def _nodes(self, ids: set[str]) -> dict[str, KgNode]:
        if not ids:
            return {}
        return {n.id: n for n in self.db.scalars(select(KgNode).where(KgNode.id.in_(ids))).all()}

    def _out(self, d: Discovery) -> DiscoveryOut:
        nodes = self._nodes({d.src_id, d.dst_id})
        src = nodes.get(d.src_id)
        dst = nodes.get(d.dst_id)
        papers = list(d.evidence_paper_ids or [])
        return DiscoveryOut(
            id=d.id,
            score=round(d.score, 4),
            relation=d.relation,
            srcNode=self._node_out(src, d.src_id),
            dstNode=self._node_out(dst, d.dst_id),
            evidencePaths=[list(p) for p in (d.evidence_paths or [])],
            evidencePaperIds=papers,
            crossDomain=src is not None and dst is not None and src.domain != dst.domain,
        )

    @staticmethod
    def _node_out(node: KgNode | None, node_id: str) -> GraphNodeOut:
        if node is None:
            return GraphNodeOut(id=node_id, label=node_id, entityType="unknown", domain="")
        return GraphNodeOut(
            id=node.id,
            label=node.canonical_name,
            entityType=node.entity_type,
            domain=node.domain,
            degree=int(node.mention_count or 0),
        )


_driver: Any = None


def _get_neo4j_driver() -> Any:
    global _driver
    if _driver is not None:
        return _driver
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("neo4j driver not installed") from exc
    _driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    return _driver


__all__ = ["GraphService"]
