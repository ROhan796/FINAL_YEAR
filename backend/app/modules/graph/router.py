"""Knowledge-graph and discovery endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from app.core.deps import DbSession, ReadDbSession
from app.modules.graph.service import GraphService
from app.schemas import DiscoveryFeedback, DiscoveryOut, GraphExplorationOut

router = APIRouter(tags=["graph"])


@router.get("/graph/exploration", response_model=GraphExplorationOut)
def exploration(
    db: ReadDbSession,
    nodeId: Annotated[str | None, Query()] = None,  # noqa: N803 - contract name
    hops: Annotated[int, Query(ge=1, le=3)] = 2,
    relation: Annotated[str | None, Query()] = None,
) -> Any:
    return GraphService(db).exploration(nodeId, hops, relation)


@router.get("/discoveries", response_model=list[DiscoveryOut])
def discoveries(
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> Any:
    # DbSession: a cold start persists the seeded cross-domain candidates.
    return GraphService(db).discoveries(limit)


@router.post("/discoveries/{discovery_id}/feedback", response_model=DiscoveryOut)
def discovery_feedback(
    discovery_id: str, payload: DiscoveryFeedback, db: DbSession
) -> Any:
    return GraphService(db).feedback(discovery_id, payload.useful)


@router.post("/discoveries/retrain")
def retrain_discoveries(
    db: DbSession,
    trainGnn: Annotated[bool, Query()] = True,
    epochs: Annotated[int, Query(ge=1, le=100)] = 25,
) -> dict[str, Any]:
    return GraphService(db).retrain_discoveries(train_gnn=trainGnn, epochs=epochs)


@router.get("/graph/projection-status")
def projection_status(db: ReadDbSession) -> dict[str, Any]:
    return GraphService(db).projection_state()


@router.post("/graph/project")
def project_graph(
    db: DbSession,
    full: Annotated[bool, Query()] = False,
) -> dict[str, Any]:
    return GraphService(db).project_neo4j(full=full)


__all__ = ["router"]
