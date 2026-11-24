"""Insights endpoints — Research Gaps, Gateway Papers, Research Debt, Cross-Collection, Affinity, Velocity."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query

from app.core.deps import CurrentUserDep, ReadDbSession
from app.modules.insights.service import InsightsService

router = APIRouter(prefix="/insights", tags=["insights"])


@router.get("/research-gaps")
def research_gaps(
    db: ReadDbSession,
    limit: Annotated[int, Query(ge=1, le=50)] = 15,
) -> list[dict[str, Any]]:
    """6.B: Cross-domain method x target co-occurrence gaps in the KG."""
    return InsightsService(db).research_gaps(limit=limit)


@router.get("/gateway-papers")
def gateway_papers(
    db: ReadDbSession,
    limit: Annotated[int, Query(ge=1, le=30)] = 8,
) -> list[dict[str, Any]]:
    """6.D: PageRank over reading-order save transitions & paper graph."""
    return InsightsService(db).gateway_papers(limit=limit)


@router.get("/research-debt")
def research_debt(
    db: ReadDbSession,
    user: CurrentUserDep,
    agingDays: Annotated[int, Query(ge=0, le=365)] = 30,
) -> dict[str, Any]:
    """7.2: Saved papers lacking workspace notes, project links, or AI summaries."""
    return InsightsService(db).research_debt(user["id"], aging_days=agingDays)


@router.get("/cross-collection")
def cross_collection(
    db: ReadDbSession,
    user: CurrentUserDep,
    limit: Annotated[int, Query(ge=1, le=20)] = 6,
) -> list[dict[str, Any]]:
    """7.3: Papers bridging two collections via centroid embedding similarity."""
    return InsightsService(db).cross_collection_insights(user["id"], limit=limit)


@router.get("/affinity")
def collaboration_affinity(
    db: ReadDbSession,
    user: CurrentUserDep,
) -> dict[str, dict[str, Any]]:
    """7.5: Pairwise research affinity scores between viewer and researchers."""
    return InsightsService(db).researcher_affinity(user["id"])


@router.get("/velocity")
def research_velocity(
    db: ReadDbSession,
    user: CurrentUserDep,
) -> dict[str, Any]:
    """7.7: Self-reflective monthly research productivity metrics."""
    return InsightsService(db).research_velocity(user["id"])


__all__ = ["router"]
