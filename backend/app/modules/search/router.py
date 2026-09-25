"""Search endpoints: hybrid retrieval, history and personalised suggestions."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Query

from app.core.deps import CurrentUserDep, DbSession, ReadDbSession
from app.modules.search.service import SearchService
from app.schemas import PaperFilters, SearchRequest, SearchResponse

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(db: DbSession, payload: SearchRequest, user: CurrentUserDep) -> Any:
    """Server-owned index — unlike the prototype, the client never uploads a corpus.

    `externalPapers` is accepted solely for legacy `/api/search` compatibility.
    """
    external = payload.model_dump().get("externalPapers")
    service = SearchService(db)
    response = service.search(
        payload.query,
        mode=payload.mode,
        filters=payload.filters,
        limit=payload.limit,
        use_llm=True,
        external_papers=external,
    )
    if payload.query.strip():
        service.record_history(
            user["id"], payload.query, payload.mode, response.total, response.latencyMs
        )
    return response


@router.get("/history", response_model=list[str])
def history(db: ReadDbSession, user: CurrentUserDep) -> Any:
    return SearchService(db).history(user["id"])


@router.post("/history", response_model=list[str])
def push_history(db: DbSession, user: CurrentUserDep, query: Annotated[str, Body(embed=True)]) -> Any:
    if query.strip():
        SearchService(db).record_history(user["id"], query, "hybrid", 0, 0)
    return SearchService(db).history(user["id"])


@router.delete("/history", response_model=list[str])
def clear_history(db: DbSession, user: CurrentUserDep) -> Any:
    from sqlalchemy import delete

    from app.models.paper import SearchHistory

    db.execute(delete(SearchHistory).where(SearchHistory.user_id == user["id"]))
    db.commit()
    return []


@router.get("/suggestions", response_model=list[str])
def suggestions(
    db: ReadDbSession,
    user: CurrentUserDep,
    limit: Annotated[int, Query(ge=1, le=20)] = 6,
) -> Any:
    interests: list[str] = []
    try:
        from app.models.user import UserProfile
        from sqlalchemy import select

        row = db.scalar(select(UserProfile).where(UserProfile.user_id == user["id"]))
        if row:
            interests = list(row.interests or [])
    except Exception:  # noqa: BLE001 - suggestions must never fail
        pass
    return SearchService(db).personalised_suggestions(user["id"], interests, limit)


__all__ = ["router", "PaperFilters"]
