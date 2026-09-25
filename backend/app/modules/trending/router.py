"""Trending endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from app.core.deps import DbSession, PageParams, Pagination, ReadDbSession
from app.modules.trending.service import TrendingService
from app.schemas import TrendingTopicOut
from app.schemas.common import PageResponse

router = APIRouter(prefix="/trending", tags=["trending"])


@router.get("", response_model=PageResponse[TrendingTopicOut])
def trending(
    db: DbSession,
    page: Annotated[PageParams, Depends(Pagination(default=20))],
    limit: Annotated[int | None, Query(ge=1, le=100)] = None,
) -> Any:
    # DbSession because a cold start persists the computed topics.
    return TrendingService(db).list(page)


__all__ = ["router"]
