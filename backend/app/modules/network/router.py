"""Researcher directory endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from app.core.deps import CurrentUserDep, DbSession, PageParams, Pagination, ReadDbSession, require
from app.modules.network.service import NetworkService
from app.schemas import ResearcherOut
from app.schemas.common import OkResponse, PageResponse

router = APIRouter(prefix="/researchers", tags=["network"])


@router.get("", response_model=PageResponse[ResearcherOut])
def list_researchers(
    db: ReadDbSession,
    user: CurrentUserDep,
    page: Annotated[PageParams, Depends(Pagination(default=24))],
    field: Annotated[str | None, Query(max_length=255)] = None,
    q: Annotated[str | None, Query(max_length=255)] = None,
) -> Any:
    return NetworkService(db).list_researchers(user["id"], field, q, page)


@router.get("/{researcher_id}", response_model=ResearcherOut)
def get_researcher(researcher_id: str, db: ReadDbSession, user: CurrentUserDep) -> Any:
    return NetworkService(db).get(researcher_id, user["id"])


@router.post("/{researcher_id}/follow", response_model=OkResponse)
def follow(
    researcher_id: str,
    db: DbSession,
    user: Annotated[Any, Depends(require("follow"))],
) -> Any:
    return NetworkService(db).toggle_follow(user["id"], researcher_id, True)


@router.delete("/{researcher_id}/follow", response_model=OkResponse)
def unfollow(researcher_id: str, db: DbSession, user: CurrentUserDep) -> Any:
    return NetworkService(db).toggle_follow(user["id"], researcher_id, False)


@router.get("/{researcher_id}/followers", response_model=list[ResearcherOut])
def followers(researcher_id: str, db: ReadDbSession, user: CurrentUserDep) -> Any:
    return NetworkService(db).followers(researcher_id, user["id"])


@router.get("/{researcher_id}/following", response_model=list[ResearcherOut])
def following(researcher_id: str, db: ReadDbSession, user: CurrentUserDep) -> Any:
    return NetworkService(db).following(researcher_id, user["id"])


__all__ = ["router"]
