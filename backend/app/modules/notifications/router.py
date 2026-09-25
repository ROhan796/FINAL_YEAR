"""HTTP layer for notifications. No SQL here — only validation and delegation."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from app.core.deps import CurrentUserDep, DbSession, PageParams, Pagination, ReadDbSession
from app.modules.notifications.service import NotificationService
from app.schemas import NotificationOut, OkResponse, UnreadCountOut
from app.schemas.common import PageResponse

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("", response_model=PageResponse[NotificationOut])
def list_notifications(
    db: ReadDbSession,
    user: CurrentUserDep,
    page: Annotated[PageParams, Depends(Pagination(default=15))],
    unread_only: Annotated[bool, Query(alias="unreadOnly")] = False,
) -> Any:
    return NotificationService(db).list_notifications(user["id"], unread_only, page)


@router.get("/unread-count", response_model=UnreadCountOut)
def unread_count(db: ReadDbSession, user: CurrentUserDep) -> Any:
    """Navbar badge — one indexed COUNT, never a full inbox fetch."""
    return UnreadCountOut(count=NotificationService(db).unread_count(user["id"]))


@router.post("/read-all", response_model=OkResponse)
def read_all_notifications(db: DbSession, user: CurrentUserDep) -> Any:
    NotificationService(db).read_all(user["id"])
    return OkResponse()


@router.patch("/{notification_id}", response_model=OkResponse)
def mark_notification_read(notification_id: str, db: DbSession, user: CurrentUserDep) -> Any:
    NotificationService(db).mark_read(user["id"], notification_id)
    return OkResponse()


__all__ = ["router", "NotificationService"]
