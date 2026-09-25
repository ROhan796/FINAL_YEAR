"""Feed endpoints: timeline, posts, likes, comments."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from app.core.deps import CurrentUserDep, DbSession, PageParams, Pagination, ReadDbSession, require
from app.core.errors import NotFoundError
from app.modules.feed.service import FeedService
from app.schemas import CommentCreate, FeedCommentOut, FeedPostOut, PostCreate, PostLikeOut
from app.schemas.common import OkResponse, PageResponse

router = APIRouter(tags=["feed"])


def _author(db: Any, user: dict[str, str]) -> dict[str, Any]:
    """Denormalise the author onto the row (single query, no N+1 on read)."""
    from sqlalchemy import select

    from app.models.user import UserProfile

    profile = db.scalar(select(UserProfile).where(UserProfile.user_id == user["id"]))
    return {
        "name": user.get("name") or (profile and profile.user.full_name if profile else "") or "Researcher",
        "title": (profile.title if profile else ""),
        "institution": (profile.institution if profile else ""),
        "avatar": (profile.avatar if profile else ""),
        "verified": (profile.user.verified if profile and profile.user else False),
    }


@router.get("/feed", response_model=PageResponse[FeedPostOut])
def feed(
    db: DbSession,
    user: CurrentUserDep,
    page: Annotated[PageParams, Depends(Pagination(default=10))],
    sort: Annotated[str, Query(pattern="^(recency|recommended)$")] = "recency",
) -> Any:
    service = FeedService(db)
    interests = service.repo.interests(user["id"])
    return service.feed(user["id"], sort, page, interests)


@router.get("/posts/{post_id}", response_model=FeedPostOut)
def get_post(post_id: str, db: ReadDbSession, user: CurrentUserDep) -> Any:
    return FeedService(db).get_post(post_id, user["id"])


@router.post("/posts", response_model=FeedPostOut, status_code=201)
def create_post(
    db: DbSession,
    user: Annotated[Any, Depends(require("post_to_feed"))],
    payload: PostCreate,
) -> Any:
    return FeedService(db).create_post(
        user_id=user["id"],
        author=_author(db, user),
        content=payload.content,
        post_type=payload.postType,
        linked_paper_id=payload.linkedPaperId,
    )


@router.post("/posts/{post_id}/like", response_model=PostLikeOut)
def like_post(post_id: str, db: DbSession, user: CurrentUserDep) -> Any:
    return FeedService(db).toggle_like(user["id"], post_id, True)


@router.delete("/posts/{post_id}/like", response_model=PostLikeOut)
def unlike_post(post_id: str, db: DbSession, user: CurrentUserDep) -> Any:
    return FeedService(db).toggle_like(user["id"], post_id, False)


@router.get("/posts/{post_id}/comments", response_model=list[FeedCommentOut])
def list_comments(post_id: str, db: ReadDbSession, user: CurrentUserDep) -> Any:
    return FeedService(db).list_comments(post_id)


@router.post("/posts/{post_id}/comments", response_model=FeedCommentOut, status_code=201)
def add_comment(
    post_id: str,
    db: DbSession,
    user: Annotated[Any, Depends(require("comment"))],
    payload: CommentCreate,
) -> Any:
    return FeedService(db).add_comment(user["id"], _author(db, user), post_id, payload.content)


@router.delete("/comments/{comment_id}", response_model=OkResponse)
def delete_comment(comment_id: str, db: DbSession, user: CurrentUserDep) -> Any:
    FeedService(db).delete_comment(user["id"], comment_id)
    return OkResponse()


__all__ = ["router"]
