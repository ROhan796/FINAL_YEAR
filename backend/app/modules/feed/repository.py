"""Feed data access: author-filtered timelines and per-viewer engagement state."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.orm import Session

from app.models.paper import Paper
from app.models.social import FeedComment, FeedPost, Follow, PostLike
from app.models.user import User


class FeedRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def followed_ids(self, user_id: str) -> list[str]:
        return list(
            self.db.scalars(select(Follow.followee_id).where(Follow.follower_id == user_id)).all()
        )

    def timeline_ids(self, user_id: str, limit: int) -> list[str]:
        """Followed authors + own posts. Cold start (nobody followed) returns the
        global timeline so the feed is never empty on first visit."""
        authors = self.followed_ids(user_id)
        conditions = [FeedPost.author_id.in_(authors + [user_id])] if authors else []
        stmt: Select[Any] = select(FeedPost.id).order_by(FeedPost.created_at.desc(), FeedPost.id.desc())
        for cond in conditions:
            stmt = stmt.where(cond)
        return list(self.db.scalars(stmt.limit(limit)).all())

    def all_ids(self, limit: int) -> list[str]:
        stmt = select(FeedPost.id).order_by(FeedPost.created_at.desc(), FeedPost.id.desc()).limit(limit)
        return list(self.db.scalars(stmt).all())

    def posts_by_ids(self, ids: list[str]) -> list[FeedPost]:
        if not ids:
            return []
        rows = self.db.scalars(select(FeedPost).where(FeedPost.id.in_(list(ids)))).all()
        by_id = {p.id: p for p in rows}
        return [by_id[i] for i in ids if i in by_id]

    def window(self, limit: int) -> list[FeedPost]:
        stmt = select(FeedPost).order_by(FeedPost.created_at.desc()).limit(limit)
        return list(self.db.scalars(stmt).all())

    def get(self, post_id: str) -> FeedPost | None:
        return self.db.get(FeedPost, post_id)

    def liked_ids(self, user_id: str, post_ids: list[str]) -> set[str]:
        if not post_ids:
            return set()
        rows = self.db.scalars(
            select(PostLike.post_id).where(
                PostLike.user_id == user_id, PostLike.post_id.in_(post_ids)
            )
        ).all()
        return set(rows)

    def comments_for(self, post_ids: list[str]) -> dict[str, list[FeedComment]]:
        if not post_ids:
            return {}
        rows = self.db.scalars(
            select(FeedComment)
            .where(FeedComment.post_id.in_(post_ids))
            .order_by(FeedComment.created_at.asc())
        ).all()
        grouped: dict[str, list[FeedComment]] = {}
        for c in rows:
            grouped.setdefault(c.post_id, []).append(c)
        return grouped

    def papers_for(self, post_ids: list[str]) -> dict[str, Paper]:
        rows = self.db.execute(
            select(FeedPost.id, Paper).join(Paper, Paper.id == FeedPost.linked_paper_id).where(
                FeedPost.id.in_(post_ids)
            )
        ).all()
        return {pid: paper for pid, paper in rows}

    # ── writes ──────────────────────────────────────────────────────────

    def is_liked(self, user_id: str, post_id: str) -> bool:
        return (
            self.db.scalar(
                select(PostLike.id).where(PostLike.user_id == user_id, PostLike.post_id == post_id)
            )
            is not None
        )

    def add_like(self, user_id: str, post_id: str) -> bool:
        if self.is_liked(user_id, post_id):
            return False
        self.db.add(PostLike(post_id=post_id, user_id=user_id))
        # Atomic counter — read-modify-write would lose updates under concurrency.
        self.db.execute(
            FeedPost.__table__.update()
            .where(FeedPost.id == post_id)
            .values(like_count=FeedPost.like_count + 1)
        )
        return True

    def remove_like(self, user_id: str, post_id: str) -> bool:
        row = self.db.scalar(
            select(PostLike).where(PostLike.user_id == user_id, PostLike.post_id == post_id)
        )
        if row is None:
            return False
        self.db.delete(row)
        self.db.execute(
            FeedPost.__table__.update()
            .where(FeedPost.id == post_id)
            .values(like_count=func.greatest(FeedPost.like_count - 1, 0))
        )
        return True

    def add_comment(self, comment: FeedComment) -> FeedComment:
        self.db.add(comment)
        self.db.execute(
            FeedPost.__table__.update()
            .where(FeedPost.id == comment.post_id)
            .values(comment_count=FeedPost.comment_count + 1)
        )
        return comment

    def delete_comment(self, comment_id: str) -> FeedComment | None:
        row = self.db.get(FeedComment, comment_id)
        if row is None:
            return None
        self.db.execute(
            FeedPost.__table__.update()
            .where(FeedPost.id == row.post_id)
            .values(comment_count=func.greatest(FeedPost.comment_count - 1, 0))
        )
        self.db.delete(row)
        return row

    def interests(self, user_id: str) -> list[str]:
        from app.models.user import UserProfile

        row = self.db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
        return list(row.interests) if row and row.interests else []


def followed_condition(user_id: str) -> Any:  # pragma: no cover - helper reuse
    return or_(
        FeedPost.author_id == user_id,
        FeedPost.author_id.in_(select(Follow.followee_id).where(Follow.follower_id == user_id)),
    )


__all__ = ["FeedRepository", "and_", "User"]
