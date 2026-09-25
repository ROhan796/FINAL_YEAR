"""Social layer: feed posts, comments, likes, follow graph, researcher profiles."""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import JSONB_TYPE, Base, TimestampMixin, UUIDPrimaryKey


class FeedPost(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "feed_posts"

    author_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Research Update | Preprint Announcement | Methodology Discussion | Dataset Release
    post_type: Mapped[str] = mapped_column(String(60), default="Research Update", nullable=False)
    linked_paper_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("papers.id", ondelete="SET NULL"), index=True
    )
    tags: Mapped[list[str]] = mapped_column(JSONB_TYPE, default=list, nullable=False)

    # Cached for feed hydration so listing posts never triggers N+1 author/paper lookups.
    author_name: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    author_title: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    author_institution: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    author_avatar: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    author_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    like_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    comment_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    linked_paper = relationship("Paper", lazy="joined")

    __table_args__ = (
        # Recency feed page: `WHERE author_id IN (...) ORDER BY created_at DESC`.
        Index("ix_feed_posts_author_created", "author_id", "created_at"),
        Index("ix_feed_posts_created", text("created_at DESC")),
    )


class FeedComment(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "feed_comments"

    post_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("feed_posts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    author_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)

    author_name: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    author_institution: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    author_avatar: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    author_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    __table_args__ = (Index("ix_feed_comments_post_created", "post_id", "created_at"),)


class PostLike(UUIDPrimaryKey, Base):
    """Row per (user, post). The response's `isLiked` is a per-viewer join."""

    __tablename__ = "post_likes"
    __table_args__ = (UniqueConstraint("post_id", "user_id", name="uq_post_like"),)

    post_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("feed_posts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )


class Follow(UUIDPrimaryKey, Base):
    __tablename__ = "follows"
    __table_args__ = (
        UniqueConstraint("follower_id", "followee_id", name="uq_follow_pair"),
        CheckConstraint("follower_id <> followee_id", name="ck_no_self_follow"),
    )

    follower_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    followee_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )


class ResearcherProfile(UUIDPrimaryKey, TimestampMixin, Base):
    """Directory entry for other people you can follow.

    `user_id` is nullable so harvested arXiv/PubMed authors appear in the network
    without owning an account — they are followable but cannot log in.
    """

    __tablename__ = "researcher_profiles"

    user_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), unique=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    institution: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    field: Mapped[str] = mapped_column(String(255), default="", nullable=False, index=True)
    bio: Mapped[str] = mapped_column(Text, default="", nullable=False)
    avatar: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    orcid: Mapped[str | None] = mapped_column(String(64))

    interests: Mapped[list[str]] = mapped_column(JSONB_TYPE, default=list, nullable=False)

    publications_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    citation_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    h_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    follower_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
