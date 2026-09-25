"""Notifications and computed trending topics."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import JSONB_TYPE, Base, TimestampMixin, UUIDPrimaryKey
from app.db.types import VectorColumn


class Notification(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "notifications"

    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # mention | comment | collaboration | paper_update | verification
    type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    link: Mapped[str | None] = mapped_column(String(1024))
    read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    actor_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("users.id", ondelete="SET NULL"))
    entity_id: Mapped[str | None] = mapped_column(String(64), index=True)

    __table_args__ = (Index("ix_notifications_user_read_created", "user_id", "read", "created_at"),)


class TrendingTopic(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "trending_topics"

    name: Mapped[str] = mapped_column(String(300), nullable=False, unique=True)
    category: Mapped[str] = mapped_column(String(160), default="", nullable=False, index=True)
    monthly_growth: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    citation_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    paper_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    related_paper_ids: Mapped[list[str]] = mapped_column(JSONB_TYPE, default=list, nullable=False)
    top_researchers: Mapped[list[str]] = mapped_column(JSONB_TYPE, default=list, nullable=False)
    ai_narrative: Mapped[str] = mapped_column(Text, default="", nullable=False)
    clusters: Mapped[list[str]] = mapped_column(JSONB_TYPE, default=list, nullable=False)
    computed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


