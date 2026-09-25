"""Manuscript submission pipeline."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import JSONB_TYPE, Base, TimestampMixin, UUIDPrimaryKey


class Manuscript(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "manuscripts"

    submitter_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str] = mapped_column(Text, default="", nullable=False)
    authors: Mapped[list[str]] = mapped_column(JSONB_TYPE, default=list, nullable=False)
    co_authors: Mapped[list[str]] = mapped_column(JSONB_TYPE, default=list, nullable=False)
    venue: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    tags: Mapped[list[str]] = mapped_column(JSONB_TYPE, default=list, nullable=False)
    year: Mapped[int] = mapped_column(Integer, default=2026, nullable=False)

    file_name: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    file_size: Mapped[str] = mapped_column(String(40), default="", nullable=False)
    file_key: Mapped[str | None] = mapped_column(String(1024))  # object storage key
    integrity_confirmed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Draft | Under AI Review | Submitted | Under Review | Published | Revision Required
    status: Mapped[str] = mapped_column(String(40), default="Draft", nullable=False, index=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reviewer_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="SET NULL"), index=True
    )

    ai_review: Mapped[dict | None] = mapped_column(JSONB_TYPE)
    ai_reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (UniqueConstraint("submitter_id", "title", name="uq_submitter_title"),)

    @property
    def ai_review_model(self) -> AIReview | None:  # pragma: no cover - convenience
        return AIReview(**self.ai_review) if self.ai_review else None


class AIReview:  # plain value object, not a table
    """Pydantic-free mirror of `ManuscriptSubmission.aiReview` in the frontend."""

    __slots__ = (
        "predictedScore",
        "titleReview",
        "executiveCritique",
        "suggestedKeywords",
        "likelyPeerQuestions",
        "clarityAssessment",
        "strengths",
        "weaknesses",
    )

    def __init__(self, **kw: object) -> None:
        for key in self.__slots__:
            setattr(self, key, kw.get(key))
