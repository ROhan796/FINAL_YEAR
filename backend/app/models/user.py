"""Users, profiles and the verification workflow."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import JSONB_TYPE, Base, TimestampMixin, UUIDPrimaryKey

if TYPE_CHECKING:
    pass


class User(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[str | None] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[str] = mapped_column(String(40), default="researcher", nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    profile: Mapped[UserProfile | None] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User {self.email} role={self.role}>"


class UserProfile(UUIDPrimaryKey, TimestampMixin, Base):
    """Public-facing profile. Maps 1:1 onto the frontend's `UserProfile`."""

    __tablename__ = "user_profiles"

    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    title: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    institution: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    field: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    bio: Mapped[str] = mapped_column(Text, default="", nullable=False)
    avatar: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    interests: Mapped[list[str]] = mapped_column(JSONB_TYPE, default=list, nullable=False)

    user: Mapped[User] = relationship(back_populates="profile")

    # Denormalised bibliometrics, refreshed by a background job.
    publications_count: Mapped[int] = mapped_column(default=0, nullable=False)
    citation_count: Mapped[int] = mapped_column(default=0, nullable=False)
    h_index: Mapped[int] = mapped_column(default=0, nullable=False)


class VerificationRequest(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "verification_requests"
    __table_args__ = (UniqueConstraint("user_id", "status", name="uq_verification_open"),)

    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    institutional_email: Mapped[str] = mapped_column(String(320), nullable=False)
    orcid: Mapped[str] = mapped_column(String(64), nullable=False)
    # pending | approved | rejected
    status: Mapped[str] = mapped_column(String(24), default="pending", nullable=False, index=True)
    decided_by: Mapped[str | None] = mapped_column(String(64))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    note: Mapped[str | None] = mapped_column(Text)
