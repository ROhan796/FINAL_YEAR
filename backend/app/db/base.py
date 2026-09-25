"""SQLAlchemy declarative base and shared column mixins."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, MetaData, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Explicit constraint names keep migrations deterministic across environments.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def new_uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {dict[str, Any]: JSONB, list[str]: JSONB}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class UUIDPrimaryKey:
    # String(64) rather than a native UUID: seed ids look like `paper-101`,
    # `col-1712345678901`, and foreign keys elsewhere are String(64).
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_uuid)


# JSONB is postgres-only; SQLite (used by the unit-test suite) needs a portable fallback.
try:  # pragma: no cover - dialect selection
    from sqlalchemy import JSON

    JSONB_TYPE = JSONB().with_variant(JSON(), "sqlite")
    UUID_TYPE = UUID(as_uuid=False).with_variant(String(36), "sqlite")
except ImportError:  # pragma: no cover
    JSONB_TYPE = JSONB  # type: ignore[assignment]
    UUID_TYPE = String(36)  # type: ignore[assignment]
