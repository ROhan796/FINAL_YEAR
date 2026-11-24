"""Dialect-aware column types.

`VectorColumn` is pgvector's `vector(n)` on PostgreSQL and JSON on SQLite so the
unit-test suite can run the whole schema without a Postgres server.
"""

from __future__ import annotations

from sqlalchemy import JSON
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeEngine

from app.core.config import settings


class _PortableVector(TypeEngine):
    """Fallback used when the `pgvector` package is not installed."""

    cache_ok = True

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.embedding_dim

    def get_col_spec(self, **kw: object) -> str:  # pragma: no cover
        return f"VECTOR({self.dim})"


def VectorColumn(dim: int | None = None):  # noqa: N802 - matches SQLAlchemy style
    """A 768-d float vector. Uses pgvector when available, JSON on SQLite.

    `none_as_null=True` matters: the default JSON type serialises Python
    `None` to the TEXT `'null'`, which defeats every `IS NULL` predicate
    (backfill, k-NN rebuild). With it, unset vectors are real SQL NULLs —
    matching how pgvector's own type binds `None`.
    """
    width = dim or settings.embedding_dim
    try:
        from pgvector.sqlalchemy import Vector

        return Vector(width).with_variant(JSON(none_as_null=True), "sqlite")
    except ImportError:  # pragma: no cover
        return _PortableVector(width).with_variant(JSON(none_as_null=True), "sqlite")


def TSVectorColumn():  # noqa: N802 - matches SQLAlchemy style
    """PostgreSQL `tsvector` for the full-text arm; JSON on SQLite."""
    from sqlalchemy.dialects.postgresql import TSVECTOR

    return TSVECTOR().with_variant(JSON(none_as_null=True), "sqlite")


def load_vector_extension(conn: Dialect) -> None:  # pragma: no cover - postgres only
    """`CREATE EXTENSION IF NOT EXISTS vector` — run once per database."""
    if conn.dialect.name == "postgresql":
        conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS pg_trgm")
