"""Paper catalog, embeddings, saved library and search history.

`papers.embedding` (SPECTER2, 768-d) and `papers.search_tsv` are what make the
hybrid retrieval arm possible inside the same transaction as the row itself.
"""

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
from app.db.types import TSVectorColumn, VectorColumn


class Paper(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "papers"

    # External ids keep ingestion idempotent across arXiv / PubMed / S2 / DOI.
    source: Mapped[str] = mapped_column(String(32), default="seed", nullable=False, index=True)
    external_id: Mapped[str | None] = mapped_column(String(255), index=True)

    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str] = mapped_column(Text, default="", nullable=False)
    authors: Mapped[list[str]] = mapped_column(JSONB_TYPE, default=list, nullable=False)
    year: Mapped[int] = mapped_column(Integer, default=2026, nullable=False, index=True)
    venue: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    doi: Mapped[str] = mapped_column(String(255), default="", nullable=False, index=True)
    pdf_url: Mapped[str | None] = mapped_column(String(1024))
    open_access: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    citation_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False, index=True)
    domain: Mapped[str] = mapped_column(String(120), default="", nullable=False, index=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    tags: Mapped[list[str]] = mapped_column(JSONB_TYPE, default=list, nullable=False)

    # SPECTER2 768-d paper vector. Same transaction as the row — no dual-write.
    embedding = mapped_column(VectorColumn(), nullable=True)

    # ── AI enrichment (cached; never recomputed inline) ──────────────────
    ai_summary: Mapped[str | None] = mapped_column(Text)
    key_contributions: Mapped[list[str] | None] = mapped_column(JSONB_TYPE)
    methodology: Mapped[str | None] = mapped_column(Text)
    limitations: Mapped[list[str] | None] = mapped_column(JSONB_TYPE)
    research_gaps: Mapped[list[str] | None] = mapped_column(JSONB_TYPE)
    citation_context: Mapped[str | None] = mapped_column(Text)
    ai_analyzed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Full-text search arm of hybrid retrieval. The GIN index lives in migration 0001
    # because `to_tsvector(...)` expressions are PostgreSQL-only.
    search_tsv = mapped_column(TSVectorColumn(), nullable=True)

    __table_args__ = (
        UniqueConstraint("source", "external_id", name="uq_papers_source_external"),
        Index("ix_papers_year_citations", "year", "citation_count"),
        Index("ix_papers_openaccess_citations", "open_access", "citation_count"),
    )

    @property
    def openAccess(self) -> bool:  # noqa: N802 - frontend field name
        return self.open_access

    @property
    def citationCount(self) -> int:  # noqa: N802
        return self.citation_count


class PaperTag(UUIDPrimaryKey, Base):
    """Normalised tag table — powers the Discover discipline chips + tag filters."""

    __tablename__ = "paper_tags"
    __table_args__ = (UniqueConstraint("paper_id", "tag", name="uq_paper_tag"),)

    paper_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tag: Mapped[str] = mapped_column(String(160), nullable=False, index=True)


class PaperChunk(UUIDPrimaryKey, TimestampMixin, Base):
    """Section-aware passage used by RAG retrieval for the workspace assistant."""

    __tablename__ = "paper_chunks"

    paper_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    section: Mapped[str] = mapped_column(String(80), default="abstract", nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Passage vector for RAG retrieval over a paper's sections.
    embedding = mapped_column(VectorColumn(), nullable=True)


class SavedPaper(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "saved_papers"
    __table_args__ = (UniqueConstraint("user_id", "paper_id", name="uq_user_saved_paper"),)

    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    paper_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True
    )


class SearchHistory(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "search_history"

    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    query: Mapped[str] = mapped_column(String(500), nullable=False)
    mode: Mapped[str] = mapped_column(String(24), default="hybrid", nullable=False)
    result_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class RelatedPaper(UUIDPrimaryKey, Base):
    """Precomputed SPECTER2 k-NN neighbours — makes /papers/:id/related a single query."""

    __tablename__ = "related_papers"
    __table_args__ = (UniqueConstraint("paper_id", "related_to_id", name="uq_related_pair"),)

    paper_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    related_to_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
