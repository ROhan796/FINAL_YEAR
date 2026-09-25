"""Knowledge graph: entities, relations, discoveries (Postgres = source of truth)."""

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


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge graph — POSTGRES IS THE SOURCE OF TRUTH.
# The Neo4j projector tails `kg_edges.updated_at` and projects these rows into a
# read-only Neo4j used for deep exploration + GNN neighbourhood sampling.
# ─────────────────────────────────────────────────────────────────────────────


class KgNode(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "kg_nodes"

    canonical_name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    normalised_name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    # METHOD | MATERIAL | DISEASE | ALGORITHM | DATASET | PHENOMENON | METRIC | CHEMICAL
    entity_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    domain: Mapped[str] = mapped_column(String(120), default="", nullable=False, index=True)
    # SciBERT 768-d — powers entity resolution (cosine > 0.92) and the GNN seed features.
    embedding = mapped_column(VectorColumn(), nullable=True)
    paper_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("papers.id", ondelete="SET NULL"), index=True
    )
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    mention_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    aliases: Mapped[list[str]] = mapped_column(JSONB_TYPE, default=list, nullable=False)

    __table_args__ = (
        UniqueConstraint("normalised_name", "entity_type", name="uq_kg_node_name_type"),
        Index("ix_kg_nodes_type_domain", "entity_type", "domain"),
    )


class KgEdge(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "kg_edges"

    src_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("kg_nodes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dst_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("kg_nodes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # used_in | derived_from | applied_to | improves | treats | causes | measures | contains
    relation: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    provenance_paper_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("papers.id", ondelete="SET NULL"), index=True
    )
    provenance_sentence: Mapped[str | None] = mapped_column(Text)

    src_domain: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    dst_domain: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    domain_cross: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint("src_id", "dst_id", "relation", name="uq_kg_edge_triple"),
        Index("ix_kg_edges_src_relation", "src_id", "relation"),
        Index("ix_kg_edges_dst_relation", "dst_id", "relation"),
        # Projector watermark — this index is what makes projection incremental.
        Index("ix_kg_edges_updated", "updated_at"),
        Index("ix_kg_edges_cross", "domain_cross", "relation"),
    )


class Discovery(UUIDPrimaryKey, TimestampMixin, Base):
    """A ranked link-prediction output, persisted so results are reviewable."""

    __tablename__ = "discoveries"

    src_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("kg_nodes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dst_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("kg_nodes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    relation: Mapped[str] = mapped_column(String(60), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    scorer: Mapped[str] = mapped_column(String(32), default="distmult", nullable=False)
    evidence_paths: Mapped[list[dict]] = mapped_column(JSONB_TYPE, default=list, nullable=False)
    evidence_paper_ids: Mapped[list[str]] = mapped_column(JSONB_TYPE, default=list, nullable=False)
    batch_id: Mapped[str | None] = mapped_column(String(64), index=True)
    useful_votes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    dismissed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    __table_args__ = (
        UniqueConstraint("src_id", "dst_id", "relation", "batch_id", name="uq_discovery_triple"),
        Index("ix_discoveries_rank", "score", "dismissed"),
    )


class GraphProjectionState(Base):
    """Single-row watermark table the projector advances after each successful sync."""

    __tablename__ = "graph_projection_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    last_projected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_edge_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    nodes_projected: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    edges_projected: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
