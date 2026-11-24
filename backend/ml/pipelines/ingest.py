"""Ingestion pipeline — fetch papers from sources and persist them.

Layer 1 of the CKGAI pipeline.

Adapters:
- arXiv  ✅ real (export.arxiv.org API, rate-limited, idempotent)
- PubMed ⬜ stub (E-utilities planned)
- Semantic Scholar ⬜ stub (Graph API planned)

Storage rules:
- idempotent upsert keyed on (source, external_id)
- `paper_chunks` are created for every stored paper (RAG fuel)
- `search_tsv` is refreshed on PostgreSQL (FTS arm)
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    papers_fetched: int = 0
    papers_stored: int = 0
    papers_skipped: int = 0
    chunks_created: int = 0
    errors: list[str] = field(default_factory=list)


def _split_chunks(text: str, *, chunk_chars: int = 1200, overlap: int = 150) -> list[tuple[int, int, str]]:
    """Character-window chunker with overlap → [(start, end, chunk), ...]."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [(0, len(text), text)]
    chunks: list[tuple[int, int, str]] = []
    step = max(1, chunk_chars - overlap)
    for start in range(0, len(text), step):
        end = min(len(text), start + chunk_chars)
        piece = text[start:end].strip()
        if piece:
            chunks.append((start, end, piece))
        if end >= len(text):
            break
    return chunks


class IngestPipeline:
    """Fetch papers from external sources and store them with chunks."""

    def __init__(self, db_session: Any = None):
        self.db = db_session

    # ── adapters ───────────────────────────────────────────────────────

    def fetch_arxiv(self, query: str = "", max_results: int = 100, category: str | None = None):
        from ml.ingest.arxiv import fetch_arxiv as _fetch

        from app.core.config import settings

        return _fetch(
            query,
            max_results=max_results,
            category=category,
            rate_limit_seconds=settings.ingest_rate_limit_seconds,
        )

    def ingest_arxiv(self, query: str, max_results: int = 100, category: str | None = None) -> IngestResult:
        """Fetch from arXiv and persist (when a db_session was provided)."""
        result = IngestResult()
        try:
            records = self.fetch_arxiv(query, max_results, category)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"arxiv fetch: {exc}")
            return result

        result.papers_fetched = len(records)
        if self.db is None:
            return result
        stored, skipped, chunks = self._store_many(records, source="arxiv")
        result.papers_stored = stored
        result.papers_skipped = skipped
        result.chunks_created = chunks
        return result

    def ingest_pubmed(self, query: str, max_results: int = 50) -> IngestResult:
        """Fetch from PubMed E-utilities and persist with chunks."""
        from app.core.config import settings
        from ml.ingest.pubmed import fetch_pubmed

        result = IngestResult()
        if not settings.pubmed_enabled:
            result.errors.append("pubmed ingestion disabled")
            return result
        try:
            records = fetch_pubmed(
                query,
                max_results=max_results,
                rate_limit_seconds=min(settings.ingest_rate_limit_seconds, 0.5),
            )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"pubmed fetch: {exc}")
            return result

        result.papers_fetched = len(records)
        if self.db is None or not records:
            return result
        stored, skipped, chunks = self._store_many(records, source="pubmed")
        result.papers_stored = stored
        result.papers_skipped = skipped
        result.chunks_created = chunks
        return result

    def ingest_semantic_scholar(self, query: str, max_results: int = 50) -> IngestResult:
        """Fetch from Semantic Scholar Graph API and persist with chunks."""
        from app.core.config import settings
        from ml.ingest.semantic_scholar import fetch_semantic_scholar

        result = IngestResult()
        if not settings.semantic_scholar_enabled:
            result.errors.append("semantic_scholar ingestion disabled")
            return result
        try:
            records = fetch_semantic_scholar(
                query,
                max_results=max_results,
                api_key=settings.semantic_scholar_api_key,
                rate_limit_seconds=min(settings.ingest_rate_limit_seconds, 2.0),
            )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"semantic_scholar fetch: {exc}")
            return result

        result.papers_fetched = len(records)
        if self.db is None or not records:
            return result
        stored, skipped, chunks = self._store_many(records, source="semantic_scholar")
        result.papers_stored = stored
        result.papers_skipped = skipped
        result.chunks_created = chunks
        return result

    # ── storage ────────────────────────────────────────────────────────

    def _store_many(self, records: list[Any], *, source: str) -> tuple[int, int, int]:
        from sqlalchemy import select

        from app.models.paper import Paper, PaperChunk, PaperTag

        stored = skipped = chunk_count = 0
        seen_external: set[str] = set()

        for rec in records:
            ext = getattr(rec, "external_id", "") or ""
            if not ext or ext in seen_external:
                skipped += 1
                continue
            seen_external.add(ext)

            existing = self.db.scalar(
                select(Paper).where(Paper.source == source, Paper.external_id == ext)
            )
            if existing is not None:
                skipped += 1
                continue

            paper = Paper(
                id=f"{source}-{re.sub(r'[^A-Za-z0-9._-]', '_', ext)[:48]}-{uuid.uuid4().hex[:6]}",
                source=source,
                external_id=ext,
                title=rec.title,
                abstract=rec.abstract,
                authors=list(rec.authors or []),
                year=int(rec.year or 2026),
                venue=getattr(rec, "venue", "") or source,
                doi=getattr(rec, "doi", "") or "",
                pdf_url=getattr(rec, "pdf_url", "") or None,
                open_access=True,
                citation_count=int(getattr(rec, "citation_count", 0) or 0),
                domain=getattr(rec, "domain", "") or "",
                published_at=getattr(rec, "published", None),
                tags=list(rec.tags or []),
            )
            self.db.add(paper)
            self.db.flush()  # get the id for chunk/tag rows

            for tag in (rec.tags or [])[:12]:
                self.db.add(PaperTag(paper_id=paper.id, tag=str(tag)[:160]))

            # RAG chunks: abstract (section=abstract) + any long body we have
            for idx, (s, e, chunk_text) in enumerate(_split_chunks(rec.abstract)):
                self.db.add(
                    PaperChunk(
                        paper_id=paper.id,
                        section="abstract",
                        chunk_index=idx,
                        text=chunk_text,
                        char_start=s,
                        char_end=e,
                    )
                )
                chunk_count += 1

            # FTS arm on PostgreSQL
            try:
                if self.db.get_bind().dialect.name == "postgresql":
                    from sqlalchemy import text

                    self.db.execute(
                        text(
                            "UPDATE papers SET search_tsv = "
                            "to_tsvector('english', coalesce(title,'') || ' ' || coalesce(abstract,'')) "
                            "WHERE id = :pid"
                        ),
                        {"pid": paper.id},
                    )
            except Exception as exc:  # noqa: BLE001 - sqlite has no tsvector
                log.debug("search_tsv_skip: %s", exc)

            stored += 1

        try:
            self.db.commit()
        except Exception as exc:  # noqa: BLE001
            self.db.rollback()
            raise RuntimeError(f"ingest commit failed: {exc}") from exc

        log.info("ingest_store: stored=%d skipped=%d chunks=%d", stored, skipped, chunk_count)
        return stored, skipped, chunk_count


__all__ = ["IngestPipeline", "IngestResult"]
