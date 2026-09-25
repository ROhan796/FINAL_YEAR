"""Collection business rules: ownership, membership, AI synthesis, export."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from app.core.errors import ConflictError, ForbiddenError, NotFoundError, ValidationAppError
from app.core.pagination import Page, PageParams
from app.models.collection import Collection
from app.models.paper import Paper
from app.modules.collections.repository import CollectionRepository
from app.schemas import AiOverviewOut, CollectionCreate, CollectionOut, CollectionPatch


class CollectionService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo = CollectionRepository(db)

    # ── reads ───────────────────────────────────────────────────────────

    def list(self, owner_id: str, page: PageParams) -> Page[CollectionOut]:
        rows = self.repo.list_for(owner_id)
        window = rows[: page.limit]
        return Page[CollectionOut](
            items=[self._out(c) for c in window],
            nextCursor=None,
            total=len(rows),
        )

    def get(self, collection_id: str, owner_id: str) -> Collection:
        col = self.repo.get(collection_id, owner_id)
        if col is None:
            raise NotFoundError(f"Collection '{collection_id}' not found")
        return col

    def _out(self, col: Collection) -> CollectionOut:
        return CollectionOut(
            id=col.id,
            name=col.name,
            description=col.description,
            pinned=col.pinned,
            paperIds=self.repo.paper_ids(col.id),
            createdDate=(col.created_at.date().isoformat() if col.created_at else ""),
            aiSummary=col.ai_summary,
            shareableLink=(
                f"{_base()}/api/v1/collections/{col.id}/share/{col.share_token}"
                if col.share_token
                else None
            ),
        )

    # ── writes ──────────────────────────────────────────────────────────

    def create(self, owner_id: str, payload: CollectionCreate) -> CollectionOut:
        if self.db.query(Collection).filter(
            Collection.owner_id == owner_id, Collection.name == payload.name
        ).first():
            raise ConflictError(f"A collection named '{payload.name}' already exists")

        col = Collection(
            owner_id=owner_id,
            name=payload.name,
            description=payload.description,
            share_token=uuid.uuid4().hex,
            ai_summary="Collection created. Add papers to generate AI collection analysis.",
        )
        self.repo.add(col)
        self.db.flush()

        # `paperId` fixes the prototype's "Create & Add" bug: create and link atomically.
        if payload.paperId:
            self._ensure_paper(payload.paperId)
            self.repo.add_paper(col.id, payload.paperId)
        self.db.commit()
        self.db.refresh(col)
        return self._out(col)

    def patch(self, collection_id: str, owner_id: str, payload: CollectionPatch) -> CollectionOut:
        col = self.get(collection_id, owner_id)
        if payload.name is not None:
            col.name = payload.name
        if payload.description is not None:
            col.description = payload.description
        if payload.pinned is not None:
            col.pinned = payload.pinned
        self.db.commit()
        self.db.refresh(col)
        return self._out(col)

    def delete(self, collection_id: str, owner_id: str) -> None:
        col = self.get(collection_id, owner_id)
        self.repo.delete(col)
        self.db.commit()

    def add_paper(self, collection_id: str, owner_id: str, paper_id: str) -> CollectionOut:
        col = self.get(collection_id, owner_id)
        self._ensure_paper(paper_id)
        if self.repo.has_paper(col.id, paper_id):
            raise ConflictError("Paper is already in this collection")
        self.repo.add_paper(col.id, paper_id)
        self.db.commit()
        return self._out(col)

    def remove_paper(self, collection_id: str, owner_id: str, paper_id: str) -> None:
        col = self.get(collection_id, owner_id)
        if not self.repo.remove_paper(col.id, paper_id):
            raise NotFoundError("Paper is not in this collection")
        self.db.commit()

    # ── AI synthesis ────────────────────────────────────────────────────

    def ai_overview(self, collection_id: str, owner_id: str) -> AiOverviewOut:
        col = self.get(collection_id, owner_id)
        paper_ids = self.repo.paper_ids(col.id)

        fresh = (
            col.ai_summary_at is not None
            and col.ai_summary
            and (datetime.now(UTC) - _aware(col.ai_summary_at)) < timedelta(days=7)
        )
        if fresh and col.ai_summary and not col.ai_summary.startswith("Collection created."):
            return AiOverviewOut(summary=col.ai_summary, executiveSummary=col.ai_summary, paperCount=len(paper_ids))

        document = self._compose(col, paper_ids)
        summary = _fallback_synthesis(document, col.name)

        try:
            from app.ai.service import summarize_text

            summary = summarize_text(document, title=col.name, alias=f"collection:{col.id}")
        except Exception:  # noqa: BLE001 - degrade, never fail the request
            pass

        col.ai_summary = summary
        col.ai_summary_at = datetime.now(UTC)
        self.db.commit()
        # Both keys must carry the same text: CollectionsView reads `executiveSummary`.
        return AiOverviewOut(summary=summary, executiveSummary=summary, paperCount=len(paper_ids))

    def _compose(self, col: Collection, paper_ids: list[str]) -> str:
        parts = [f"Collection: {col.name}.", col.description or ""]
        if paper_ids:
            rows = self.db.query(Paper).filter(Paper.id.in_(paper_ids[:40])).all()
            for p in rows:
                parts.append(f"- {p.title} ({p.year}, {p.venue}): {p.abstract[:600]}")
        return "\n".join(p for p in parts if p)

    # ── export ──────────────────────────────────────────────────────────

    def export(self, collection_id: str, owner_id: str, fmt: str) -> str:
        col = self.get(collection_id, owner_id)
        paper_ids = self.repo.paper_ids(col.id)
        papers = self.db.query(Paper).filter(Paper.id.in_(paper_ids)).all() if paper_ids else []

        if fmt == "md":
            lines = [f"# {col.name}", "", col.description, ""]
            for p in papers:
                lines.append(f"- **{p.title}** — {', '.join(p.authors)} ({p.year}). {p.venue}. https://doi.org/{p.doi}")
            return "\n".join(lines)

        from app.modules.papers.citations import format_citation

        sep = "\n\n" if fmt == "BibTeX" else "\n"
        try:
            return sep.join(format_citation(p, fmt) for p in papers)
        except ValueError as exc:
            raise ValidationAppError(str(exc)) from exc

    def _ensure_paper(self, paper_id: str) -> Paper:
        paper = self.db.get(Paper, paper_id)
        if paper is None:
            raise NotFoundError(f"Paper '{paper_id}' not found")
        return paper

    def assert_owner(self, collection_id: str, owner_id: str) -> Collection:
        col = self.repo.get(collection_id)
        if col is None:
            raise NotFoundError(f"Collection '{collection_id}' not found")
        if col.owner_id != owner_id:
            raise ForbiddenError("You do not own this collection")
        return col


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _base() -> str:
    from app.core.config import settings

    return settings.app_base_url.rstrip("/")


def _fallback_synthesis(document: str, name: str) -> str:
    """Term-frequency extractive synthesis — the path when no LLM is configured."""
    import re

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", document) if s.strip()]
    papers = [s for s in sentences if s.startswith("- ")]
    head = f'"{name}" brings together {max(0, len(papers))} papers spanning '
    topics = sorted({w.lower() for w in re.findall(r"[A-Za-z]{6,}", document)})[:8]
    return head + (", ".join(topics) if topics else "a focused set of sources") + ". " + " ".join(papers[:3])


__all__ = ["CollectionService"]
