"""Business rules for the paper catalog. Owns transactions; router stays thin."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.core.errors import ConflictError, NotFoundError, ValidationAppError
from app.core.pagination import Cursor, Page, PageParams
from app.models.paper import Paper, SavedPaper
from app.modules.papers.citations import format_citation
from app.modules.papers.repository import PaperRepository
from app.schemas import CitationFormat, PaperFilters, PaperOut


class PaperService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo = PaperRepository(db)

    def list_papers(self, filters: PaperFilters, page: PageParams) -> Page[PaperOut]:
        rows, total = self.repo.list_papers(filters, page)
        return self._paginate(rows, page, total)

    def list_saved(self, user_id: str, query: str | None, page: PageParams) -> Page[PaperOut]:
        rows, total = self.repo.list_saved(user_id, query, page)
        return self._paginate(rows, page, total)

    @staticmethod
    def _paginate(
        rows: list[Paper],
        page: PageParams,
        total: int,
        key_fn: Any = None,
    ) -> Page[PaperOut]:
        has_more = len(rows) > page.limit
        visible = rows[:page.limit]
        next_cursor = None
        if has_more and visible:
            last = visible[-1]
            key = key_fn(last) if key_fn else last.citation_count
            next_cursor = Cursor(k=key, i=last.id).encode()
        return Page(
            items=[PaperOut.model_validate(p, from_attributes=True) for p in visible],
            nextCursor=next_cursor,
            total=total,
        )

    def get(self, paper_id: str) -> Paper:
        paper = self.repo.get(paper_id)
        if paper is None:
            raise NotFoundError(f"Paper '{paper_id}' not found")
        return paper

    def get_out(self, paper_id: str) -> PaperOut:
        return PaperOut.model_validate(self.get(paper_id), from_attributes=True)

    def related(self, paper_id: str, limit: int = 3) -> list[PaperOut]:
        self.get(paper_id)
        return [PaperOut.model_validate(p, from_attributes=True) for p in self.repo.related(paper_id, limit)]

    def toggle_save(self, user_id: str, paper_id: str) -> dict[str, Any]:
        self.get(paper_id)
        created = self.repo.save(user_id, paper_id)
        self.db.commit()
        return {"saved": True, "created": created}

    def unsave(self, user_id: str, paper_id: str) -> dict[str, Any]:
        removed = self.repo.unsave(user_id, paper_id)
        self.db.commit()
        if not removed:
            raise NotFoundError("Paper was not in your library")
        return {"saved": False}

    def saved_ids(self, user_id: str) -> list[str]:
        return self.repo.saved_ids(user_id)

    def citation(self, paper_id: str, fmt: CitationFormat) -> str:
        paper = self.get(paper_id)
        try:
            return format_citation(paper, fmt)
        except ValueError as exc:
            raise ValidationAppError(str(exc)) from exc

    # ── AI analysis ─────────────────────────────────────────────────────

    def analyze(self, paper_id: str, *, force: bool = False) -> PaperOut:
        """Summary + insights, cached on the row. Never recomputed inline by accident."""
        paper = self.get(paper_id)

        fresh = (
            paper.ai_analyzed_at is not None
            and paper.ai_summary
            and (datetime.now(UTC) - paper.ai_analyzed_at).days < 30
        )
        if fresh and not force:
            return PaperOut.model_validate(paper, from_attributes=True)

        from app.ai.service import summarize_paper, generate_insights  # lazy: LLM optional

        summary = summarize_paper(
            title=paper.title,
            abstract=paper.abstract,
            authors=paper.authors,
            venue=paper.venue,
            alias=f"paper:{paper.id}:summary",
        )
        insights = generate_insights(
            title=paper.title,
            abstract=paper.abstract,
            authors=paper.authors,
            tags=paper.tags,
            alias=f"paper:{paper.id}:insights",
        )

        paper.ai_summary = summary
        paper.key_contributions = insights.get("keyContributions") or []
        paper.methodology = insights.get("methodology") or ""
        paper.limitations = insights.get("limitations") or []
        paper.research_gaps = insights.get("researchGaps") or []
        paper.citation_context = insights.get("citationContext") or ""
        paper.ai_analyzed_at = datetime.now(UTC)
        self.db.add(paper)
        self.db.commit()
        self.db.refresh(paper)
        return PaperOut.model_validate(paper, from_attributes=True)

    def ensure_saved_row(self, user_id: str, paper_id: str) -> SavedPaper:  # pragma: no cover
        if not self.repo.is_saved(user_id, paper_id):
            row = SavedPaper(user_id=user_id, paper_id=paper_id)
            self.db.add(row)
            return row
        raise ConflictError("Already saved")
