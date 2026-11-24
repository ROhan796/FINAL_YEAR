"""HTTP layer for papers. No SQL here — only validation and delegation."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from app.core.deps import CurrentUserDep, DbSession, PageParams, Pagination, ReadDbSession, require
from app.models.paper import Paper
from app.schemas import (
    CitationFormat,
    CitationOut,
    OkResponse,
    PaperFilters,
    PaperOut,
    SearchRequest,
)
from app.schemas.common import PageResponse
from app.modules.papers.service import PaperService

router = APIRouter(prefix="/papers", tags=["papers"])


def _service(db: ReadDbSession) -> PaperService:
    return PaperService(db)


def _filters(
    q: Annotated[str | None, Query(max_length=500)] = None,
    tag: Annotated[str | None, Query(max_length=160)] = None,
    year: Annotated[int | None, Query(ge=1900, le=2100)] = None,
    minCitations: Annotated[int | None, Query(ge=0, alias="minCitations")] = None,
    openAccess: Annotated[bool | None, Query(alias="openAccess")] = None,
    sort: Annotated[str, Query(pattern="^(relevance|citations|newest|year)$")] = "relevance",
) -> PaperFilters:
    return PaperFilters(
        q=q, tag=tag, year=year, minCitations=minCitations, openAccess=openAccess, sort=sort  # type: ignore[arg-type]
    )


@router.get("", response_model=PageResponse[PaperOut])
def list_papers(
    db: ReadDbSession,
    filters: Annotated[PaperFilters, Depends(_filters)],
    page: Annotated[PageParams, Depends(Pagination(default=20))],
) -> Any:
    return PaperService(db).list_papers(filters, page)


@router.get("/tags", response_model=list[str])
def list_tags(db: ReadDbSession) -> Any:
    """Discipline chips on the Discover view."""
    return PaperService(db).repo.all_tags()


@router.get("/{paper_id}", response_model=PaperOut)
def get_paper(paper_id: str, db: ReadDbSession) -> Any:
    return PaperService(db).get_out(paper_id)


@router.get("/{paper_id}/related", response_model=list[PaperOut])
def related_papers(
    paper_id: str,
    db: ReadDbSession,
    limit: Annotated[int, Query(ge=1, le=10)] = 3,
) -> Any:
    return PaperService(db).related(paper_id, limit)


@router.get("/{paper_id}/citation", response_model=CitationOut)
def citation(
    paper_id: str,
    db: ReadDbSession,
    format: Annotated[CitationFormat, Query()] = "APA",
) -> Any:
    return CitationOut(format=format, citation=PaperService(db).citation(paper_id, format))


@router.post("/{paper_id}/save", response_model=OkResponse)
def save_paper(paper_id: str, db: DbSession, user: CurrentUserDep) -> Any:
    PaperService(db).toggle_save(user["id"], paper_id)
    return OkResponse()


@router.delete("/{paper_id}/save", response_model=OkResponse)
def unsave_paper(paper_id: str, db: DbSession, user: CurrentUserDep) -> Any:
    PaperService(db).unsave(user["id"], paper_id)
    return OkResponse()


@router.post("/{paper_id}/ai-analysis", response_model=PaperOut)
def analyze_paper(
    paper_id: str,
    db: DbSession,
    user: Annotated[Any, Depends(require("search"))],
    force: Annotated[bool, Query(description="Recompute instead of using the cache")] = False,
) -> Any:
    return PaperService(db).analyze(paper_id, force=force)


@router.post("/{paper_id}/citation-polarity")
def citation_polarity(
    paper_id: str,
    db: ReadDbSession,
    citationSentence: Annotated[str, Query(max_length=2000)] = "",
) -> dict[str, Any]:
    """6.A: Classify whether a citing sentence supports, disputes, or neutrally cites this paper."""
    from pathlib import Path
    from app.core.config import settings
    from ml.relation.citation_polarity import CitationPolarityClassifier

    paper = PaperService(db).get(paper_id)
    sentence = citationSentence.strip() or (
        f"Building upon {paper.title}, recent work validates its empirical findings."
    )
    ckpt = Path(settings.ml_checkpoint_dir) / "citation_polarity.pt"
    clf = CitationPolarityClassifier(ckpt if ckpt.exists() else None)
    res = clf.classify(sentence, paper.abstract)
    return {
        "paperId": paper.id,
        "citationSentence": sentence,
        "label": res.label,
        "confidence": res.confidence,
        "probabilities": res.probabilities,
        "evidence": res.evidence,
        "modelUsed": res.model_used,
    }


saved_router = APIRouter(tags=["papers"])


@saved_router.get("/saved", response_model=PageResponse[PaperOut])
def list_saved(
    db: ReadDbSession,
    user: CurrentUserDep,
    page: Annotated[PageParams, Depends(Pagination(default=20))],
    query: Annotated[str | None, Query(max_length=500)] = None,
) -> Any:
    """LibraryView: the signed-in researcher's saved papers."""
    return PaperService(db).list_saved(user["id"], query, page)


__all__ = ["router", "saved_router", "PaperService", "Paper", "SearchRequest"]
