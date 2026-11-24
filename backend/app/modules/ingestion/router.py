"""Ingestion endpoints — trigger arXiv / PubMed / Semantic Scholar ingestion & backfill."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from app.core.deps import CurrentUserDep, DbSession, ReadDbSession

router = APIRouter(prefix="/ingest", tags=["ingestion"])


class IngestTriggerBody(BaseModel):
    query: str = Field(default="graph neural networks", min_length=1, max_length=300)
    category: str | None = None
    maxResults: int = Field(default=15, ge=1, le=100)
    embed: bool = True
    buildKg: bool = True


async def _dispatch_ingest(
    source: str,
    payload: IngestTriggerBody,
    background: BackgroundTasks,
) -> dict[str, Any]:
    from tasks import enqueue_task

    job_id = await enqueue_task(
        "ingest_source",
        source,
        payload.query,
        payload.maxResults,
        payload.category,
        payload.embed,
        payload.buildKg,
    )
    if job_id:
        return {"queued": True, "jobId": job_id, "source": source, "backend": "arq"}

    # Redis/Arq unavailable -> run cleanly in FastAPI BackgroundTasks
    def _run_sync() -> None:
        from sqlalchemy.orm import Session

        from app.db.session import get_engine
        from ml.pipelines.index import IndexPipeline
        from ml.pipelines.ingest import IngestPipeline
        from ml.serving.backfill import backfill_all

        with Session(get_engine()) as db:
            pipe = IngestPipeline(db)
            if source == "pubmed":
                res = pipe.ingest_pubmed(payload.query, max_results=payload.maxResults)
            elif source == "semantic_scholar":
                res = pipe.ingest_semantic_scholar(payload.query, max_results=payload.maxResults)
            else:
                res = pipe.ingest_arxiv(
                    payload.query, max_results=payload.maxResults, category=payload.category
                )
            if payload.embed and res.papers_stored > 0:
                backfill_all(db, limit=payload.maxResults)
            if payload.buildKg and res.papers_stored > 0:
                IndexPipeline(db).index_all_missing(limit=payload.maxResults)

    background.add_task(_run_sync)
    return {"queued": True, "jobId": None, "source": source, "backend": "in-process"}


@router.post("/arxiv")
async def ingest_arxiv(
    payload: IngestTriggerBody,
    background: BackgroundTasks,
    user: CurrentUserDep,  # noqa: ARG001
) -> Any:
    return await _dispatch_ingest("arxiv", payload, background)


@router.post("/pubmed")
async def ingest_pubmed(
    payload: IngestTriggerBody,
    background: BackgroundTasks,
    user: CurrentUserDep,  # noqa: ARG001
) -> Any:
    return await _dispatch_ingest("pubmed", payload, background)


@router.post("/semantic-scholar")
async def ingest_semantic_scholar(
    payload: IngestTriggerBody,
    background: BackgroundTasks,
    user: CurrentUserDep,  # noqa: ARG001
) -> Any:
    return await _dispatch_ingest("semantic_scholar", payload, background)


@router.post("/backfill")
async def trigger_backfill(
    background: BackgroundTasks,
    user: CurrentUserDep,  # noqa: ARG001
    limit: int = 100,
) -> Any:
    from tasks import enqueue_task

    job_id = await enqueue_task("batch_embed", None, limit)
    if job_id:
        return {"queued": True, "jobId": job_id, "backend": "arq"}

    def _run() -> None:
        from sqlalchemy.orm import Session

        from app.db.session import get_engine
        from ml.serving.backfill import backfill_all

        with Session(get_engine()) as db:
            backfill_all(db, limit=limit)

    background.add_task(_run)
    return {"queued": True, "jobId": None, "backend": "in-process"}


@router.get("/status")
def ingest_status(db: ReadDbSession) -> dict[str, Any]:
    from sqlalchemy import func, select

    from app.core.config import settings
    from app.models.knowledge import Discovery, KgEdge, KgNode
    from app.models.paper import Paper, PaperChunk

    total_papers = int(db.scalar(select(func.count()).select_from(Paper)) or 0)
    embedded_papers = int(
        db.scalar(select(func.count()).select_from(Paper).where(Paper.embedding.is_not(None))) or 0
    )
    chunks = int(db.scalar(select(func.count()).select_from(PaperChunk)) or 0)
    nodes = int(db.scalar(select(func.count()).select_from(KgNode)) or 0)
    edges = int(db.scalar(select(func.count()).select_from(KgEdge)) or 0)
    discoveries = int(db.scalar(select(func.count()).select_from(Discovery)) or 0)

    sources_rows = db.execute(select(Paper.source, func.count()).group_by(Paper.source)).all()
    return {
        "totalPapers": total_papers,
        "embeddedPapers": embedded_papers,
        "paperChunks": chunks,
        "kgNodes": nodes,
        "kgEdges": edges,
        "discoveries": discoveries,
        "sources": {str(s): int(c) for s, c in sources_rows},
        "adapters": {
            "arxiv": settings.arxiv_enabled,
            "pubmed": settings.pubmed_enabled,
            "semanticScholar": settings.semantic_scholar_enabled,
        },
        "queueEnabled": settings.queue_enabled,
    }


__all__ = ["router"]
