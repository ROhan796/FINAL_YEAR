"""Background job definitions for Arq workers.

Heavy operations are enqueued here to avoid blocking API requests:
- Paper ingestion from arXiv / PubMed / Semantic Scholar
- Embedding generation (SPECTER2 / SciBERT batch backfill)
- NER + RE + KG construction
- Neo4j incremental projection
- PyTorch GNN training & cross-domain discovery generation
- Feedback-driven Platt calibration retraining
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


async def ingest_paper_job(ctx: dict, paper_data: dict) -> dict:  # noqa: ARG001
    """Background job: index a single paper through NER -> RE -> KG."""
    from sqlalchemy.orm import Session

    from app.db.session import get_engine
    from ml.pipelines.index import IndexPipeline

    with Session(get_engine()) as db:
        pipeline = IndexPipeline(db)
        result = pipeline.index_paper(
            paper_id=paper_data.get("id", ""),
            title=paper_data.get("title", ""),
            abstract=paper_data.get("abstract", ""),
            domain=paper_data.get("domain", ""),
        )
    return {
        "papers_indexed": result.papers_indexed,
        "entities_extracted": result.entities_extracted,
        "relations_extracted": result.relations_extracted,
        "kg_nodes_created": result.kg_nodes_created,
        "kg_edges_created": result.kg_edges_created,
    }


async def ingest_source_job(
    ctx: dict,  # noqa: ARG001
    source: str,
    query: str,
    max_results: int = 25,
    category: str | None = None,
    embed: bool = True,
    build_kg: bool = True,
) -> dict[str, Any]:
    """Background job: fetch from arXiv/PubMed/S2, embed, and optionally build KG."""
    from sqlalchemy.orm import Session

    from app.db.session import get_engine
    from ml.pipelines.index import IndexPipeline
    from ml.pipelines.ingest import IngestPipeline
    from ml.serving.backfill import backfill_all

    with Session(get_engine()) as db:
        ingest = IngestPipeline(db)
        if source == "pubmed":
            res = ingest.ingest_pubmed(query, max_results=max_results)
        elif source == "semantic_scholar":
            res = ingest.ingest_semantic_scholar(query, max_results=max_results)
        else:
            res = ingest.ingest_arxiv(query, max_results=max_results, category=category)

        backfill_stats: dict[str, int] = {}
        if embed and res.papers_stored > 0:
            backfill_stats = backfill_all(db, limit=max_results)

        kg_stats: dict[str, int] = {}
        if build_kg and res.papers_stored > 0:
            idx = IndexPipeline(db).index_all_missing(limit=max_results)
            kg_stats = {"nodes": idx.kg_nodes_created, "edges": idx.kg_edges_created}

    return {
        "source": source,
        "fetched": res.papers_fetched,
        "stored": res.papers_stored,
        "skipped": res.papers_skipped,
        "chunks": res.chunks_created,
        "backfill": backfill_stats,
        "kg": kg_stats,
        "errors": res.errors,
    }


async def batch_embed_job(ctx: dict, paper_ids: list[str] | None = None, limit: int = 200) -> dict:  # noqa: ARG001
    """Background job: backfill real SPECTER2/SciBERT embeddings and related_papers."""
    from sqlalchemy.orm import Session

    from app.db.session import get_engine
    from ml.serving.backfill import backfill_all

    with Session(get_engine()) as db:
        stats = backfill_all(db, limit=limit)
    return {"requested": len(paper_ids or []), **stats}


async def project_neo4j_job(ctx: dict, full: bool = False) -> dict:  # noqa: ARG001
    """Background job: incrementally project Postgres KG edges to Neo4j."""
    from sqlalchemy.orm import Session

    from app.db.session import get_engine
    from ml.graph.projector import Neo4jProjector

    with Session(get_engine()) as db:
        res = Neo4jProjector(db).project(full=full)
    return {
        "status": res.status,
        "nodes_projected": res.nodes_projected,
        "edges_projected": res.edges_projected,
        "watermark": res.watermark,
        "error": res.error,
    }


async def discovery_job(ctx: dict, train: bool = True, epochs: int = 25) -> dict:  # noqa: ARG001
    """Background job: run GNN + DistMult cross-domain discovery and persist results."""
    from sqlalchemy.orm import Session

    from app.db.session import get_engine
    from ml.pipelines.discovery import DiscoveryPipeline

    with Session(get_engine()) as db:
        pipeline = DiscoveryPipeline(db)
        result = pipeline.run(train=train, epochs=epochs, persist=True)
    return {
        "discoveries_found": len(result.discoveries),
        "total_candidates": result.total_candidates,
        "model_trained": result.model_trained,
        "persisted": result.persisted_count,
        "hits@10": result.hits_at_10,
        "mrr": result.mrr,
    }


async def retrain_discovery_job(ctx: dict) -> dict:  # noqa: ARG001
    """Background job: recalibrate discovery scores from user feedback votes."""
    from sqlalchemy.orm import Session

    from app.core.config import settings
    from app.db.session import get_engine
    from ml.discovery.calibration import recalibrate_from_db

    with Session(get_engine()) as db:
        return recalibrate_from_db(db, checkpoint_dir=settings.ml_checkpoint_dir)


async def gnn_training_job(ctx: dict) -> dict:  # noqa: ARG001
    """Scheduled job: retrain CKGAI_GNN on latest KG and refresh discoveries."""
    return await discovery_job(ctx, train=True, epochs=30)


from arq import cron
from arq.connections import RedisSettings


def _redis_settings() -> RedisSettings:
    from app.core.config import settings

    return RedisSettings.from_dsn(settings.redis_url)


async def enqueue_task(job_name: str, *args: Any, **kwargs: Any) -> str | None:
    """Best-effort Arq enqueue. Returns job_id or None if Redis/queue is unavailable."""
    from app.core.config import settings

    if not settings.queue_enabled:
        return None
    try:
        from arq import create_pool

        pool = await create_pool(_redis_settings())
        try:
            job = await pool.enqueue_job(job_name, *args, **kwargs)
            return job.job_id if job else None
        finally:
            await pool.aclose()
    except Exception as exc:  # noqa: BLE001 - Redis optional in dev
        log.debug("arq_enqueue_unavailable (%s): %s", job_name, exc)
        return None


JOB_DEFINITIONS = {
    "ingest_paper": ingest_paper_job,
    "ingest_source": ingest_source_job,
    "batch_embed": batch_embed_job,
    "project_neo4j": project_neo4j_job,
    "run_discovery": discovery_job,
    "retrain_discovery": retrain_discovery_job,
    "train_gnn": gnn_training_job,
}

CRON_JOBS = [
    cron(gnn_training_job, hour=2, minute=0),
    cron(project_neo4j_job, minute={0, 15, 30, 45}),
]


class WorkerSettings:
    """Entrypoint for `arq tasks.WorkerSettings`."""

    functions = [
        ingest_paper_job,
        ingest_source_job,
        batch_embed_job,
        project_neo4j_job,
        discovery_job,
        retrain_discovery_job,
        gnn_training_job,
    ]
    cron_jobs = CRON_JOBS
    redis_settings = _redis_settings()


__all__ = [
    "CRON_JOBS",
    "JOB_DEFINITIONS",
    "WorkerSettings",
    "batch_embed_job",
    "discovery_job",
    "enqueue_task",
    "gnn_training_job",
    "ingest_paper_job",
    "ingest_source_job",
    "project_neo4j_job",
    "retrain_discovery_job",
]
