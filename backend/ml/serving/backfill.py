"""Embedding backfill — make semantic search actually semantic.

Fills `papers.embedding` (SPECTER2, 768-d) and `paper_chunks.embedding`
(SciBERT mean-pool) for every row where the vector is NULL, then rebuilds
the `related_papers` k-NN table.

Run:
    python -m scripts.backfill_embeddings            # all missing
    python -m scripts.backfill_embeddings --limit 50
    python -m scripts.backfill_embeddings --related  # only rebuild k-NN
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

BATCH = 32


def _paper_text(title: str, abstract: str) -> str:
    """Official SPECTER2 input format: title [SEP] abstract."""
    return f"{title} [SEP] {abstract or ''}".strip()


def backfill_paper_embeddings(db: Any, *, limit: int | None = None, batch: int = BATCH) -> int:
    """Encode papers with NULL embedding using the configured paper encoder."""
    from sqlalchemy import select

    from app.core.config import settings
    from app.models.paper import Paper
    from ml.encoders.registry import get_encoder, resolve_device

    stmt = select(Paper).where(Paper.embedding.is_(None)).order_by(Paper.created_at)
    if limit:
        stmt = stmt.limit(limit)
    papers = list(db.scalars(stmt).all())
    if not papers:
        log.info("backfill_papers: nothing to embed")
        return 0

    encoder = get_encoder(
        settings.paper_encoder, dim=settings.embedding_dim, device=resolve_device(settings.ml_device)
    )
    log.info("backfill_papers: %d papers via %s", len(papers), encoder.name)

    encoded = 0
    for i in range(0, len(papers), batch):
        chunk = papers[i : i + batch]
        texts = [_paper_text(p.title, p.abstract) for p in chunk]
        vectors = encoder.encode(texts, kind="paper")
        for p, vec in zip(chunk, vectors, strict=False):
            p.embedding = [float(x) for x in vec]
        db.commit()
        encoded += len(chunk)
        log.info("backfill_papers: %d/%d", encoded, len(papers))
    return encoded


def backfill_chunk_embeddings(db: Any, *, limit: int | None = None, batch: int = BATCH) -> int:
    """Encode paper chunks (RAG passages) with the passage encoder."""
    from sqlalchemy import select

    from app.core.config import settings
    from app.models.paper import PaperChunk
    from ml.encoders.registry import get_encoder, resolve_device

    stmt = select(PaperChunk).where(PaperChunk.embedding.is_(None))
    if limit:
        stmt = stmt.limit(limit)
    chunks = list(db.scalars(stmt).all())
    if not chunks:
        log.info("backfill_chunks: nothing to embed")
        return 0

    encoder = get_encoder(
        settings.passage_encoder, dim=settings.embedding_dim, device=resolve_device(settings.ml_device)
    )
    encoded = 0
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        vectors = encoder.encode([c.text for c in part], kind="passage")
        for c, vec in zip(part, vectors, strict=False):
            c.embedding = [float(x) for x in vec]
        db.commit()
        encoded += len(part)
    log.info("backfill_chunks: %d via %s", encoded, encoder.name)
    return encoded


def rebuild_related_papers(db: Any, *, k: int = 5, limit: int | None = None) -> int:
    """Cosine k-NN over papers.embedding → related_papers table."""
    from sqlalchemy import delete, select

    from app.models.paper import Paper, RelatedPaper

    stmt = select(Paper).where(Paper.embedding.is_not(None))
    if limit:
        stmt = stmt.limit(limit)
    papers = list(db.scalars(stmt).all())
    if len(papers) < 2:
        return 0

    ids: list[str] = []
    vectors: list[Any] = []
    for p in papers:
        emb = p.embedding
        if emb is None:
            continue
        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim != 1 or arr.size < 8:  # guard against 0-d / truncated rows
            continue
        ids.append(p.id)
        vectors.append(arr)
    if len(ids) < 2:
        return 0

    mat = np.vstack(vectors)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms

    sim = mat @ mat.T
    np.fill_diagonal(sim, -1.0)

    db.execute(delete(RelatedPaper))
    count = 0
    for i, pid in enumerate(ids):
        top = np.argsort(sim[i])[::-1][:k]
        for j in top:
            if sim[i, j] <= 0:
                continue
            db.add(
                RelatedPaper(paper_id=pid, related_to_id=ids[int(j)], score=float(sim[i, j]))
            )
            count += 1
    db.commit()
    log.info("rebuild_related: %d pairs over %d papers", count, len(papers))
    return count


def backfill_all(db: Any, *, limit: int | None = None) -> dict[str, int]:
    """Run every backfill stage in the correct order."""
    started = time.perf_counter()
    papers = backfill_paper_embeddings(db, limit=limit)
    chunks = backfill_chunk_embeddings(db, limit=limit)
    related = rebuild_related_papers(db)
    return {
        "papers_embedded": papers,
        "chunks_embedded": chunks,
        "related_pairs": related,
        "seconds": int(time.perf_counter() - started),
    }


__all__ = [
    "backfill_all",
    "backfill_chunk_embeddings",
    "backfill_paper_embeddings",
    "rebuild_related_papers",
]
