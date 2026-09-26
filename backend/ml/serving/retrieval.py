"""Hybrid retrieval — combine vector search + FTS + KG for search results.

Per spec:
- Arm A: pgvector HNSW cosine (vector search)
- Arm B: PostgreSQL tsvector (full-text)
- Arm C: KG entity linking (hybrid only)
- Reciprocal Rank Fusion (RRF, k=60)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RetrievedItem:
    """A single retrieval result."""
    id: str
    score: float = 0.0
    cosine: float = 0.0
    arms: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)


def reciprocal_rank_fusion(rankings: dict[str, list[str]], k: int = 60) -> dict[str, float]:
    """RRF over {arm: [doc_id, ...]} rankings."""
    scores: dict[str, float] = {}
    for ranking in rankings.values():
        for i, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + i + 1)
    return scores


def cosine_similarity_matrix(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and all documents."""
    query_norm = np.linalg.norm(query_emb)
    doc_norms = np.linalg.norm(doc_embs, axis=1)
    if query_norm == 0:
        return np.zeros(len(doc_embs))
    denom = doc_norms * query_norm
    denom[denom == 0] = 1.0
    return doc_embs @ query_emb / denom


def hybrid_search(
    query: str,
    query_embedding: np.ndarray | None,
    doc_ids: list[str],
    doc_embeddings: np.ndarray | None,
    doc_texts: list[str] | None = None,
    mode: str = "hybrid",
    limit: int = 20,
) -> list[RetrievedItem]:
    """Run hybrid search combining vector and text retrieval.

    This is the ML-serving-level implementation. The actual Postgres-backed
    implementation lives in app/modules/search/repository.py.
    """
    candidates: dict[str, RetrievedItem] = {}

    # Arm A: Vector search
    if mode in ("semantic", "hybrid") and query_embedding is not None and doc_embeddings is not None:
        sims = cosine_similarity_matrix(query_embedding, doc_embeddings)
        top_k = np.argsort(sims)[::-1][:limit]
        for rank, idx in enumerate(top_k):
            doc_id = doc_ids[idx]
            item = candidates.setdefault(doc_id, RetrievedItem(id=doc_id))
            item.cosine = float(sims[idx])
            item.arms.append("vector")

    # Arm B: Text/keyword search
    if mode in ("keyword", "hybrid") and doc_texts is not None:
        terms = set(query.lower().split())
        scores = {}
        for i, text in enumerate(doc_texts):
            text_lower = text.lower()
            score = sum(1 for t in terms if t in text_lower)
            if score > 0:
                scores[doc_ids[i]] = score
        sorted_docs = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]
        for doc_id in sorted_docs:
            item = candidates.setdefault(doc_id, RetrievedItem(id=doc_id))
            item.arms.append("fts")

    # RRF fusion
    if mode == "hybrid":
        vector_rank = {did: i for i, did in enumerate(
            sorted(candidates.keys(), key=lambda x: candidates[x].cosine, reverse=True)
        ) if "vector" in candidates[x].arms}
        fts_rank = {did: i for i, did in enumerate(
            sorted(candidates.keys(), key=lambda x: len(candidates[x].matched_terms), reverse=True)
        ) if "fts" in candidates[x].arms}

        fused = reciprocal_rank_fusion({"vector": list(vector_rank.keys()), "fts": list(fts_rank.keys())})
        for doc_id, score in fused.items():
            if doc_id in candidates:
                candidates[doc_id].score = score

    # Sort by score
    results = sorted(candidates.values(), key=lambda x: x.score, reverse=True)
    return results[:limit]
