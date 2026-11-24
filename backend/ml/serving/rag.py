"""RAG retrieval — retrieve relevant chunks for workspace chat assistant.

Flow:
1. Embed query
2. Retrieve relevant chunks from project papers
3. Construct context for LLM
"""

from __future__ import annotations

import logging

import numpy as np

from ml.serving.embed_store import embed_queries, embed_chunks

log = logging.getLogger(__name__)


def retrieve_chunks(
    query: str,
    chunk_texts: list[str],
    chunk_metadata: list[dict] | None = None,
    top_k: int = 8,
    max_tokens: int = 4000,
) -> list[dict]:
    """Retrieve relevant chunks for RAG.

    Args:
        query: User's question
        chunk_texts: List of chunk text passages
        chunk_metadata: Optional metadata per chunk (paper_id, section, etc.)
        top_k: Number of chunks to retrieve
        max_tokens: Approximate token budget for context

    Returns:
        List of dicts with 'text', 'score', 'metadata'
    """
    if not chunk_texts:
        return []

    try:
        # Embed query and chunks
        query_emb = embed_queries([query])[0]
        chunk_embs = embed_chunks(chunk_texts)

        # Cosine similarity
        chunk_norms = np.linalg.norm(chunk_embs, axis=1)
        chunk_norms[chunk_norms == 0] = 1.0
        scores = chunk_embs @ query_emb / (chunk_norms * np.linalg.norm(query_emb) + 1e-9)

        # Top-k retrieval
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        total_len = 0
        for idx in top_indices:
            text = chunk_texts[idx]
            # Rough token estimate (4 chars per token)
            est_tokens = len(text) // 4
            if total_len + est_tokens > max_tokens:
                break
            total_len += est_tokens

            result = {
                "text": text,
                "score": float(scores[idx]),
                "metadata": chunk_metadata[idx] if chunk_metadata else {},
            }
            results.append(result)

        return results

    except Exception as exc:
        log.warning("rag_retrieval_failed: %s", exc)
        # Fallback: return first few chunks
        return [
            {"text": t, "score": 0.0, "metadata": {}}
            for t in chunk_texts[:top_k]
        ]


def retrieve_for_project(
    db,
    paper_ids: list[str],
    query: str,
    *,
    top_k: int = 8,
    max_tokens: int = 4000,
) -> list[dict]:
    """DB-backed RAG: pull `paper_chunks` for a project's pinned papers and
    rank them against the user message.

    Uses stored chunk embeddings when present (cosine), otherwise a cheap
    lexical overlap score. Never raises — chat must work without ML.
    """
    if not paper_ids:
        return []
    try:
        from sqlalchemy import select

        from app.models.paper import PaperChunk

        rows = list(
            db.scalars(
                select(PaperChunk).where(PaperChunk.paper_id.in_(list(paper_ids))).limit(400)
            ).all()
        )
        if not rows:
            return []
        texts = [r.text for r in rows]
        meta = [
            {"paper_id": r.paper_id, "section": r.section, "chunk_id": r.id} for r in rows
        ]

        # Prefer stored embeddings (already backfilled) over re-encoding.
        if all(r.embedding is not None for r in rows[: min(32, len(rows))]):
            import numpy as np

            from ml.serving.embed_store import embed_queries

            try:
                q = embed_queries([query])[0]
                mat = np.array([np.asarray(r.embedding, dtype=np.float32) for r in rows])
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                qn = np.linalg.norm(q)
                scores = (mat / norms) @ (q / (qn or 1.0))
                order = np.argsort(scores)[::-1][:top_k]
                out, total = [], 0
                for i in order:
                    est = len(texts[int(i)]) // 4
                    if total + est > max_tokens:
                        break
                    total += est
                    out.append(
                        {"text": texts[int(i)], "score": float(scores[int(i)]), "metadata": meta[int(i)]}
                    )
                return out
            except Exception:  # noqa: BLE001 - fall through to lexical
                pass

        # Lexical fallback (also the no-ML path)
        terms = {t for t in query.lower().split() if len(t) > 2}
        scored = []
        for i, t in enumerate(texts):
            tl = t.lower()
            score = sum(1 for term in terms if term in tl)
            if score:
                scored.append((score, i))
        scored.sort(reverse=True)
        out, total = [], 0
        for score, i in scored[:top_k]:
            est = len(texts[i]) // 4
            if total + est > max_tokens:
                break
            total += est
            out.append({"text": texts[i], "score": float(score), "metadata": meta[i]})
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("retrieve_for_project_failed: %s", exc)
        return []


__all__ = ["retrieve_chunks", "retrieve_for_project"]
