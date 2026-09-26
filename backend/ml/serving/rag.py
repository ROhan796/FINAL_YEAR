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
