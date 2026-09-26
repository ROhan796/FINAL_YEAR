"""ML serving layer — online surface for the ML pipeline.

Provides:
- Query embedding (cached)
- Paper embedding (batch)
- Chunk embedding (batch)
- Hybrid retrieval
- RAG for chat assistant
"""

from __future__ import annotations

from ml.serving.embed_store import embed_queries, embed_papers, embed_chunks
from ml.serving.retrieval import hybrid_search
from ml.serving.rag import retrieve_chunks

__all__ = ["embed_queries", "embed_papers", "embed_chunks", "hybrid_search", "retrieve_chunks"]
