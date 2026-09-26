"""Embedding store — encode and cache query/paper/chunk embeddings.

Uses the configured encoder (SPECTER2 for papers, SciBERT for passages).
Falls back to HashEncoder when models are unavailable.
"""

from __future__ import annotations

import logging

import numpy as np

from ml.encoders.registry import get_encoder

log = logging.getLogger(__name__)

# Module-level encoder instances (lazy-loaded)
_paper_encoder = None
_passage_encoder = None


def _get_paper_encoder():
    global _paper_encoder
    if _paper_encoder is None:
        from app.core.config import settings
        _paper_encoder = get_encoder(
            settings.paper_encoder,
            dim=settings.embedding_dim,
            device=settings.ml_device,
        )
    return _paper_encoder


def _get_passage_encoder():
    global _passage_encoder
    if _passage_encoder is None:
        from app.core.config import settings
        _passage_encoder = get_encoder(
            settings.passage_encoder,
            dim=settings.embedding_dim,
            device=settings.ml_device,
        )
    return _passage_encoder


def embed_queries(texts: list[str]) -> np.ndarray:
    """Embed search queries using the paper encoder.

    Args:
        texts: List of query strings.

    Returns:
        np.ndarray of shape (len(texts), dim)
    """
    encoder = _get_paper_encoder()
    return encoder.encode(texts, kind="query")


def embed_papers(texts: list[str]) -> np.ndarray:
    """Embed paper titles+abstracts.

    Args:
        texts: List of "Title: ... Abstract: ..." strings.

    Returns:
        np.ndarray of shape (len(texts), dim)
    """
    encoder = _get_paper_encoder()
    return encoder.encode(texts, kind="paper")


def embed_chunks(texts: list[str]) -> np.ndarray:
    """Embed text chunks/passages for RAG retrieval.

    Args:
        texts: List of passage strings.

    Returns:
        np.ndarray of shape (len(texts), dim)
    """
    encoder = _get_passage_encoder()
    return encoder.encode(texts, kind="passage")


def embed_entities(texts: list[str]) -> np.ndarray:
    """Embed entity names for KG node initialization.

    Args:
        texts: List of entity name strings.

    Returns:
        np.ndarray of shape (len(texts), dim)
    """
    encoder = _get_passage_encoder()
    return encoder.encode(texts, kind="entity")
