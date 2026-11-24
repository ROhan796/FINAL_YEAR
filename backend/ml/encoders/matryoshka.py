"""Matryoshka Representation Learning (MRL) embeddings support.

Enables multi-scale embeddings (e.g. 768-d -> 128-d prefix truncation + L2 norm)
for 6x faster coarse retrieval followed by exact 768-d reranking, plus a
PyTorch MRL contrastive loss and optional linear projection head.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from ml.encoders import EncodeKind, EncoderBase
from ml.encoders.registry import get_encoder, register_encoder

log = logging.getLogger(__name__)

MRL_SCALES: tuple[int, ...] = (64, 128, 256, 512, 768)


def truncate_and_normalize(vectors: np.ndarray, dim: int = 128) -> np.ndarray:
    """Slice first `dim` dimensions and L2-renormalize."""
    arr = np.asarray(vectors, dtype=np.float32)
    single = arr.ndim == 1
    if single:
        arr = arr.reshape(1, -1)
    d = min(dim, arr.shape[1])
    sub = arr[:, :d].copy()
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out = (sub / norms).astype(np.float32)
    return out[0] if single else out


def two_stage_matryoshka_rank(
    query_vec: np.ndarray,
    corpus_ids: list[str],
    corpus_matrix: np.ndarray,
    *,
    coarse_dim: int = 128,
    shortlist_k: int = 150,
    top_k: int = 50,
) -> list[tuple[str, float]]:
    """Fast 128-d Matryoshka coarse shortlist -> 768-d exact cosine rerank."""
    if len(corpus_ids) == 0 or corpus_matrix.size == 0:
        return []

    q_coarse = truncate_and_normalize(query_vec, coarse_dim)
    m_coarse = truncate_and_normalize(corpus_matrix, coarse_dim)
    coarse_scores = m_coarse @ q_coarse

    k_short = min(shortlist_k, len(corpus_ids))
    short_idx = np.argsort(coarse_scores)[::-1][:k_short]

    q_full = truncate_and_normalize(query_vec, corpus_matrix.shape[1])
    m_sub = truncate_and_normalize(corpus_matrix[short_idx], corpus_matrix.shape[1])
    fine_scores = m_sub @ q_full

    order = np.argsort(fine_scores)[::-1][:top_k]
    return [(corpus_ids[int(short_idx[i])], float(fine_scores[i])) for i in order]


def matryoshka_contrastive_loss(
    anchors,
    positives,
    scales: Sequence[int] = MRL_SCALES,
    temperature: float = 0.07,
):
    """PyTorch Matryoshka InfoNCE loss summed across prefix scales."""
    import torch
    import torch.nn.functional as F

    total_loss = torch.tensor(0.0, device=anchors.device)
    labels = torch.arange(anchors.size(0), device=anchors.device)
    active = 0
    for m in scales:
        if m > anchors.size(-1):
            continue
        a_m = F.normalize(anchors[:, :m], p=2, dim=-1)
        p_m = F.normalize(positives[:, :m], p=2, dim=-1)
        logits = (a_m @ p_m.T) / temperature
        total_loss = total_loss + F.cross_entropy(logits, labels)
        active += 1
    return total_loss / max(1, active)


@register_encoder("matryoshka")
class MatryoshkaEncoder(EncoderBase):
    """Matryoshka-wrapped scientific encoder (SPECTER2 backbone + multi-scale output)."""

    def __init__(self, dim: int = 768, device: str = "cpu", base_name: str = "specter2") -> None:
        self.dim = dim
        self.name = "matryoshka"
        self._device = device
        self._base_name = base_name
        self._base: EncoderBase | None = None

    def _ensure_base(self) -> EncoderBase:
        if self._base is None:
            self._base = get_encoder(self._base_name, dim=768, device=self._device)
        return self._base

    def encode(self, texts: list[str], *, kind: EncodeKind = "paper") -> np.ndarray:
        full = self._ensure_base().encode(texts, kind=kind)
        return truncate_and_normalize(full, self.dim)

    def warmup(self) -> None:
        self._ensure_base().warmup()


__all__ = [
    "MRL_SCALES",
    "MatryoshkaEncoder",
    "matryoshka_contrastive_loss",
    "truncate_and_normalize",
    "two_stage_matryoshka_rank",
]
