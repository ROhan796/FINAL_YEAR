"""Link prediction scorers.

Four scoring functions for triple plausibility:
- Dot: z_i^T · z_j
- DistMult: z_i^T · diag(r) · z_j (default)
- TransE: -‖z_i + r - z_j‖₂
- MLP: MLP([z_i ‖ r ‖ z_j])
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ScorerBase(ABC):
    """Base class for link prediction scorers."""

    @abstractmethod
    def score(self, z_src: np.ndarray, z_dst: np.ndarray,
              z_rel: np.ndarray | None = None) -> np.ndarray:
        """Score triples. Higher = more plausible."""
        ...


class DotScorer(ScorerBase):
    """Dot product: score = z_i^T · z_j"""

    def score(self, z_src: np.ndarray, z_dst: np.ndarray,
              z_rel: np.ndarray | None = None) -> np.ndarray:
        return np.sum(z_src * z_dst, axis=-1)


class DistMultScorer(ScorerBase):
    """DistMult: score = z_i^T · diag(r) · z_j = Σ_k z_ik · r_k · z_jk"""

    def score(self, z_src: np.ndarray, z_dst: np.ndarray,
              z_rel: np.ndarray | None = None) -> np.ndarray:
        if z_rel is None:
            return np.sum(z_src * z_dst, axis=-1)
        return np.sum(z_src * z_rel * z_dst, axis=-1)


class TransEScorer(ScorerBase):
    """TransE: score = -‖z_i + r - z_j‖₂"""

    def score(self, z_src: np.ndarray, z_dst: np.ndarray,
              z_rel: np.ndarray | None = None) -> np.ndarray:
        if z_rel is None:
            return -np.linalg.norm(z_src - z_dst, axis=-1)
        return -np.linalg.norm(z_src + z_rel - z_dst, axis=-1)


class MLPScorer(ScorerBase):
    """MLP: score = MLP([z_i ‖ r ‖ z_j])

    TRAINING STATUS: Architecture defined. Random weights.
    """

    def __init__(self, input_dim: int = 768 * 3, hidden_dim: int = 256) -> None:
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.01
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, 1).astype(np.float32) * 0.01
        self.b2 = np.zeros(1, dtype=np.float32)

    def score(self, z_src: np.ndarray, z_dst: np.ndarray,
              z_rel: np.ndarray | None = None) -> np.ndarray:
        if z_rel is None:
            z_rel = np.zeros_like(z_src)
        x = np.concatenate([z_src, z_rel, z_dst], axis=-1)
        h = np.maximum(x @ self.W1 + self.b1, 0)  # ReLU
        out = (h @ self.W2 + self.b2).squeeze(-1)
        return out


def get_scorer(name: str, dim: int = 768) -> ScorerBase:
    """Factory for scorers."""
    scorers = {
        "dot": DotScorer,
        "distmult": DistMultScorer,
        "transe": TransEScorer,
        "mlp": lambda: MLPScorer(dim * 3),
    }
    factory = scorers.get(name, DistMultScorer)
    return factory() if callable(factory) else factory


__all__ = ["ScorerBase", "DotScorer", "DistMultScorer", "TransEScorer", "MLPScorer", "get_scorer"]
