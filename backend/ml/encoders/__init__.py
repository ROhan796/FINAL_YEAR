"""Encoder base protocol and types.

Every encoder implements the same interface so the rest of the pipeline
is model-agnostic. When torch is unavailable, a numpy fallback provides
deterministic hash-based embeddings for development and testing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

EncodeKind = Literal["paper", "passage", "entity", "query"]


class EncoderBase(ABC):
    """Protocol for all text encoders in the CKGAI pipeline."""

    dim: int
    name: str

    @abstractmethod
    def encode(self, texts: list[str], *, kind: EncodeKind = "passage") -> np.ndarray:
        """Encode a batch of texts into dense vectors.

        Args:
            texts: List of strings to encode.
            kind: The type of text (paper, passage, entity, query).

        Returns:
            np.ndarray of shape (len(texts), self.dim), dtype float32.
        """
        ...

    def encode_single(self, text: str, *, kind: EncodeKind = "passage") -> np.ndarray:
        """Encode a single text. Convenience wrapper."""
        return self.encode([text], kind=kind)[0]

    def warmup(self) -> None:
        """Pre-load model weights into memory. Called at startup."""
        pass


class HashEncoder(EncoderBase):
    """Deterministic numpy fallback — no torch required.

    Produces 768-d vectors via a seeded hash of each token. These are NOT
    semantically meaningful but are deterministic, fast, and allow the full
    pipeline to be tested without GPU or model downloads.

    Clearly labeled as DEVELOPMENT / PLACEHOLDER — never use in production.
    """

    def __init__(self, dim: int = 768, name: str = "hash-fallback") -> None:
        self.dim = dim
        self.name = name
        self._rng = np.random.RandomState(42)

    def encode(self, texts: list[str], *, kind: EncodeKind = "passage") -> np.ndarray:
        results = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # Deterministic seed from text content
            seed = int.from_bytes(
                text.encode("utf-8")[:8].ljust(8, b"\x00"), "big"
            ) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dim).astype(np.float32)
            # Normalize to unit vector
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            results[i] = vec
        return results

    def warmup(self) -> None:
        # Hash encoder needs no warmup
        _ = self.encode(["warmup"])


__all__ = ["EncoderBase", "HashEncoder", "EncodeKind"]
