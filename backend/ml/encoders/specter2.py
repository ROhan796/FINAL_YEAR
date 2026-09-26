"""SPECTER2 encoder — paper-level citation-aware embeddings.

Uses allenai/specter2_base for paper-level semantic search.
SPECTER2 is trained on citation links — papers that cite each other
are close even with disjoint vocabularies.

When torch/transformers is unavailable, falls back to HashEncoder.
"""

from __future__ import annotations

import logging

import numpy as np

from ml.encoders import EncodeKind, EncoderBase, HashEncoder
from ml.encoders.registry import register_encoder

log = logging.getLogger(__name__)


@register_encoder("specter2")
class SPECTER2Encoder(EncoderBase):
    """SPECTER2 encoder for paper-level embeddings.

    Optimal for paper semantic search, related papers, and entity resolution.
    Uses title+abstract concatenation with CLS pooling.
    """

    def __init__(self, dim: int = 768, device: str = "cpu") -> None:
        self.dim = dim
        self.name = "specter2"
        self._device = device
        self._model = None
        self._tokenizer = None
        self._fallback: HashEncoder | None = None

    def _load(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            model_name = "allenai/specter2_base"
            log.info("loading_specter2: model=%s device=%s", model_name, self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name).to(self._device)
            self._model.eval()
            return True
        except Exception as exc:
            log.warning("specter2_load_failed: %s", exc)
            self._fallback = HashEncoder(dim=self.dim, name="specter2-fallback")
            return False

    def encode(self, texts: list[str], *, kind: EncodeKind = "paper") -> np.ndarray:
        if not self._load():
            return self._fallback.encode(texts, kind=kind)

        import torch

        all_embeddings = []
        batch_size = 16  # SPECTER2 is larger, use smaller batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**encoded)
                # SPECTER2 uses CLS token pooling
                emb = outputs.last_hidden_state[:, 0, :]
                # L2 normalize
                norms = torch.norm(emb, p=2, dim=1, keepdim=True).clamp(min=1e-9)
                emb = emb / norms

            all_embeddings.append(emb.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)

    def warmup(self) -> None:
        self._load()
        if self._model is not None:
            _ = self.encode(["warmup"])
            log.info("specter2_warmup_complete")


__all__ = ["SPECTER2Encoder"]
