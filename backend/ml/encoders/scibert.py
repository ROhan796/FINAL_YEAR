"""SciBERT encoder — scientific text understanding.

Uses allenai/scibert_scivocab_uncased for:
- scientific language understanding
- passage/entity/query representations
- NER and RE backbone

When torch/transformers is unavailable, falls back to HashEncoder.
"""

from __future__ import annotations

import logging

import numpy as np

from ml.encoders import EncodeKind, EncoderBase, HashEncoder
from ml.encoders.registry import register_encoder

log = logging.getLogger(__name__)


@register_encoder("scibert")
class SciBERTEncoder(EncoderBase):
    """SciBERT-based encoder for scientific text.

    Pooling strategies by kind:
    - paper: CLS token (captures full-document meaning)
    - passage: mean pooling over non-special tokens
    - entity: CLS token (entity-level representation)
    - query: CLS token (search query representation)
    """

    def __init__(self, dim: int = 768, device: str = "cpu") -> None:
        self.dim = dim
        self.name = "scibert"
        self._device = device
        self._model = None
        self._tokenizer = None
        self._fallback: HashEncoder | None = None

    def _load(self) -> bool:
        """Try to load the SciBERT model. Returns True on success."""
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            model_name = "allenai/scibert_scivocab_uncased"
            log.info("loading_scibert: model=%s device=%s", model_name, self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name).to(self._device)
            self._model.eval()
            return True
        except Exception as exc:
            log.warning("scibert_load_failed: %s", exc)
            self._fallback = HashEncoder(dim=self.dim, name="scibert-fallback")
            return False

    def encode(self, texts: list[str], *, kind: EncodeKind = "passage") -> np.ndarray:
        if not self._load():
            return self._fallback.encode(texts, kind=kind)

        import torch

        all_embeddings = []
        batch_size = 32
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
                hidden = outputs.last_hidden_state  # (batch, seq_len, 768)
                attention_mask = encoded["attention_mask"]

                if kind in ("paper", "entity", "query"):
                    # CLS token pooling
                    emb = hidden[:, 0, :]
                else:
                    # Mean pooling over non-padding tokens
                    mask_expanded = attention_mask.unsqueeze(-1).float()
                    sum_hidden = (hidden * mask_expanded).sum(dim=1)
                    counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
                    emb = sum_hidden / counts

                # L2 normalize
                norms = torch.norm(emb, p=2, dim=1, keepdim=True).clamp(min=1e-9)
                emb = emb / norms

            all_embeddings.append(emb.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)

    def warmup(self) -> None:
        self._load()
        if self._model is not None:
            _ = self.encode(["warmup encoding check"])
            log.info("scibert_warmup_complete")


__all__ = ["SciBERTEncoder"]
