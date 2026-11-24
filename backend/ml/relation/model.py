"""Relation Extraction model — Entity Pair MLP Classifier.

Architecture per spec:
- Entity span representation: MeanPool over entity tokens
- Pair representation: [h_e1 || h_e2 || h_e1⊙h_e2 || h_cls]
- Classifier: softmax(W_RE · h_pair + b_RE) over 9 classes

TRAINING STATUS: Architecture defined. No trained checkpoint included.
Fallback uses co-occurrence + type constraint scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ml.relation.types import (
    RELATION_LABELS,
    NUM_RELATIONS,
    REL2ID,
    ID2REL,
    ExtractedRelation,
    are_type_compatible,
)

log = logging.getLogger(__name__)


@dataclass
class REResult:
    """Result from RE pipeline."""
    relations: list[ExtractedRelation]
    model_used: str  # "transformer" or "cooccurrence"


class CooccurrenceRE:
    """Co-occurrence based relation extraction fallback.

    When the transformer model is unavailable, this extracts relations
    based on entity co-occurrence in the same sentence with type constraints.

    This is a DEVELOPMENT / PLACEHOLDER, not a trained model.
    """

    def predict(self, sentence: str, entities: list[dict]) -> list[ExtractedRelation]:
        """Extract relations from entity pairs via co-occurrence heuristic."""
        relations: list[ExtractedRelation] = []

        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i >= j:
                    continue
                # Try all relation types
                for rel in RELATION_LABELS:
                    if rel == "no_relation":
                        continue
                    if not are_type_compatible(
                        e1.get("label", ""), e2.get("label", ""), rel
                    ):
                        continue
                    # Heuristic: if both entities are in the same sentence,
                    # assign a moderate confidence relation
                    confidence = 0.5
                    # Boost if entities are close together
                    if abs(e1.get("start", 0) - e2.get("start", 0)) < 50:
                        confidence = 0.7
                    relations.append(ExtractedRelation(
                        head=e1["text"],
                        tail=e2["text"],
                        relation=rel,
                        confidence=confidence,
                        head_type=e1.get("label", ""),
                        tail_type=e2.get("label", ""),
                    ))

        # Keep only highest confidence relation per pair
        best: dict[tuple[str, str], ExtractedRelation] = {}
        for rel in relations:
            key = (rel.head, rel.tail)
            if key not in best or rel.confidence > best[key].confidence:
                best[key] = rel
        return list(best.values())


class TransformerRE:
    """Transformer-based Relation Extraction.

    Architecture:
    - SciBERT backbone for context encoding
    - Entity span pooling: MeanPool over entity tokens
    - Pair representation: [h_e1 || h_e2 || h_e1⊙h_e2 || h_cls]
    - MLP classifier over 9 relation types

    TRAINING STATUS: Architecture defined. Without a fine-tuned checkpoint,
    uses co-occurrence fallback.
    """

    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        device: str = "cpu",
        checkpoint_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.checkpoint_path = checkpoint_path
        self._model = None
        self._clf = None
        self._tokenizer = None
        self._has_trained_checkpoint = False
        self._fallback = CooccurrenceRE()

    def _resolve_checkpoint(self):
        from pathlib import Path

        if self.checkpoint_path:
            return Path(self.checkpoint_path)
        try:
            from app.core.config import settings

            return Path(settings.ml_checkpoint_dir) / "re_head.pt"
        except Exception:
            return Path("./models/checkpoints/re_head.pt")

    def _load(self) -> bool:
        if self._model is not None and self._has_trained_checkpoint:
            return True
        ckpt_path = self._resolve_checkpoint()
        if not ckpt_path.exists():
            return False
        try:
            import torch
            import torch.nn as nn
            from transformers import AutoModel, AutoTokenizer

            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()

            pair_dim = int(ckpt.get("pair_dim", 768 * 4))
            clf = nn.Sequential(
                nn.Linear(pair_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(256, NUM_RELATIONS),
            ).to(self.device)
            clf.load_state_dict(ckpt["state_dict"])
            clf.eval()
            self._clf = clf
            self._has_trained_checkpoint = True
            log.info("loaded_trained_re_checkpoint: %s", ckpt_path)
            return True
        except Exception as exc:
            log.warning("re_model_load_failed: %s", exc)
            return False

    def predict(self, sentence: str, entities: list[dict]) -> REResult:
        """Extract relations from entity pairs in a sentence."""
        if len(entities) < 2:
            return REResult(relations=[], model_used="none")
        if not self._load():
            return REResult(
                relations=self._fallback.predict(sentence, entities),
                model_used="cooccurrence",
            )
        return self._transformer_predict(sentence, entities)

    def _transformer_predict(self, sentence: str, entities: list[dict]) -> REResult:
        """Trained SciBERT + pair MLP classifier with type-constraint masking."""
        import torch

        encoded = self._tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            hidden = self._model(**encoded).last_hidden_state[0]
            cls_embedding = hidden[0]
            seq_tokens = self._tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

        def _span_vec(span_text: str):
            toks = set(self._tokenizer.tokenize(span_text.lower()))
            idxs = [k for k, t in enumerate(seq_tokens) if t.lstrip("#").lower() in toks]
            if idxs:
                return hidden[idxs].mean(dim=0)
            return hidden[1:-1].mean(dim=0) if hidden.size(0) > 2 else cls_embedding

        relations: list[ExtractedRelation] = []
        for i, e1 in enumerate(entities):
            h_e1 = _span_vec(e1["text"])
            for j, e2 in enumerate(entities):
                if i >= j or e1["text"].lower() == e2["text"].lower():
                    continue
                h_e2 = _span_vec(e2["text"])
                pair_repr = torch.cat([h_e1, h_e2, h_e1 * h_e2, cls_embedding], dim=-1).unsqueeze(0)
                with torch.no_grad():
                    logits = self._clf(pair_repr)[0].clone()

                # Apply type-constraint mask before softmax
                for rel_idx, rel_name in enumerate(RELATION_LABELS):
                    if rel_name == "no_relation":
                        continue
                    if not are_type_compatible(e1.get("label", ""), e2.get("label", ""), rel_name):
                        logits[rel_idx] = -1e9

                probs = torch.softmax(logits, dim=-1)
                best_idx = int(torch.argmax(probs).item())
                rel_name = ID2REL.get(best_idx, "no_relation")
                conf = float(probs[best_idx].item())
                if rel_name != "no_relation" and conf >= 0.25:
                    relations.append(
                        ExtractedRelation(
                            head=e1["text"],
                            tail=e2["text"],
                            relation=rel_name,
                            confidence=round(conf, 4),
                            head_type=e1.get("label", ""),
                            tail_type=e2.get("label", ""),
                        )
                    )

        if not relations:
            relations = self._fallback.predict(sentence, entities)
            return REResult(relations=relations, model_used="transformer+cooccurrence")

        return REResult(relations=relations, model_used="transformer+mlp")


class REPipeline:
    """High-level RE pipeline.

    Combines entity pair extraction and relation classification.
    """

    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased",
                 device: str = "cpu") -> None:
        self._re = TransformerRE(model_name=model_name, device=device)

    def extract_relations(self, sentence: str, entities: list[dict]) -> list[ExtractedRelation]:
        """Extract relations between entities in a sentence."""
        result = self._re.predict(sentence, entities)
        return result.relations

    def extract_from_document(self, sentences: list[str],
                              entities_per_sentence: list[list[dict]]) -> list[ExtractedRelation]:
        """Extract relations from multiple sentences."""
        all_relations: list[ExtractedRelation] = []
        for sent, ents in zip(sentences, entities_per_sentence):
            all_relations.extend(self.extract_relations(sent, ents))
        return all_relations

    def warmup(self) -> None:
        self._re._load()


__all__ = ["REPipeline", "REResult", "CooccurrenceRE", "TransformerRE"]
