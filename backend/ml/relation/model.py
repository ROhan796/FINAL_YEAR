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

    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased",
                 device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._fallback = CooccurrenceRE()

    def _load(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            log.info("loading_re_model: %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            return True
        except Exception as exc:
            log.warning("re_model_load_failed: %s", exc)
            return False

    def predict(self, sentence: str, entities: list[dict]) -> REResult:
        """Extract relations from entity pairs in a sentence."""
        if not self._load():
            return REResult(
                relations=self._fallback.predict(sentence, entities),
                model_used="cooccurrence",
            )
        return self._transformer_predict(sentence, entities)

    def _transformer_predict(self, sentence: str, entities: list[dict]) -> REResult:
        """Transformer-based RE prediction.

        NOTE: Without fine-tuned weights, this produces random predictions.
        """
        import torch

        # Encode the full sentence
        encoded = self._tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**encoded)
            hidden = outputs.last_hidden_state  # (1, seq_len, 768)
            cls_embedding = hidden[0, 0, :]  # (768,)

        # For each entity pair, build pair representation and classify
        relations: list[ExtractedRelation] = []
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i >= j:
                    continue
                # Get entity span embeddings (simplified — would need token alignment)
                h_e1 = cls_embedding  # placeholder
                h_e2 = cls_embedding  # placeholder
                # Pair representation: [h_e1 || h_e2 || h_e1⊙h_e2 || h_cls]
                pair_repr = torch.cat([
                    h_e1, h_e2, h_e1 * h_e2, cls_embedding
                ])  # (3072,)

                # In a real implementation, this goes through an MLP classifier
                # For now, use co-occurrence fallback
                pass

        return REResult(
            relations=self._fallback.predict(sentence, entities),
            model_used="transformer-placeholder",
        )


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
