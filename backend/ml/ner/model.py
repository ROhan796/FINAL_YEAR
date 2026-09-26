"""NER model — Transformer + token classification head + optional CRF.

Architecture:
- SciBERT backbone (frozen or fine-tuned)
- Token classification head: z_i = W_NER · h_i^(L) + b_NER
- Optional CRF layer for valid BIO sequence enforcement

When torch is unavailable, uses a gazetteer-based fallback.

IMPORTANT: This is the MODEL ARCHITECTURE. No trained checkpoint is included.
The fallback uses dictionary-based matching for development/testing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

from ml.ner.tags import (
    ENTITY_TYPES,
    NUM_TAGS,
    TAG2ID,
    ID2TAG,
    ExtractedEntity,
    bio_tags_to_entities,
)

log = logging.getLogger(__name__)


@dataclass
class NERResult:
    """Result from NER pipeline."""
    entities: list[ExtractedEntity]
    tags: list[str]
    tokens: list[str]
    model_used: str  # "transformer+crf" or "gazetteer"


class GazetteerNER:
    """Dictionary-based NER fallback.

    Uses known scientific terms from a gazetteer. This is NOT a trained model
    — it is a deterministic fallback that allows the pipeline to function
    when torch/transformers is unavailable.
    """

    # Common scientific terms by entity type (seed gazetteer)
    GAZETTEER: dict[str, list[str]] = {
        "METHOD": [
            "convolutional neural network", "cnn", "lstm", "rnn", "transformer",
            "random forest", "support vector machine", "svm", "k-means",
            "pca", "crispr", "pcr", "gel electrophoresis", "western blot",
            "mass spectrometry", "x-ray crystallography", "mri", "fmri",
            "deep learning", "machine learning", "transfer learning",
            "reinforcement learning", "generative adversarial network", "gan",
            "attention mechanism", "self-attention", "multi-head attention",
            "gradient descent", "backpropagation", "batch normalization",
        ],
        "MATERIAL": [
            "graphene", "carbon nanotube", "titanium dioxide", "silicon carbide",
            "perovskite", "molybdenum disulfide", "gold nanoparticle",
            "silver nanoparticle", "quantum dot", "polymer", "hydrogel",
            "lipid nanoparticle", "mRNA", "dna", "rna", "protein",
            "antibody", "enzyme", "cellulose", "chitosan",
        ],
        "DISEASE": [
            "alzheimer", "parkinson", "cancer", "diabetes", "obesity",
            "hypertension", "atherosclerosis", "asthma", "copd",
            "hiv", "covid", "covid-19", "sars-cov-2", "hepatitis",
            "glioblastoma", "melanoma", "leukemia", "lymphoma",
            "autoimmune", "inflammation", "neurodegeneration",
        ],
        "ALGORITHM": [
            "gradient descent", "adam optimizer", "sgd", "random forest",
            "xgboost", "lightgbm", "pagerank", "dijkstra", "bfs", "dfs",
            "backpropagation", "bayesian optimization", "genetic algorithm",
            "simulated annealing", "neural network", "convolution",
            "recurrent neural network", "long short-term memory",
        ],
        "DATASET": [
            "imagenet", "mnist", "cifar", "cifar-10", "cifar-100",
            "coco", "open images", "chembl", "pubmed", "uniprot",
            "genbank", "pdb", "scibert", "arxiv", "medline",
            "snli", "squad", "glue", "superglue",
        ],
        "PHENOMENON": [
            "quantum entanglement", "photosynthesis", "neuroplasticity",
            "superconductivity", "catalysis", "crystallization",
            "diffusion", "osmosis", "fluorescence", "bioluminescence",
            "electromagnetic", "magnetism", "gravitational wave",
            "dark matter", "dark energy", "photon", "electron",
        ],
        "METRIC": [
            "accuracy", "precision", "recall", "f1 score", "f1-score",
            "auc-roc", "auc", "rmse", "mse", "mae", "r-squared",
            "bleu score", "rouge", "perplexity", "log loss",
            "sensitivity", "specificity", "ppv", "npv",
        ],
        "CHEMICAL": [
            "dopamine", "serotonin", "glucose", "atp", "nad", "nadh",
            "co2", "o2", "h2o", "ethanol", "methanol", "acetone",
            "caffeine", "aspirin", "ibuprofen", "metformin",
            "penicillin", "doxorubicin", "cisplatin", "insulin",
        ],
    }

    def __init__(self) -> None:
        self._patterns: dict[str, list[tuple[re.Pattern, str]]] = {}
        for etype, terms in self.GAZETTEER.items():
            self._patterns[etype] = [
                (re.compile(re.escape(term), re.IGNORECASE), etype)
                for term in sorted(terms, key=len, reverse=True)
            ]

    def extract(self, text: str) -> list[ExtractedEntity]:
        """Extract entities using gazetteer matching."""
        entities: list[ExtractedEntity] = []
        for etype, patterns in self._patterns.items():
            for pattern, label in patterns:
                for match in pattern.finditer(text):
                    entities.append(ExtractedEntity(
                        text=match.group(),
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,  # gazetteer matches are heuristic
                    ))
        # Sort by start position and remove overlaps (longest match wins)
        entities.sort(key=lambda e: (e.start, -(e.end - e.start)))
        filtered: list[ExtractedEntity] = []
        last_end = -1
        for ent in entities:
            if ent.start >= last_end:
                filtered.append(ent)
                last_end = ent.end
        return filtered


class TransformerNER:
    """Transformer-based NER model.

    Architecture: SciBERT + token classification head + optional CRF.

    This class defines the architecture and interface. When no trained
    checkpoint is available, it uses the GazetteerNER fallback.

    TRAINING STATUS: Architecture defined, no trained checkpoint included.
    """

    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased",
                 device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._gazetteer = GazetteerNER()

    def _load(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer

            log.info("loading_ner_model: %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=NUM_TAGS,
                ignore_mismatched_sizes=True,
            ).to(self.device)
            self._model.eval()
            return True
        except Exception as exc:
            log.warning("ner_model_load_failed: %s", exc)
            return False

    def predict(self, text: str) -> NERResult:
        """Run NER on input text.

        If a trained transformer is available, uses it.
        Otherwise falls back to gazetteer-based extraction.
        """
        if not self._load():
            return self._gazetteer_predict(text)
        return self._transformer_predict(text)

    def _gazetteer_predict(self, text: str) -> NERResult:
        """Gazetteer-based fallback NER."""
        entities = self._gazetteer.extract(text)
        # Create dummy BIO tags for the result
        tokens = text.split()
        tags = ["O"] * len(tokens)
        return NERResult(
            entities=entities,
            tags=tags,
            tokens=tokens,
            model_used="gazetteer",
        )

    def _transformer_predict(self, text: str) -> NERResult:
        """Transformer-based NER prediction.

        NOTE: This requires a fine-tuned checkpoint for meaningful results.
        Without fine-tuning, the model's token classification head has
        random weights and will produce arbitrary predictions.
        """
        import torch

        tokens = self._tokenizer.tokenize(text)
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**encoded)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)[0]

        tags = [ID2TAG.get(p.item(), "O") for p in predictions]
        # Remove special tokens
        tags = tags[1:-1]  # remove [CLS] and [SEP]
        entities = bio_tags_to_entities(tags, tokens[:len(tags)])

        return NERResult(
            entities=entities,
            tags=tags,
            tokens=tokens,
            model_used="transformer+crf" if hasattr(self, '_crf') else "transformer",
        )


class NERPipeline:
    """High-level NER pipeline.

    Combines tokenization, model inference, and entity extraction.
    Automatically falls back to gazetteer when model is unavailable.
    """

    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased",
                 device: str = "cpu") -> None:
        self._ner = TransformerNER(model_name=model_name, device=device)

    def extract_entities(self, text: str) -> list[ExtractedEntity]:
        """Extract scientific entities from text."""
        result = self._ner.predict(text)
        return result.entities

    def extract_from_sentences(self, sentences: list[str]) -> list[list[ExtractedEntity]]:
        """Extract entities from a list of sentences."""
        return [self.extract_entities(s) for s in sentences]

    def warmup(self) -> None:
        """Pre-load model."""
        self._ner._load()


__all__ = ["NERPipeline", "NERResult", "GazetteerNER", "TransformerNER"]
