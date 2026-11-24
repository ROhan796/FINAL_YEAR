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
            "pca", "crispr", "crispr-cas9", "pcr", "gel electrophoresis", "western blot",
            "mass spectrometry", "x-ray crystallography", "mri", "fmri",
            "deep learning", "machine learning", "transfer learning",
            "reinforcement learning", "generative adversarial network", "gan",
            "attention mechanism", "self-attention", "multi-head attention",
            "gradient descent", "backpropagation", "batch normalization",
            "graph neural network", "graph attention network", "gnn", "gat", "r-gcn",
            "contrastive learning", "specter2", "scibert", "bert", "roberta",
            "vision transformer", "vit", "diffusion model", "alphafold",
            "molecular docking", "biosensor", "field-effect transistor",
            "knowledge graph", "link prediction", "pre-training", "fine-tuning",
        ],
        "MATERIAL": [
            "graphene", "carbon nanotube", "titanium dioxide", "silicon carbide",
            "perovskite", "molybdenum disulfide", "gold nanoparticle",
            "silver nanoparticle", "quantum dot", "polymer", "hydrogel",
            "lipid nanoparticle", "mrna", "dna", "rna", "protein",
            "antibody", "enzyme", "cellulose", "chitosan", "nanomaterial",
        ],
        "DISEASE": [
            "alzheimer", "parkinson", "cancer", "diabetes", "obesity",
            "hypertension", "atherosclerosis", "asthma", "copd",
            "hiv", "covid", "covid-19", "sars-cov-2", "hepatitis",
            "glioblastoma", "melanoma", "leukemia", "lymphoma",
            "autoimmune", "inflammation", "neurodegeneration",
            "sickle cell disease", "neurodegenerative disease", "beta-thalassemia",
        ],
        "ALGORITHM": [
            "gradient descent", "adam optimizer", "sgd", "random forest",
            "xgboost", "lightgbm", "pagerank", "dijkstra", "bfs", "dfs",
            "backpropagation", "bayesian optimization", "genetic algorithm",
            "simulated annealing", "neural network", "convolution",
            "recurrent neural network", "long short-term memory",
            "distmult", "transe", "reciprocal rank fusion", "viterbi",
        ],
        "DATASET": [
            "imagenet", "mnist", "cifar", "cifar-10", "cifar-100",
            "coco", "open images", "chembl", "pubmed", "uniprot",
            "genbank", "pdb", "scierc", "arxiv", "medline",
            "snli", "squad", "glue", "superglue", "cora", "citeseer",
        ],
        "PHENOMENON": [
            "quantum entanglement", "photosynthesis", "neuroplasticity",
            "superconductivity", "catalysis", "crystallization",
            "diffusion", "osmosis", "fluorescence", "bioluminescence",
            "electromagnetic", "magnetism", "gravitational wave",
            "dark matter", "dark energy", "photon", "electron",
            "protein folding", "over-smoothing", "anisotropy",
        ],
        "METRIC": [
            "accuracy", "precision", "recall", "f1 score", "f1-score",
            "auc-roc", "auc", "rmse", "mse", "mae", "r-squared",
            "bleu score", "rouge", "perplexity", "log loss",
            "sensitivity", "specificity", "ppv", "npv", "mrr", "hits@10",
        ],
        "CHEMICAL": [
            "dopamine", "serotonin", "glucose", "atp", "nad", "nadh",
            "co2", "o2", "h2o", "ethanol", "methanol", "acetone",
            "caffeine", "aspirin", "ibuprofen", "metformin",
            "penicillin", "doxorubicin", "cisplatin", "insulin",
            "beta-amyloid", "fetal hemoglobin", "bcl11a",
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

    Architecture: SciBERT + token classification head + LinearChainCRF.
    Loads trained checkpoint `ner_head.pt` from `ml_checkpoint_dir` when present,
    combining neural CRF predictions with GazetteerNER for high recall.
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
        self._head = None
        self._tokenizer = None
        self._crf = None
        self._has_trained_checkpoint = False
        self._gazetteer = GazetteerNER()

    def _resolve_checkpoint(self):
        from pathlib import Path

        if self.checkpoint_path:
            return Path(self.checkpoint_path)
        try:
            from app.core.config import settings

            return Path(settings.ml_checkpoint_dir) / "ner_head.pt"
        except Exception:
            return Path("./models/checkpoints/ner_head.pt")

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

            from ml.ner.crf import LinearChainCRF

            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()

            hidden_size = int(ckpt.get("hidden_size", 768))
            head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, NUM_TAGS)).to(self.device)
            head.load_state_dict(ckpt["head_state_dict"])
            head.eval()
            self._head = head

            crf = LinearChainCRF(NUM_TAGS)
            crf.load_transitions(
                ckpt["crf_transitions"],
                start_transitions=ckpt.get("crf_start"),
            )
            self._crf = crf
            self._has_trained_checkpoint = True
            log.info("loaded_trained_ner_checkpoint: %s", ckpt_path)
            return True
        except Exception as exc:
            log.warning("ner_model_load_failed: %s", exc)
            return False

    def predict(self, text: str) -> NERResult:
        """Run NER on input text (trained SciBERT+CRF when checkpoint exists, else gazetteer)."""
        if not self._load():
            return self._gazetteer_predict(text)
        res = self._transformer_predict(text)
        # Supplement with high-precision gazetteer matches not already spanned
        gaz = self._gazetteer.extract(text)
        existing_lower = {e.text.lower() for e in res.entities}
        merged = list(res.entities) + [e for e in gaz if e.text.lower() not in existing_lower]
        return NERResult(entities=merged, tags=res.tags, tokens=res.tokens, model_used=res.model_used)

    def _gazetteer_predict(self, text: str) -> NERResult:
        """Gazetteer-based fallback NER."""
        entities = self._gazetteer.extract(text)
        tokens = text.split()
        tags = ["O"] * len(tokens)
        return NERResult(
            entities=entities,
            tags=tags,
            tokens=tokens,
            model_used="gazetteer",
        )

    def _transformer_predict(self, text: str) -> NERResult:
        """Trained SciBERT + token head + CRF Viterbi decode."""
        import torch

        words = text.split()
        if not words:
            return NERResult(entities=[], tags=[], tokens=[], model_used="transformer+crf")

        encoded = self._tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            hidden = self._model(**encoded).last_hidden_state[0]
            word_ids = encoded.word_ids(batch_index=0)
            word_vecs = []
            prev = None
            for pos, wid in enumerate(word_ids):
                if wid is None or wid == prev:
                    continue
                prev = wid
                word_vecs.append(hidden[pos])
            if not word_vecs:
                return NERResult(entities=[], tags=["O"] * len(words), tokens=words, model_used="transformer+crf")
            logits = self._head(torch.stack(word_vecs)).cpu().numpy()

        tag_ids = self._crf.decode(logits) if self._crf is not None else logits.argmax(axis=1).tolist()
        tags = [ID2TAG.get(t, "O") for t in tag_ids]
        entities = bio_tags_to_entities(tags, words[: len(tags)])

        return NERResult(
            entities=entities,
            tags=tags,
            tokens=words[: len(tags)],
            model_used="transformer+crf",
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
