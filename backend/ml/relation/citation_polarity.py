"""Citation Polarity Classifier (Algorithm 6.A: supports | disputes | neutral).

Classifies how a citing sentence relates to a cited paper's claim/abstract:
- supports: agrees with, extends, confirms, builds upon
- disputes: contradicts, refutes, challenges, fails to replicate
- neutral: mentions as background/baseline without taking a stance

Architecture:
- Optional PyTorch MLP head over SciBERT/embedding features `[h_cit || h_abs || h_cit*h_abs]`
- Trained checkpoint support (`citation_polarity.pt`)
- Deterministic linguistic + polarity cue fallback when checkpoint/LLM is absent
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

PolarityLabel = Literal["supports", "disputes", "neutral"]
POLARITY_CLASSES: tuple[PolarityLabel, ...] = ("supports", "disputes", "neutral")

_SUPPORT_CUES = (
    "confirm", "confirms", "confirmed", "support", "supports", "supported",
    "consistent with", "corroborate", "validates", "extend", "extends",
    "builds upon", "building on", "in line with", "outperforms using",
    "successfully reproduces", "demonstrates similar", "agrees with",
)
_DISPUTE_CUES = (
    "contradict", "contradicts", "dispute", "disputes", "refute", "refutes",
    "challenge", "challenges", "inconsistent with", "fails to", "failed to",
    "contrary to", "unlike", "overestimates", "underperforms", "flawed",
    "limitation of", "breaks down", "does not hold", "cannot replicate",
)


@dataclass(frozen=True)
class CitationPolarityResult:
    label: PolarityLabel
    confidence: float
    probabilities: dict[str, float]
    evidence: str
    model_used: str


class CitationPolarityClassifier:
    """3-class citation intent/polarity classifier (supports / disputes / neutral)."""

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._head = None
        if self.checkpoint_path and self.checkpoint_path.exists():
            self._load_checkpoint(self.checkpoint_path)

    def _load_checkpoint(self, path: Path) -> None:
        try:
            import torch
            import torch.nn as nn

            state = torch.load(path, map_location="cpu", weights_only=True)
            in_dim = int(state.get("in_dim", 768 * 3))
            head = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 3),
            )
            head.load_state_dict(state["state_dict"])
            head.eval()
            self._head = head
            log.info("citation_polarity_checkpoint_loaded: %s", path)
        except Exception as exc:  # noqa: BLE001
            log.warning("citation_polarity_load_failed: %s", exc)
            self._head = None

    def classify(self, citation_sentence: str, cited_abstract: str = "") -> CitationPolarityResult:
        """Classify a single citation sentence against the cited paper abstract."""
        sent_l = (citation_sentence or "").lower()
        sup_hits = [c for c in _SUPPORT_CUES if c in sent_l]
        dis_hits = [c for c in _DISPUTE_CUES if c in sent_l]

        # Score heuristic cues cleanly as prior probabilities
        s_score = 0.15 + 0.35 * min(2, len(sup_hits))
        d_score = 0.10 + 0.40 * min(2, len(dis_hits))
        n_score = 0.55 if (not sup_hits and not dis_hits) else 0.15

        model_used = "heuristic-cue"

        if self._head is not None:
            try:
                import torch
                from ml.serving.embed_store import embed_chunks

                embs = embed_chunks([citation_sentence or "", cited_abstract or citation_sentence or ""])
                h1 = torch.from_numpy(embs[0]).float()
                h2 = torch.from_numpy(embs[1]).float()
                feat = torch.cat([h1, h2, h1 * h2], dim=-1).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(self._head(feat), dim=-1)[0].numpy()
                # Blend trained head (0.7) with explicit lexical cue signal (0.3)
                cue = np.array([s_score, d_score, n_score], dtype=np.float32)
                cue /= cue.sum()
                blended = 0.7 * probs + 0.3 * cue
                s_score, d_score, n_score = float(blended[0]), float(blended[1]), float(blended[2])
                model_used = "scibert+head"
            except Exception as exc:  # noqa: BLE001
                log.debug("citation_polarity_head_fallback: %s", exc)

        total = s_score + d_score + n_score or 1.0
        probs_dict = {
            "supports": round(s_score / total, 4),
            "disputes": round(d_score / total, 4),
            "neutral": round(n_score / total, 4),
        }
        label = max(probs_dict, key=probs_dict.get)  # type: ignore[arg-type]
        conf = probs_dict[label]

        cues = sup_hits if label == "supports" else (dis_hits if label == "disputes" else [])
        if cues:
            evidence = f"Detected {label} cue(s): {', '.join(cues[:3])}."
        else:
            evidence = f"Classified as {label} citation context ({model_used})."

        return CitationPolarityResult(
            label=label,  # type: ignore[arg-type]
            confidence=conf,
            probabilities=probs_dict,
            evidence=evidence,
            model_used=model_used,
        )


def train_polarity_checkpoint(output_path: Path) -> dict[str, float]:
    """Train a lightweight 3-class citation polarity head on scientific citation examples."""
    import torch
    import torch.nn as nn
    from ml.serving.embed_store import embed_chunks

    samples: list[tuple[str, str, int]] = [
        ("Our experiments confirm the scaling laws reported by Kaplan et al.", "Neural scaling laws govern transformer loss.", 0),
        ("This finding supports prior observations of attention sink phenomena.", "Attention sinks stabilize streaming LLMs.", 0),
        ("We extend the Graph Attention formulation and validate their accuracy gains.", "GAT uses masked self-attention on graphs.", 0),
        ("Consistent with Vaswani et al., multi-head attention improves translation BLEU.", "Transformer replaces recurrence with attention.", 0),
        ("However, our ablation disputes the claim that deeper GCNs always help.", "Deep GCNs improve node classification.", 1),
        ("Contrary to prior work, the baseline fails to generalize under domain shift.", "Model generalizes across medical domains.", 1),
        ("We challenge this assumption and show the method underperforms linear baselines.", "Complex neural ranker beats linear models.", 1),
        ("Their reported gain cannot replicate when leakage is removed from splits.", "Dataset split yields 95% accuracy.", 1),
        ("For background on transformer architectures, we refer the reader to Vaswani et al.", "Transformer replaces recurrence with attention.", 2),
        ("We compare against standard BERT and RoBERTa baselines on GLUE.", "BERT pretrains bidirectional representations.", 2),
        ("Graph neural networks have been widely applied to molecular property prediction.", "GNNs learn node representations.", 2),
        ("Following standard evaluation protocol, we report macro F1 across five folds.", "Cross-validation measures robustness.", 2),
    ]

    texts_c = [s[0] for s in samples]
    texts_a = [s[1] for s in samples]
    y = torch.tensor([s[2] for s in samples], dtype=torch.long)

    emb_c = torch.from_numpy(embed_chunks(texts_c)).float()
    emb_a = torch.from_numpy(embed_chunks(texts_a)).float()
    x = torch.cat([emb_c, emb_a, emb_c * emb_a], dim=-1)

    head = nn.Sequential(
        nn.Linear(x.size(1), 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 3),
    )
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    head.train()
    final_loss = 0.0
    for _ in range(80):
        opt.zero_grad()
        logits = head(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        final_loss = float(loss.item())

    head.eval()
    with torch.no_grad():
        acc = float((head(x).argmax(dim=-1) == y).float().mean().item())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"in_dim": int(x.size(1)), "state_dict": head.state_dict()}, output_path)
    return {"loss": round(final_loss, 4), "train_acc": round(acc, 4)}


__all__ = [
    "POLARITY_CLASSES",
    "CitationPolarityClassifier",
    "CitationPolarityResult",
    "PolarityLabel",
    "train_polarity_checkpoint",
]
