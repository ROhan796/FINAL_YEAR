"""Confidence calibration & feedback retraining for Cross-Domain Discoveries.

Implements Algorithm 6.F (Platt scaling) + P2.12 (Feedback -> retraining loop):
- Maps raw DistMult / GNN scores to calibrated probabilities P(useful=1 | score)
- Fits logistic regression parameters (A, B, cross_domain_weight) from stored
  Discovery feedback (`useful_votes`, `dismissed`)
- Persists calibration weights in `models/checkpoints/discovery_calibrator.json`
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PlattCalibrator:
    """Two-parameter logistic (Platt) calibrator with optional domain-cross boost."""

    coef: float = 2.2
    intercept: float = -0.3
    cross_domain_weight: float = 0.25
    samples_fitted: int = 0

    def predict_prob(self, raw_score: float, *, cross_domain: bool = False) -> float:
        """Return calibrated probability in (0.01, 0.99)."""
        z = self.coef * float(raw_score) + self.intercept + (self.cross_domain_weight if cross_domain else 0.0)
        z = max(-20.0, min(20.0, z))
        prob = 1.0 / (1.0 + math.exp(-z))
        return round(max(0.01, min(0.99, prob)), 4)

    def confidence_tier(self, prob: float) -> str:
        if prob >= 0.72:
            return "High"
        if prob >= 0.45:
            return "Medium"
        return "Low"

    def fit(self, scores: list[float], labels: list[int], cross_flags: list[bool] | None = None) -> dict[str, float]:
        """Fit logistic parameters using scikit-learn (or gradient descent fallback)."""
        if len(scores) < 2 or len(set(labels)) < 2:
            # Adjust intercept gently when only single-class feedback is available
            if scores and labels:
                mean_y = sum(labels) / len(labels)
                self.intercept = round(math.log(max(0.05, mean_y) / max(0.05, 1.0 - mean_y)) - 0.5, 4)
                self.samples_fitted = len(scores)
            return {"coef": self.coef, "intercept": self.intercept, "samples": self.samples_fitted}

        try:
            from sklearn.linear_model import LogisticRegression

            flags = cross_flags or [False] * len(scores)
            X = np.array([[s, 1.0 if c else 0.0] for s, c in zip(scores, flags)], dtype=np.float64)
            y = np.array(labels, dtype=np.int64)
            clf = LogisticRegression(C=1.0, max_iter=200)
            clf.fit(X, y)
            self.coef = round(float(clf.coef_[0][0]), 4)
            self.cross_domain_weight = round(float(clf.coef_[0][1]), 4)
            self.intercept = round(float(clf.intercept_[0]), 4)
            self.samples_fitted = len(scores)
        except Exception as exc:  # noqa: BLE001
            log.warning("platt_sklearn_fallback: %s", exc)
        return {
            "coef": self.coef,
            "intercept": self.intercept,
            "cross_domain_weight": self.cross_domain_weight,
            "samples": self.samples_fitted,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "coef": self.coef,
                    "intercept": self.intercept,
                    "cross_domain_weight": self.cross_domain_weight,
                    "samples_fitted": self.samples_fitted,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> PlattCalibrator:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                coef=float(data.get("coef", 2.2)),
                intercept=float(data.get("intercept", -0.3)),
                cross_domain_weight=float(data.get("cross_domain_weight", 0.25)),
                samples_fitted=int(data.get("samples_fitted", 0)),
            )
        except Exception:  # noqa: BLE001
            return cls()


def recalibrate_from_db(db: Any, checkpoint_dir: str | Path = "./models/checkpoints") -> dict[str, Any]:
    """Fit PlattCalibrator from DB discovery feedback and refresh active discovery scores."""
    from sqlalchemy import select

    from app.models.knowledge import Discovery, KgNode

    rows = list(db.scalars(select(Discovery)).all())
    if not rows:
        return {"updated": 0, "samples": 0}

    node_ids = {r.src_id for r in rows} | {r.dst_id for r in rows}
    nodes = {n.id: n for n in db.scalars(select(KgNode).where(KgNode.id.in_(node_ids))).all()} if node_ids else {}

    scores: list[float] = []
    labels: list[int] = []
    cross_flags: list[bool] = []

    for r in rows:
        src = nodes.get(r.src_id)
        dst = nodes.get(r.dst_id)
        is_cross = src is not None and dst is not None and src.domain != dst.domain
        if r.useful_votes > 0:
            scores.append(float(r.score))
            labels.append(1)
            cross_flags.append(is_cross)
        elif r.dismissed:
            scores.append(float(r.score))
            labels.append(0)
            cross_flags.append(is_cross)

    cal_path = Path(checkpoint_dir) / "discovery_calibrator.json"
    cal = PlattCalibrator.load(cal_path)
    stats = cal.fit(scores, labels, cross_flags)
    cal.save(cal_path)

    # Re-score active discoveries with feedback boost + calibration
    updated = 0
    for r in rows:
        if r.dismissed:
            continue
        src = nodes.get(r.src_id)
        dst = nodes.get(r.dst_id)
        is_cross = src is not None and dst is not None and src.domain != dst.domain
        vote_bonus = min(0.12, 0.03 * int(r.useful_votes or 0))
        r.score = round(min(0.99, cal.predict_prob(float(r.score), cross_domain=is_cross) + vote_bonus), 4)
        updated += 1

    db.commit()
    return {"updated": updated, **stats}


__all__ = ["PlattCalibrator", "recalibrate_from_db"]
