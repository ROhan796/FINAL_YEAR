"""Discovery ranker — cross-domain link prediction and calibrated ranking.

Candidate generation bounded by:
1. Type filtering (`are_type_compatible`)
2. Domain filtering (only cross-domain pairs for discovery)
3. ANN/cosine pre-filtering (top-200 neighbors per node)
4. Platt confidence calibration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ml.discovery.calibration import PlattCalibrator
from ml.discovery.scorers import get_scorer
from ml.relation.types import are_type_compatible

log = logging.getLogger(__name__)


@dataclass
class Discovery:
    """A single predicted discovery."""

    src_id: str
    dst_id: str
    src_name: str
    dst_name: str
    relation: str
    score: float
    src_domain: str = ""
    dst_domain: str = ""
    cross_domain: bool = True
    evidence_paths: list[list[str]] = field(default_factory=list)
    evidence_paper_ids: list[str] = field(default_factory=list)
    scorer: str = "distmult"
    confidence_tier: str = "Medium"


@dataclass
class DiscoveryResult:
    """Result of discovery ranking."""

    discoveries: list[Discovery]
    total_candidates: int = 0
    scorer_used: str = "distmult"


class DiscoveryRanker:
    """Rank potential cross-domain discoveries with DistMult + Platt calibration."""

    def __init__(
        self,
        scorer_name: str = "distmult",
        cross_domain_boost: float = 1.3,
        top_k: int = 100,
        candidate_top_k: int = 200,
        dim: int = 256,
        calibrator: PlattCalibrator | None = None,
    ) -> None:
        self.scorer_name = scorer_name
        self.cross_domain_boost = cross_domain_boost
        self.top_k = top_k
        self.candidate_top_k = candidate_top_k
        self.scorer = get_scorer(scorer_name, dim)
        self.relation_embeddings: dict[str, np.ndarray] = {}
        self.calibrator = calibrator or self._load_default_calibrator()

    @staticmethod
    def _load_default_calibrator() -> PlattCalibrator:
        try:
            from app.core.config import settings

            return PlattCalibrator.load(Path(settings.ml_checkpoint_dir) / "discovery_calibrator.json")
        except Exception:
            return PlattCalibrator()

    def set_relation_embeddings(self, embeddings: dict[str, np.ndarray]) -> None:
        self.relation_embeddings = embeddings

    def discover(
        self,
        node_embeddings: np.ndarray,
        node_ids: list[str],
        node_domains: list[str],
        node_names: list[str],
        existing_edges: set[tuple[str, str, str]] | None = None,
        relation_labels: list[str] | None = None,
        node_types: list[str] | None = None,
        node_paper_ids: dict[str, str] | None = None,
    ) -> DiscoveryResult:
        N = len(node_ids)
        if N < 2:
            return DiscoveryResult(discoveries=[], total_candidates=0, scorer_used=self.scorer_name)

        existing = existing_edges or set()
        relations = relation_labels or ["used_in", "applied_to", "improves", "treats"]
        dim = node_embeddings.shape[1]

        # Normalize embeddings for cosine ANN pre-filter
        norms = np.linalg.norm(node_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = node_embeddings / norms
        sim_matrix = normed @ normed.T

        candidates: list[Discovery] = []
        seen_pairs: set[tuple[str, str, str]] = set()
        total_candidates = 0

        for rel_idx, rel in enumerate(relations):
            rel_emb = self.relation_embeddings.get(rel)
            if rel_emb is None or rel_emb.shape[0] != dim:
                rng = np.random.default_rng(100 + rel_idx)
                rel_emb = (rng.standard_normal(dim).astype(np.float32) * 0.25) + 0.5

            for i in range(N):
                # Top-K ANN neighbors instead of full O(N^2)
                neighbor_order = np.argsort(sim_matrix[i])[::-1][: min(N, self.candidate_top_k)]
                for j in neighbor_order:
                    j = int(j)
                    if i == j:
                        continue
                    if node_domains[i] and node_domains[j] and node_domains[i] == node_domains[j]:
                        continue
                    if node_types and not are_type_compatible(node_types[i], node_types[j], rel):
                        continue
                    if (node_ids[i], rel, node_ids[j]) in existing:
                        continue
                    if (node_ids[i], rel, node_ids[j]) in seen_pairs:
                        continue

                    total_candidates += 1
                    raw_score = float(
                        self.scorer.score(
                            node_embeddings[i : i + 1],
                            node_embeddings[j : j + 1],
                            rel_emb.reshape(1, -1),
                        )[0]
                    )
                    is_cross = node_domains[i] != node_domains[j]
                    if is_cross:
                        raw_score *= self.cross_domain_boost

                    cal_score = self.calibrator.predict_prob(raw_score, cross_domain=is_cross)
                    seen_pairs.add((node_ids[i], rel, node_ids[j]))

                    papers: list[str] = []
                    if node_paper_ids:
                        for nid in (node_ids[i], node_ids[j]):
                            pid = node_paper_ids.get(nid)
                            if pid and pid not in papers:
                                papers.append(pid)

                    candidates.append(
                        Discovery(
                            src_id=node_ids[i],
                            dst_id=node_ids[j],
                            src_name=node_names[i],
                            dst_name=node_names[j],
                            relation=rel,
                            score=cal_score,
                            src_domain=node_domains[i],
                            dst_domain=node_domains[j],
                            cross_domain=is_cross,
                            evidence_paths=[[node_names[i], rel, node_names[j]]],
                            evidence_paper_ids=papers,
                            scorer=self.scorer_name,
                            confidence_tier=self.calibrator.confidence_tier(cal_score),
                        )
                    )

        candidates.sort(key=lambda d: d.score, reverse=True)
        top = candidates[: self.top_k]

        log.info(
            "discovery_complete: total_candidates=%s top_k=%s scorer=%s",
            total_candidates,
            len(top),
            self.scorer_name,
        )
        return DiscoveryResult(
            discoveries=top,
            total_candidates=total_candidates,
            scorer_used=self.scorer_name,
        )


__all__ = ["DiscoveryRanker", "Discovery", "DiscoveryResult"]
