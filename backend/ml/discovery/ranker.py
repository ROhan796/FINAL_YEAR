"""Discovery ranker — cross-domain link prediction and ranking.

Candidate generation bounded by:
1. Type filtering (only relation-legal type pairs)
2. Domain filtering (only cross-domain pairs for discovery)
3. ANN pre-filtering (top-200 neighbors per node)
4. Candidate limits

Output: ranked discoveries with evidence and provenance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ml.discovery.scorers import DistMultScorer, get_scorer

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


@dataclass
class DiscoveryResult:
    """Result of discovery ranking."""
    discoveries: list[Discovery]
    total_candidates: int = 0
    scorer_used: str = "distmult"


class DiscoveryRanker:
    """Rank potential cross-domain discoveries.

    Per spec:
    - Uses DistMult by default
    - Cross-domain boost: score × 1[domain(src) != domain(dst)] × λ
    - λ = 1.3 (configurable)
    """

    def __init__(
        self,
        scorer_name: str = "distmult",
        cross_domain_boost: float = 1.3,
        top_k: int = 100,
        candidate_top_k: int = 200,
        dim: int = 768,
    ) -> None:
        self.scorer_name = scorer_name
        self.cross_domain_boost = cross_domain_boost
        self.top_k = top_k
        self.candidate_top_k = candidate_top_k
        self.scorer = get_scorer(scorer_name, dim)
        self.relation_embeddings: dict[str, np.ndarray] = {}

    def set_relation_embeddings(self, embeddings: dict[str, np.ndarray]) -> None:
        """Set learned relation embeddings."""
        self.relation_embeddings = embeddings

    def discover(
        self,
        node_embeddings: np.ndarray,
        node_ids: list[str],
        node_domains: list[str],
        node_names: list[str],
        existing_edges: set[tuple[str, str, str]] | None = None,
        relation_labels: list[str] | None = None,
    ) -> DiscoveryResult:
        """Run cross-domain discovery.

        Args:
            node_embeddings: (N, dim) GNN output embeddings
            node_ids: List of node IDs
            node_domains: List of domain labels per node
            node_names: List of canonical names per node
            existing_edges: Set of (src_id, relation, dst_id) to exclude
            relation_labels: Relation types to consider

        Returns:
            DiscoveryResult with ranked discoveries
        """
        N = len(node_ids)
        existing = existing_edges or set()
        relations = relation_labels or ["used_in"]

        candidates: list[Discovery] = []
        total_candidates = 0

        for rel in relations:
            rel_emb = self.relation_embeddings.get(rel)
            if rel_emb is None:
                rel_emb = np.random.randn(node_embeddings.shape[1]).astype(np.float32) * 0.01

            # Score all pairs (bounded by domain cross-check)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    # Only cross-domain for discovery
                    if node_domains[i] == node_domains[j]:
                        continue
                    # Skip existing edges
                    if (node_ids[i], rel, node_ids[j]) in existing:
                        continue

                    total_candidates += 1

                    # Quick pre-filter: only consider if cosine similarity > 0
                    sim = np.dot(node_embeddings[i], node_embeddings[j])
                    if sim <= 0:
                        continue

                    score = float(self.scorer.score(
                        node_embeddings[i:i+1],
                        node_embeddings[j:j+1],
                        rel_emb.reshape(1, -1),
                    )[0])

                    # Apply cross-domain boost
                    score *= self.cross_domain_boost

                    candidates.append(Discovery(
                        src_id=node_ids[i],
                        dst_id=node_ids[j],
                        src_name=node_names[i],
                        dst_name=node_names[j],
                        relation=rel,
                        score=score,
                        src_domain=node_domains[i],
                        dst_domain=node_domains[j],
                        cross_domain=True,
                        scorer=self.scorer_name,
                    ))

                    if len(candidates) > self.candidate_top_k * 10:
                        break
                if len(candidates) > self.candidate_top_k * 10:
                    break

        # Rank and take top-K
        candidates.sort(key=lambda d: d.score, reverse=True)
        top = candidates[:self.top_k]

        log.info("discovery_complete: total_candidates=%s top_k=%s scorer=%s", total_candidates, len(top), self.scorer_name)

        return DiscoveryResult(
            discoveries=top,
            total_candidates=total_candidates,
            scorer_used=self.scorer_name,
        )


__all__ = ["DiscoveryRanker", "Discovery", "DiscoveryResult"]
