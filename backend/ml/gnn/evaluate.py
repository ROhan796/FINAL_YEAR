"""GNN evaluation metrics.

Per spec: Hits@K, MRR, AUC-ROC, P@K
Targets: Hits@10 > 0.85, MRR > 0.4
"""

from __future__ import annotations

import numpy as np


def hits_at_k(ranks: np.ndarray, k: int = 10) -> float:
    """Fraction of true links in top-K predictions."""
    return float(np.mean(ranks <= k))


def mean_reciprocal_rank(ranks: np.ndarray) -> float:
    """Mean Reciprocal Rank."""
    return float(np.mean(1.0 / ranks))


def precision_at_k(ranks: np.ndarray, k: int = 10) -> float:
    """Precision at K."""
    return float(np.mean(ranks <= k))


def evaluate_link_prediction(
    embeddings: np.ndarray,
    true_edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    num_nodes: int,
) -> dict[str, float]:
    """Full evaluation of link prediction quality.

    Args:
        embeddings: (N, dim) node embeddings
        true_edges: (src_ids, dst_ids, edge_types)
        num_nodes: Total number of nodes

    Returns:
        Dict with Hits@1, Hits@3, Hits@10, MRR, P@10
    """
    src, dst, types = true_edges
    ranks = np.zeros(len(src), dtype=np.float32)

    for i in range(len(src)):
        src_emb = embeddings[src[i]]
        scores = embeddings @ src_emb
        # Exclude true edge from ranking
        scores[dst[i]] = -np.inf
        # Count how many nodes score higher than the true target
        ranks[i] = np.sum(scores > scores[dst[i]]) + 1

    return {
        "hits@1": hits_at_k(ranks, 1),
        "hits@3": hits_at_k(ranks, 3),
        "hits@10": hits_at_k(ranks, 10),
        "mrr": mean_reciprocal_rank(ranks),
        "p@10": precision_at_k(ranks, 10),
    }
