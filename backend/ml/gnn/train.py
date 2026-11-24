"""PyTorch GNN training loop.

Spec:
- Loss: BCE + margin ranking (gamma = 1.0)
- Optimizer: AdamW lr = 1e-3, weight decay 1e-5
- Cosine annealing schedule
- Gradient clipping: max norm = 1.0
- Negative sampling: 5 per positive, type-aware
- Eval: Hits@10, MRR
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.gnn.model import CKGAI_GNN

log = logging.getLogger(__name__)


@dataclass
class TrainMetrics:
    """Training metrics per epoch."""

    epoch: int
    loss: float
    hits_at_10: float
    mrr: float


class GNNTrainer:
    """Real PyTorch autograd trainer for CKGAI_GNN + DistMult relation embeddings."""

    def __init__(
        self,
        model: CKGAI_GNN,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        margin: float = 1.0,
        negative_samples: int = 5,
        max_grad_norm: float = 1.0,
        total_epochs: int = 30,
    ) -> None:
        self.model = model
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.margin = margin
        self.neg_samples = negative_samples
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_epochs)
        )

    def generate_negatives(
        self,
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
        edge_types: np.ndarray,
        num_nodes: int,
        num_neg: int = 5,
        node_types: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate type-constrained negative samples."""
        N = len(edge_src)
        if N == 0 or num_nodes <= 1:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        type_buckets: dict[str, np.ndarray] = {}
        if node_types and len(node_types) == num_nodes:
            grouped: dict[str, list[int]] = {}
            for idx, nt in enumerate(node_types):
                grouped.setdefault(nt, []).append(idx)
            type_buckets = {k: np.array(v, dtype=np.int64) for k, v in grouped.items()}

        neg_src = np.zeros(N * num_neg, dtype=np.int64)
        neg_dst = np.zeros(N * num_neg, dtype=np.int64)
        neg_types = np.zeros(N * num_neg, dtype=np.int64)

        idx = 0
        for i in range(N):
            for _ in range(num_neg):
                if np.random.random() < 0.5:
                    pool = (
                        type_buckets.get(node_types[int(edge_src[i])])
                        if node_types
                        else None
                    )
                    cand = int(np.random.choice(pool)) if pool is not None and len(pool) > 1 else np.random.randint(0, num_nodes)
                    neg_src[idx] = cand
                    neg_dst[idx] = edge_dst[i]
                else:
                    pool = (
                        type_buckets.get(node_types[int(edge_dst[i])])
                        if node_types
                        else None
                    )
                    cand = int(np.random.choice(pool)) if pool is not None and len(pool) > 1 else np.random.randint(0, num_nodes)
                    neg_src[idx] = edge_src[i]
                    neg_dst[idx] = cand
                neg_types[idx] = edge_types[i]
                idx += 1

        return neg_src[:idx], neg_dst[:idx], neg_types[:idx]

    def _distmult_scores(
        self,
        Z: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        rel_idx: torch.Tensor,
    ) -> torch.Tensor:
        r = self.model.rel_embeddings[rel_idx % self.model.rel_embeddings.size(0)]
        return (Z[src] * r * Z[dst]).sum(dim=-1)

    def train_epoch(
        self,
        node_features: np.ndarray,
        adj_by_relation: dict[int, list[tuple[int, int]]],
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
        edge_types: np.ndarray,
        num_nodes: int,
        node_types: list[str] | None = None,
    ) -> float:
        """Train one full-graph epoch with BCE + margin ranking loss."""
        if len(edge_src) == 0 or num_nodes < 2:
            return 0.0

        self.model.train()
        device = self.model.rel_embeddings.device
        x = torch.as_tensor(node_features, dtype=torch.float32, device=device)

        adj_list = self.model._build_adj_list(adj_by_relation, num_nodes)
        Z, _ = self.model.forward_tensor(x, adj_by_relation, adj_list)

        p_src = torch.as_tensor(edge_src, dtype=torch.long, device=device)
        p_dst = torch.as_tensor(edge_dst, dtype=torch.long, device=device)
        p_rel = torch.as_tensor(edge_types, dtype=torch.long, device=device)

        n_src_np, n_dst_np, n_rel_np = self.generate_negatives(
            edge_src, edge_dst, edge_types, num_nodes, self.neg_samples, node_types=node_types
        )
        n_src = torch.as_tensor(n_src_np, dtype=torch.long, device=device)
        n_dst = torch.as_tensor(n_dst_np, dtype=torch.long, device=device)
        n_rel = torch.as_tensor(n_rel_np, dtype=torch.long, device=device)

        pos_scores = self._distmult_scores(Z, p_src, p_dst, p_rel)
        neg_scores = self._distmult_scores(Z, n_src, n_dst, n_rel)

        # BCE loss on pos/neg logits
        logits = torch.cat([pos_scores, neg_scores], dim=0)
        targets = torch.cat(
            [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0
        )
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        # Margin ranking loss: repeat positives to align with neg_samples negatives
        pos_rep = pos_scores.repeat_interleave(self.neg_samples)
        margin_loss = F.relu(self.margin - pos_rep + neg_scores).mean()

        loss = bce_loss + margin_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        return float(loss.item())

    def fit(
        self,
        node_features: np.ndarray,
        adj_by_relation: dict[int, list[tuple[int, int]]],
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
        edge_types: np.ndarray,
        num_nodes: int,
        *,
        epochs: int = 30,
        node_types: list[str] | None = None,
    ) -> list[TrainMetrics]:
        history: list[TrainMetrics] = []
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(
                node_features, adj_by_relation, edge_src, edge_dst, edge_types,
                num_nodes, node_types=node_types,
            )
            if epoch == 1 or epoch == epochs or epoch % 10 == 0:
                embs = self.model.predict_embeddings(node_features, adj_by_relation)
                h10, mrr = self.evaluate(embs, (edge_src, edge_dst, edge_types), num_nodes)
                history.append(TrainMetrics(epoch=epoch, loss=loss, hits_at_10=h10, mrr=mrr))
            else:
                history.append(TrainMetrics(epoch=epoch, loss=loss, hits_at_10=0.0, mrr=0.0))
        return history

    def evaluate(
        self,
        embeddings: np.ndarray,
        test_edges: tuple[np.ndarray, np.ndarray, np.ndarray],
        num_nodes: int,
    ) -> tuple[float, float]:
        """Evaluate Hits@10 and MRR using learned DistMult relation weights."""
        test_src, test_dst, test_types = test_edges
        if len(test_src) == 0:
            return 0.0, 0.0

        rel_mat = self.model.relation_embedding_matrix()
        hits = 0
        mrr = 0.0

        for i in range(len(test_src)):
            s = int(test_src[i])
            d = int(test_dst[i])
            r_idx = int(test_types[i]) % rel_mat.shape[0]
            query_vec = embeddings[s] * rel_mat[r_idx]
            scores = embeddings @ query_vec
            true_score = float(scores[d])
            rank = int(np.sum(scores > true_score)) + 1
            if rank <= 10:
                hits += 1
            mrr += 1.0 / rank

        return hits / len(test_src), mrr / len(test_src)


__all__ = ["GNNTrainer", "TrainMetrics"]
