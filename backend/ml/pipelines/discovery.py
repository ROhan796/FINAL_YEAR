"""Discovery pipeline — PyTorch GNN training + link prediction + DB persistence.

Layers 6-7 of the CKGAI pipeline.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ml.discovery.ranker import Discovery, DiscoveryRanker
from ml.gnn.dataset import GraphDataset
from ml.gnn.model import CKGAI_GNN
from ml.gnn.train import GNNTrainer

log = logging.getLogger(__name__)


@dataclass
class DiscoveryPipelineResult:
    discoveries: list[Discovery] = field(default_factory=list)
    total_candidates: int = 0
    hits_at_10: float = 0.0
    mrr: float = 0.0
    model_trained: bool = False
    persisted_count: int = 0


class DiscoveryPipeline:
    """Run GNN training / checkpoint loading -> DistMult ranking -> DB persistence."""

    def __init__(self, db_session=None, device: str = "cpu", checkpoint_dir: str | None = None):
        self.db = db_session
        self.device = device
        self.dataset = GraphDataset(db_session)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self._default_ckpt_dir()
        self.gnn = self._load_or_init_gnn()
        self.ranker = DiscoveryRanker(dim=self.gnn.out_dim)

    @staticmethod
    def _default_ckpt_dir() -> Path:
        try:
            from app.core.config import settings

            return Path(settings.ml_checkpoint_dir)
        except Exception:
            return Path("./models/checkpoints")

    def _load_or_init_gnn(self) -> CKGAI_GNN:
        ckpt = self.checkpoint_dir / "ckgai_gnn.pt"
        if ckpt.exists():
            try:
                return CKGAI_GNN.load_checkpoint(ckpt, device=self.device)
            except Exception as exc:  # noqa: BLE001
                log.warning("gnn_checkpoint_load_failed: %s", exc)
        return CKGAI_GNN(use_virtual_node=True).to(self.device)

    def run(self, train: bool = False, epochs: int = 30, persist: bool = False) -> DiscoveryPipelineResult:
        result = DiscoveryPipelineResult()

        try:
            graph = self.dataset.load_graph(feature_dim=768)
            log.info("discovery_graph_loaded: nodes=%s edges=%s", len(graph.node_ids), len(graph.edge_src))

            if train and len(graph.edge_src) > 0:
                trainer = GNNTrainer(self.gnn, total_epochs=epochs)
                history = trainer.fit(
                    graph.node_features,
                    graph.adj_by_relation,
                    graph.edge_src,
                    graph.edge_dst,
                    graph.edge_types,
                    len(graph.node_ids),
                    epochs=epochs,
                    node_types=graph.node_types,
                )
                if history:
                    result.hits_at_10 = history[-1].hits_at_10
                    result.mrr = history[-1].mrr
                    log.info(
                        "gnn_train_complete: epochs=%d loss=%.4f hits@10=%.3f mrr=%.3f",
                        epochs,
                        history[-1].loss,
                        result.hits_at_10,
                        result.mrr,
                    )
                self.gnn.save_checkpoint(
                    self.checkpoint_dir / "ckgai_gnn.pt",
                    metadata={"hits@10": result.hits_at_10, "mrr": result.mrr},
                )
                result.model_trained = True

            embeddings = self.gnn.predict_embeddings(graph.node_features, graph.adj_by_relation)
            rel_mat = self.gnn.relation_embedding_matrix()
            rel_names = graph.relation_names or ["used_in", "applied_to", "improves", "treats"]
            rel_map = {
                name: rel_mat[idx % rel_mat.shape[0]]
                for idx, name in enumerate(rel_names)
            }
            self.ranker.set_relation_embeddings(rel_map)

            existing_edges: set[tuple[str, str, str]] = set()
            for s, d, t in zip(graph.edge_src, graph.edge_dst, graph.edge_types):
                if s < len(graph.node_ids) and d < len(graph.node_ids):
                    rname = rel_names[int(t)] if int(t) < len(rel_names) else str(t)
                    existing_edges.add((graph.node_ids[s], rname, graph.node_ids[d]))

            node_paper_ids: dict[str, str] = {}
            if self.db is not None:
                from sqlalchemy import select
                from app.models.knowledge import KgNode

                for n in self.db.scalars(select(KgNode)).all():
                    if n.paper_id:
                        node_paper_ids[n.id] = n.paper_id

            discovery_result = self.ranker.discover(
                node_embeddings=embeddings,
                node_ids=graph.node_ids,
                node_domains=graph.node_domains or ["General"] * len(graph.node_ids),
                node_names=graph.node_names or graph.node_ids,
                existing_edges=existing_edges,
                relation_labels=list({"used_in", "applied_to", "improves", "treats", *rel_names}),
                node_types=graph.node_types or None,
                node_paper_ids=node_paper_ids,
            )

            result.discoveries = discovery_result.discoveries
            result.total_candidates = discovery_result.total_candidates

            if persist and self.db is not None and result.discoveries:
                result.persisted_count = self._persist_discoveries(result.discoveries)

        except Exception as exc:  # noqa: BLE001
            log.error("discovery_pipeline_failed: %s", exc)

        return result

    def _persist_discoveries(self, discoveries: list[Discovery]) -> int:
        from sqlalchemy import delete

        from app.models.knowledge import Discovery as DiscoveryRow

        # Replace non-dismissed, un-voted coldstart/previous batch rows cleanly
        self.db.execute(
            delete(DiscoveryRow).where(
                DiscoveryRow.dismissed.is_(False),
                DiscoveryRow.useful_votes == 0,
            )
        )
        batch_id = f"gnn-{uuid.uuid4().hex[:8]}"
        seen: set[tuple[str, str, str]] = set()
        inserted = 0
        for d in discoveries[:50]:
            key = (d.src_id, d.dst_id, d.relation)
            if key in seen:
                continue
            seen.add(key)
            self.db.add(
                DiscoveryRow(
                    src_id=d.src_id,
                    dst_id=d.dst_id,
                    relation=d.relation,
                    score=float(d.score),
                    scorer=d.scorer,
                    evidence_paths=d.evidence_paths,
                    evidence_paper_ids=d.evidence_paper_ids,
                    batch_id=batch_id,
                )
            )
            inserted += 1
        self.db.commit()
        return inserted

    def warmup(self) -> None:
        self.gnn.warmup()


__all__ = ["DiscoveryPipeline", "DiscoveryPipelineResult"]
