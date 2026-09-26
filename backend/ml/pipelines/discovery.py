"""Discovery pipeline — GNN training + link prediction + ranking.

Layers 6-7 of the CKGAI pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ml.gnn.model import CKGAI_GNN
from ml.gnn.dataset import GraphDataset
from ml.gnn.train import GNNTrainer
from ml.discovery.ranker import DiscoveryRanker, Discovery

log = logging.getLogger(__name__)


@dataclass
class DiscoveryPipelineResult:
    discoveries: list[Discovery] = field(default_factory=list)
    total_candidates: int = 0
    hits_at_10: float = 0.0
    mrr: float = 0.0
    model_trained: bool = False


class DiscoveryPipeline:
    """Run the full discovery pipeline:
    1. Load graph data from Postgres
    2. Train GNN (or load checkpoint)
    3. Generate node embeddings
    4. Run link prediction
    5. Rank cross-domain discoveries
    """

    def __init__(self, db_session=None, device: str = "cpu"):
        self.db = db_session
        self.dataset = GraphDataset(db_session)
        self.gnn = CKGAI_GNN(use_virtual_node=True)
        self.ranker = DiscoveryRanker()

    def run(self, train: bool = False) -> DiscoveryPipelineResult:
        """Run the full discovery pipeline."""
        result = DiscoveryPipelineResult()

        try:
            # Load graph
            graph = self.dataset.load_graph(feature_dim=768)
            log.info("discovery_graph_loaded: nodes=%s edges=%s", len(graph.node_ids), len(graph.edge_src))

            if train:
                # Train GNN
                trainer = GNNTrainer(self.gnn)
                for epoch in range(10):
                    loss = trainer.train_epoch(
                        graph.node_features,
                        graph.adj_by_relation,
                        graph.edge_src, graph.edge_dst, graph.edge_types,
                        len(graph.node_ids),
                    )
                    log.info("discovery_train_epoch: epoch=%s loss=%s", epoch, f"{loss:.4f}")
                result.model_trained = True

            # Generate embeddings
            embeddings = self.gnn.predict_embeddings(
                graph.node_features, graph.adj_by_relation
            )

            # Get node metadata
            node_domains = ["unknown"] * len(graph.node_ids)
            node_names = graph.node_ids  # use IDs as names for now

            # Run discovery
            existing_edges = set()
            for s, d, t in zip(graph.edge_src, graph.edge_dst, graph.edge_types):
                if s < len(graph.node_ids) and d < len(graph.node_ids):
                    existing_edges.add((graph.node_ids[s], str(t), graph.node_ids[d]))

            discovery_result = self.ranker.discover(
                node_embeddings=embeddings,
                node_ids=graph.node_ids,
                node_domains=node_domains,
                node_names=node_names,
                existing_edges=existing_edges,
            )

            result.discoveries = discovery_result.discoveries
            result.total_candidates = discovery_result.total_candidates

        except Exception as exc:
            log.error("discovery_pipeline_failed: %s", exc)

        return result

    def warmup(self) -> None:
        self.gnn.warmup()
