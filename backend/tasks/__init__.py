"""Background job definitions for Arq workers.

Heavy operations are enqueued here to avoid blocking API requests:
- Paper ingestion from external sources
- Embedding generation (batch)
- NER + RE on papers
- KG construction
- GNN training
- Discovery generation
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


async def ingest_paper_job(ctx: dict, paper_data: dict) -> dict:
    """Background job: ingest a single paper through the full pipeline."""
    from ml.pipelines.index import IndexPipeline

    pipeline = IndexPipeline()
    result = pipeline.index_paper(
        paper_id=paper_data.get("id", ""),
        title=paper_data.get("title", ""),
        abstract=paper_data.get("abstract", ""),
        domain=paper_data.get("domain", ""),
    )
    return {
        "papers_indexed": result.papers_indexed,
        "entities_extracted": result.entities_extracted,
        "relations_extracted": result.relations_extracted,
        "kg_nodes_created": result.kg_nodes_created,
        "kg_edges_created": result.kg_edges_created,
    }


async def batch_embed_job(ctx: dict, paper_ids: list[str]) -> dict:
    """Background job: generate embeddings for a batch of papers."""
    log.info("batch_embed_job", count=len(paper_ids))
    # In production, reads papers from DB, generates embeddings, stores them
    return {"processed": len(paper_ids)}


async def discovery_job(ctx: dict) -> dict:
    """Background job: run the full discovery pipeline."""
    from ml.pipelines.discovery import DiscoveryPipeline

    pipeline = DiscoveryPipeline()
    result = pipeline.run(train=True)
    return {
        "discoveries_found": len(result.discoveries),
        "total_candidates": result.total_candidates,
        "model_trained": result.model_trained,
    }


async def gnn_training_job(ctx: dict) -> dict:
    """Background job: train the GNN model."""
    from ml.gnn.dataset import GraphDataset
    from ml.gnn.model import CKGAI_GNN
    from ml.gnn.train import GNNTrainer

    dataset = GraphDataset()
    graph = dataset.load_graph()

    model = CKGAI_GNN(use_virtual_node=True)
    trainer = GNNTrainer(model)

    losses = []
    for epoch in range(10):
        loss = trainer.train_epoch(
            graph.node_features,
            graph.adj_by_relation,
            graph.edge_src, graph.edge_dst, graph.edge_types,
            len(graph.node_ids),
        )
        losses.append(loss)

    return {
        "epochs": len(losses),
        "final_loss": losses[-1] if losses else 0,
    }


# Arq job configuration
from arq import cron

JOB_DEFINITIONS = {
    "ingest_paper": ingest_paper_job,
    "batch_embed": batch_embed_job,
    "run_discovery": discovery_job,
    "train_gnn": gnn_training_job,
}

# Cron jobs (optional)
CRON_JOBS = [
    cron(gnn_training_job, hour=2, minute=0),  # Train GNN at 2 AM daily
]
