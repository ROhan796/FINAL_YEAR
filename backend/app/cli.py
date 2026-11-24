"""Unified CLI entrypoint (`ckgai` script in pyproject.toml).

Usage:
    ckgai seed
    ckgai ingest --source arxiv --query "graph neural networks" --max 50 --embed
    ckgai backfill --limit 100
    ckgai index-kg --limit 100
    ckgai train-gnn --epochs 25
    ckgai discover --train
    ckgai project-neo4j
    ckgai health
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ckgai", description="CKGAI Backend & ML Operations CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("seed", help="Seed development corpus into database")

    p_ingest = sub.add_parser("ingest", help="Ingest papers from arXiv, PubMed, or Semantic Scholar")
    p_ingest.add_argument("--source", choices=["arxiv", "pubmed", "semantic_scholar"], default="arxiv")
    p_ingest.add_argument("--query", default="graph neural networks")
    p_ingest.add_argument("--category", default=None)
    p_ingest.add_argument("--max", type=int, default=50)
    p_ingest.add_argument("--embed", action="store_true")
    p_ingest.add_argument("--kg", action="store_true")

    p_backfill = sub.add_parser("backfill", help="Backfill SPECTER2/SciBERT embeddings & related papers")
    p_backfill.add_argument("--limit", type=int, default=None)

    p_kg = sub.add_parser("index-kg", help="Run NER + RE + KG construction over stored papers")
    p_kg.add_argument("--limit", type=int, default=200)

    p_gnn = sub.add_parser("train-gnn", help="Train PyTorch CKGAI GNN on current KG")
    p_gnn.add_argument("--epochs", type=int, default=30)

    p_disc = sub.add_parser("discover", help="Run cross-domain link prediction & populate discoveries")
    p_disc.add_argument("--train", action="store_true")
    p_disc.add_argument("--limit", type=int, default=50)

    p_proj = sub.add_parser("project-neo4j", help="Run incremental Neo4j graph projection")
    p_proj.add_argument("--full", action="store_true")

    sub.add_parser("health", help="Check database and ML plane readiness")

    args = parser.parse_args(argv)

    from app.db.base import Base
    from app.db.session import get_engine
    import app.models  # noqa: F401
    from sqlalchemy.orm import Session

    engine = get_engine()
    Base.metadata.create_all(engine)

    if args.command == "seed":
        from scripts.seed import seed

        seed()
        return 0

    if args.command == "health":
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("database: ok")
        return 0

    with Session(engine) as db:
        if args.command == "ingest":
            from ml.pipelines.ingest import IngestPipeline

            pipe = IngestPipeline(db)
            if args.source == "arxiv":
                res = pipe.ingest_arxiv(args.query, max_results=args.max, category=args.category)
            elif args.source == "pubmed":
                res = pipe.ingest_pubmed(args.query, max_results=args.max)
            else:
                res = pipe.ingest_semantic_scholar(args.query, max_results=args.max)
            print(res)
            if args.embed:
                from ml.serving.backfill import backfill_all

                print("backfill:", backfill_all(db))
            if args.kg:
                from ml.pipelines.index import IndexPipeline

                print("kg:", IndexPipeline(db).index_all_missing(limit=args.max))
            return 0

        if args.command == "backfill":
            from ml.serving.backfill import backfill_all

            print(backfill_all(db, limit=args.limit))
            return 0

        if args.command == "index-kg":
            from ml.pipelines.index import IndexPipeline

            print(IndexPipeline(db).index_all_missing(limit=args.limit))
            return 0

        if args.command == "train-gnn":
            from ml.pipelines.discovery import DiscoveryPipeline

            res = DiscoveryPipeline(db).run(train=True, epochs=args.epochs, persist=True)
            print(
                f"trained={res.model_trained} discoveries={len(res.discoveries)} "
                f"hits@10={res.hits_at_10:.3f} mrr={res.mrr:.3f}"
            )
            return 0

        if args.command == "discover":
            from ml.pipelines.discovery import DiscoveryPipeline

            res = DiscoveryPipeline(db).run(train=args.train, persist=True)
            print(f"discoveries={len(res.discoveries)} candidates={res.total_candidates}")
            return 0

        if args.command == "project-neo4j":
            from ml.graph.projector import Neo4jProjector

            print(Neo4jProjector(db).project(full=args.full))
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
