#!/usr/bin/env python3
"""End-to-end corpus ingestion: arXiv → store → chunks → embeddings → k-NN.

    cd backend
    python -m scripts.ingest_corpus --query "graph neural networks" --max 100
    python -m scripts.ingest_corpus --category cs.LG --max 200 --embed
    python -m scripts.ingest_corpus --ids 1706.03762,1810.04805 --embed
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest real papers from arXiv")
    parser.add_argument("--query", default="", help="free-text arXiv query")
    parser.add_argument("--category", default=None, help="arXiv category, e.g. cs.LG")
    parser.add_argument("--max", type=int, default=100, help="max papers to fetch")
    parser.add_argument("--ids", default="", help="comma-separated arXiv ids (overrides query)")
    parser.add_argument("--embed", action="store_true", help="run embedding backfill after ingest")
    parser.add_argument("--kg", action="store_true", help="run NER/RE/KG build after ingest")
    args = parser.parse_args()

    from app.db.session import get_engine
    from app.db.base import Base
    import app.models  # noqa: F401
    from sqlalchemy.orm import Session

    engine = get_engine()
    Base.metadata.create_all(engine)

    from ml.pipelines.ingest import IngestPipeline

    with Session(engine) as db:
        pipeline = IngestPipeline(db)

        if args.ids:
            from ml.ingest.arxiv import fetch_arxiv_by_ids

            records = fetch_arxiv_by_ids([i.strip() for i in args.ids.split(",") if i.strip()])
            stored, skipped, chunks = pipeline._store_many(records, source="arxiv")
            print(f"by-ids: fetched={len(records)} stored={stored} skipped={skipped} chunks={chunks}")
        else:
            result = pipeline.ingest_arxiv(args.query, max_results=args.max, category=args.category)
            print(
                f"arxiv: fetched={result.papers_fetched} stored={result.papers_stored} "
                f"skipped={result.papers_skipped} chunks={result.chunks_created}"
            )
            if result.errors:
                print("errors:", result.errors)

        if args.embed:
            from ml.serving.backfill import backfill_all

            stats = backfill_all(db)
            print("backfill:", stats)

        if args.kg:
            from sqlalchemy import select

            from app.models.paper import Paper
            from ml.pipelines.index import IndexPipeline

            papers = list(db.scalars(select(Paper).limit(args.max)).all())
            indexer = IndexPipeline(db)
            total_nodes = total_edges = 0
            for p in papers:
                r = indexer.index_paper(p.id, p.title, p.abstract, domain=p.domain)
                total_nodes += r.kg_nodes_created
                total_edges += r.kg_edges_created
            print(f"kg: papers={len(papers)} nodes={total_nodes} edges={total_edges}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
