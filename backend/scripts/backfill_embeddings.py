#!/usr/bin/env python3
"""Embedding backfill CLI.

    cd backend
    python -m scripts.backfill_embeddings              # embed everything missing
    python -m scripts.backfill_embeddings --limit 50
    python -m scripts.backfill_embeddings --related    # only rebuild k-NN
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill SPECTER2/SciBERT embeddings")
    parser.add_argument("--limit", type=int, default=None, help="max papers/chunks to embed")
    parser.add_argument("--related", action="store_true", help="only rebuild related_papers k-NN")
    parser.add_argument("--papers-only", action="store_true", help="only papers.embedding")
    args = parser.parse_args()

    from app.db.session import get_engine
    from app.db.base import Base
    import app.models  # noqa: F401 - register tables
    from sqlalchemy.orm import Session

    engine = get_engine()
    Base.metadata.create_all(engine)

    from ml.serving.backfill import (
        backfill_chunk_embeddings,
        backfill_paper_embeddings,
        rebuild_related_papers,
    )

    with Session(engine) as db:
        if args.related:
            n = rebuild_related_papers(db)
            print(f"related pairs rebuilt: {n}")
            return 0
        if args.papers_only:
            n = backfill_paper_embeddings(db, limit=args.limit)
            print(f"paper embeddings written: {n}")
            return 0
        p = backfill_paper_embeddings(db, limit=args.limit)
        c = backfill_chunk_embeddings(db, limit=args.limit)
        r = rebuild_related_papers(db)
        print(f"paper embeddings: {p}\nchunk embeddings: {c}\nrelated pairs: {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
