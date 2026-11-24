import sys

sys.path.insert(0, ".")
from sqlalchemy.orm import Session
from sqlalchemy import select, text

from app.db.session import get_engine
from app.db.base import Base
import app.models  # noqa
from app.models.paper import Paper
from ml.serving.backfill import backfill_paper_embeddings

e = get_engine()
with Session(e) as db:
    n = backfill_paper_embeddings(db, limit=3)
    print("encoded:", n)
    rows = db.execute(
        text(
            "select id, typeof(embedding), substr(coalesce(embedding,'X'),1,40) "
            "from papers where source='arxiv' limit 5"
        )
    ).all()
    for r in rows:
        print(r)
    # what does the ORM see?
    p = db.scalar(select(Paper).where(Paper.source == "arxiv").limit(1))
    print("orm:", type(p.embedding).__name__, repr(p.embedding)[:60])
