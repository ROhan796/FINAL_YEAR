import sys

sys.path.insert(0, ".")
from sqlalchemy import create_engine, text

from app.core.config import settings

e = create_engine(settings.database_url)
with e.connect() as c:
    def n(sql):
        return c.execute(text(sql)).scalar()

    print("papers:", n("select count(*) from papers"))
    print("papers w/ emb:", n("select count(*) from papers where embedding is not null and embedding != 'null'"))
    print("chunks:", n("select count(*) from paper_chunks"))
    print("chunks w/ emb:", n("select count(*) from paper_chunks where embedding is not null and embedding != 'null'"))
    has_rel = c.execute(text("select name from sqlite_master where name='related_papers'")).first()
    print("related pairs:", n("select count(*) from related_papers") if has_rel else "no table")
    print("arxiv papers:", n("select count(*) from papers where source='arxiv'"))
    # sample related scores
    if has_rel:
        rows = c.execute(text("select paper_id, related_to_id, score from related_papers limit 3")).all()
        for r in rows:
            print(" rel:", r)
