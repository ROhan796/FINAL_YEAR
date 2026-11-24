import sys

sys.path.insert(0, ".")
from sqlalchemy.orm import Session

from app.db.session import get_engine
from app.db.base import Base
import app.models  # noqa
from app.models.paper import Paper, PaperChunk
from sqlalchemy import select, text

e = get_engine()
with Session(e) as db:
    p = db.scalar(select(Paper).where(Paper.source == "arxiv").limit(1))
    print("before:", repr(p.embedding)[:80])
    p.embedding = [0.1] * 768
    db.commit()
    db.refresh(p)
    print("after assign list:", repr(p.embedding)[:80])
    raw = db.execute(
        text("select substr(embedding,1,30) from papers where id=:i"), {"i": p.id}
    ).scalar()
    print("raw:", raw)

    # now via numpy float32
    import numpy as np

    vec = np.random.rand(768).astype(np.float32)
    p.embedding = [float(x) for x in vec]
    db.commit()
    db.refresh(p)
    raw = db.execute(
        text("select substr(embedding,1,30) from papers where id=:i"), {"i": p.id}
    ).scalar()
    print("raw after np:", raw)
