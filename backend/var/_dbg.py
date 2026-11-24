import sqlite3

conn = sqlite3.connect("ckgai_dev.db")
cur = conn.execute(
    "select source, typeof(embedding), substr(coalesce(embedding,'XNULL'),1,50) from papers limit 10"
)
for r in cur:
    print(r)
print("--- groups ---")
print(conn.execute("select typeof(embedding), count(*) from papers group by 1").fetchall())
print(
    "non-null:",
    conn.execute("select count(*) from papers where embedding is not null").fetchone(),
)
print(
    "literal null str:",
    conn.execute(
        "select count(*) from papers where embedding = 'null'"
    ).fetchone(),
)
print(
    "chunks emb:",
    conn.execute(
        "select count(*) from paper_chunks where embedding is not null"
    ).fetchone(),
)
print(
    "chunks emb null str:",
    conn.execute(
        "select count(*) from paper_chunks where embedding = 'null'"
    ).fetchone(),
)
