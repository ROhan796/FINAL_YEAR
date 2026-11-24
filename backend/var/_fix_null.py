import sqlite3

conn = sqlite3.connect("ckgai_dev.db")
for table in ("papers", "paper_chunks", "research_nodes", "concepts", "citations", "entity_mentions"):
    try:
        cur = conn.execute(
            f"UPDATE {table} SET embedding = NULL WHERE embedding = 'null'"
        )
        print(table, "papers/chunks fixed:", cur.rowcount)
    except sqlite3.OperationalError as e:
        print(table, "skip:", e)
try:
    cur = conn.execute("UPDATE papers SET search_tsv = NULL WHERE search_tsv = 'null'")
    print("search_tsv fixed:", cur.rowcount)
except sqlite3.OperationalError as e:
    print("search_tsv skip:", e)
conn.commit()
# verify
print(
    "remaining literal nulls:",
    conn.execute(
        "select (select count(*) from papers where embedding='null') + "
        "(select count(*) from paper_chunks where embedding='null')"
    ).fetchone()[0],
)
print(
    "papers null:",
    conn.execute("select count(*) from papers where embedding is null").fetchone()[0],
)
