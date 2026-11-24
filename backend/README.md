# ckgai-backend

FastAPI backend + modular 7-layer CKGAI ML pipeline for the AI-Native Research Intelligence Platform.

## Run

```bash
# API server (from backend/)
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Arq background worker (requires Redis)
.venv\Scripts\python.exe -m arq tasks.WorkerSettings

# Unified CLI
.venv\Scripts\python.exe -m app.cli --help
```

Or run `start_all.bat` in the repository root.

## Testing

```bash
cd backend
.venv\Scripts\python.exe -m pytest -v
```

## Layout

```text
app/
  main.py        FastAPI factory, CORS, OpenTelemetry, lifespan, health, /api/v1/meta
  cli.py         `ckgai` CLI (seed, ingest, backfill, index-kg, train-gnn, discover, project-neo4j)
  ai/            LLMProvider (Gemini / OpenAI / Fallback) + 5 frozen /api/* routes + SSE stream
  core/          config, security (JWT), deps, cache (Redis), telemetry (OTel), logging, pagination
  db/            SQLAlchemy engine/session (Postgres+pgvector & SQLite dialect-aware)
  models/        34 ORM tables (including reading_annotations)
  modules/       14 domain modules (auth, users, papers, search, feed, collections, network,
                 workspace, publishing, notifications, trending, graph, ingestion, insights)
ml/              7-layer pipeline (ingest, preprocess, encoders+MRL, ner+train, relation+polarity,
                 graph+projector, pytorch gnn+train, discovery+platt calibration, serving)
tasks/           Arq async job definitions + WorkerSettings
tests/           Pytest API contract, ML pipeline, and insights test suite
docs/            01-system-architecture.md, 02-api-contract.md
```

## Docs

- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health/live · `/health/ready`
- Architecture: [`docs/01-system-architecture.md`](./docs/01-system-architecture.md)
- API Contract: [`docs/02-api-contract.md`](./docs/02-api-contract.md)
