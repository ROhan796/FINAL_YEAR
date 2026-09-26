FROM python:3.11-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[all]" 2>/dev/null || \
    pip install --no-cache-dir fastapi uvicorn pydantic pydantic-settings \
    sqlalchemy psycopg[binary,pool] alembic pgvector \
    python-jose[cryptography] passlib[bcrypt] python-multipart \
    httpx redis aiosqlite orjson structlog tenacity numpy scikit-learn

# Copy code
COPY . .

# Dev entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
