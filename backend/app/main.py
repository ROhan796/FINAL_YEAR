"""Application factory.

Layers (see docs/01): transport → application (routers/services) → domain
(modules) → infrastructure (db, ml, providers). Nothing below the transport
layer imports from here.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import legacy, v1
from app.core.config import settings
from app.core.errors import register_exception_handlers
from app.core.logging import get_logger, request_id_var

log = get_logger(__name__)

API_DESCRIPTION = (
    "CKGAI — AI Scientific Paper Knowledge Engine. "
    "Serves the Next.js client; the five legacy `/api/*` AI routes are frozen."
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from app.db.session import get_engine

    engine = get_engine()
    if settings.app_env == "dev":
        # Dev convenience: ensure the schema exists before the first request.
        # Production uses `alembic upgrade head` (see README).
        from pathlib import Path

        from app.db.base import Base
        import app.models  # noqa: F401  - registers every table on Base

        if settings.database_url.startswith("sqlite"):
            db_path = settings.database_url.split("///", 1)[-1]
            if db_path and db_path != ":memory:":
                Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

        Base.metadata.create_all(engine)
        log.info("database_ready", url=settings.database_url.split("@")[-1])

    if settings.ml_enabled:
        try:
            from ml.warmup import warm  # optional heavy imports happen off the request path

            warm()
        except Exception as exc:  # noqa: BLE001 - the API must boot without the ML plane
            log.warning("ml_warmup_failed", error=str(exc))

    yield

    from app.db.session import dispose_engines

    dispose_engines()


def create_app() -> FastAPI:
    app = FastAPI(
        title="CKGAI API",
        description=API_DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url=None,
        openapi_url="/openapi.json" if not settings.is_production else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-Id"],
    )

    @app.middleware("http")
    async def _request_id(request: Request, call_next: Any) -> Any:  # noqa: ANN401
        rid = request.headers.get("X-Request-Id") or ""
        request_id_var.set(rid or "")
        response = await call_next(request)
        if rid:
            response.headers["X-Request-Id"] = rid
        return response

    register_exception_handlers(app)

    from app.core.telemetry import configure_telemetry

    configure_telemetry(app)

    app.include_router(v1)
    app.include_router(legacy)

    _register_health(app)
    _register_meta(app)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, Any]:
        return {"name": "CKGAI API", "version": "1.0.0", "docs": "/docs"}

    return app


def _register_health(app: FastAPI) -> None:
    @app.get("/health/live", tags=["health"])
    def live() -> dict[str, str]:
        """Liveness: process is up. Never touches the database."""
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    def ready(request: Request) -> JSONResponse:
        """Readiness: database reachable. Load balancers gate on this."""
        try:
            from sqlalchemy import text

            from app.db.session import get_engine

            with get_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as exc:  # noqa: BLE001
            log.error("readiness_failed", error=str(exc))
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "degraded", "database": "unreachable"},
            )
        payload: dict[str, Any] = {"status": "ok", "database": "ok"}
        if settings.neo4j_enabled:
            payload["graph"] = _graph_status()
        return JSONResponse(content=payload)


def _graph_status() -> str:
    try:
        from app.modules.graph.service import _get_neo4j_driver

        _get_neo4j_driver().verify_connectivity()
        return "ok"
    except Exception:  # noqa: BLE001 - Neo4j is optional
        return "degraded"


def _register_meta(app: FastAPI) -> None:
    @app.get("/api/v1/meta", tags=["health"])
    def meta() -> dict[str, Any]:
        """Feature flags and limits the client can adapt to."""
        return {
            "appEnv": settings.app_env,
            "devAutoLogin": settings.dev_auto_login,
            "maxUploadBytes": settings.max_upload_bytes,
            "searchModes": ["semantic", "keyword", "hybrid"],
            "roles": ["reader", "researcher", "verified_researcher", "editor", "reviewer", "admin"],
            "aiRoutes": [
                "/api/search",
                "/api/summarize",
                "/api/generate-insights",
                "/api/analyze-draft",
                "/api/chat-assistant",
            ],
        }


app = create_app()

__all__ = ["app", "create_app"]
