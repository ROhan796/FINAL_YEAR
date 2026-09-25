"""Single mount point for the versioned API.

Everything the frontend calls lives under ``/api/v1`` except the five frozen
legacy AI routes, which keep their historical ``/api/*`` paths (see
``app.ai.router``). The routers carry their own prefixes so this file only
wires them up — no path construction happens here.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.ai.router import router as legacy_ai_router
from app.modules.auth.router import router as auth_router
from app.modules.collections.router import router as collections_router
from app.modules.feed.router import router as feed_router
from app.modules.graph.router import router as graph_router
from app.modules.network.router import router as network_router
from app.modules.notifications.router import router as notifications_router
from app.modules.papers.router import router as papers_router
from app.modules.papers.router import saved_router as saved_paper_router
from app.modules.publishing.router import router as publishing_router
from app.modules.search.router import router as search_router
from app.modules.trending.router import router as trending_router
from app.modules.users.router import router as users_router
from app.modules.users.router import verification_router as verification_request_router
from app.modules.workspace.router import router as workspace_router

v1 = APIRouter(prefix="/api/v1")

v1.include_router(auth_router)
v1.include_router(users_router)
v1.include_router(verification_request_router)
v1.include_router(papers_router)
v1.include_router(saved_paper_router)
v1.include_router(search_router)
v1.include_router(feed_router)
v1.include_router(collections_router)
v1.include_router(network_router)
v1.include_router(workspace_router)
v1.include_router(publishing_router)
v1.include_router(notifications_router)
v1.include_router(trending_router)
v1.include_router(graph_router)

#: Frozen prototype routes — paths and shapes must never change.
legacy = APIRouter()
legacy.include_router(legacy_ai_router)

__all__ = ["v1", "legacy"]
