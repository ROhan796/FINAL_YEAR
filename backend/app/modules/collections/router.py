"""Collection endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Response

from app.core.deps import CurrentUserDep, DbSession, PageParams, Pagination, ReadDbSession
from app.modules.collections.service import CollectionService
from app.schemas import (
    AiOverviewOut,
    CitationFormat,
    CollectionCreate,
    CollectionOut,
    CollectionPaperIn,
    CollectionPatch,
)
from app.schemas.common import OkResponse, PageResponse

router = APIRouter(prefix="/collections", tags=["collections"])


@router.get("", response_model=PageResponse[CollectionOut])
def list_collections(
    db: ReadDbSession,
    user: CurrentUserDep,
    page: Annotated[PageParams, Depends(Pagination(default=50))],
) -> Any:
    return CollectionService(db).list(user["id"], page)


@router.post("", response_model=CollectionOut, status_code=201)
def create_collection(db: DbSession, user: CurrentUserDep, payload: CollectionCreate) -> Any:
    return CollectionService(db).create(user["id"], payload)


@router.get("/{collection_id}", response_model=CollectionOut)
def get_collection(collection_id: str, db: ReadDbSession, user: CurrentUserDep) -> Any:
    service = CollectionService(db)
    return service._out(service.get(collection_id, user["id"]))


@router.patch("/{collection_id}", response_model=CollectionOut)
def patch_collection(
    collection_id: str, db: DbSession, user: CurrentUserDep, payload: CollectionPatch
) -> Any:
    return CollectionService(db).patch(collection_id, user["id"], payload)


@router.delete("/{collection_id}", response_model=OkResponse)
def delete_collection(collection_id: str, db: DbSession, user: CurrentUserDep) -> Any:
    CollectionService(db).delete(collection_id, user["id"])
    return OkResponse()


@router.post("/{collection_id}/papers", response_model=CollectionOut, status_code=201)
def add_paper(
    collection_id: str, db: DbSession, user: CurrentUserDep, payload: CollectionPaperIn
) -> Any:
    return CollectionService(db).add_paper(collection_id, user["id"], payload.paperId)


@router.delete("/{collection_id}/papers/{paper_id}", response_model=CollectionOut)
def remove_paper(collection_id: str, paper_id: str, db: DbSession, user: CurrentUserDep) -> Any:
    service = CollectionService(db)
    service.remove_paper(collection_id, user["id"], paper_id)
    return service._out(service.get(collection_id, user["id"]))


@router.post("/{collection_id}/ai-overview", response_model=AiOverviewOut)
def ai_overview(collection_id: str, db: DbSession, user: CurrentUserDep) -> Any:
    return CollectionService(db).ai_overview(collection_id, user["id"])


@router.get("/{collection_id}/clusters")
def collection_clusters(collection_id: str, db: ReadDbSession, user: CurrentUserDep) -> dict[str, Any]:
    """6.E: Contrastive collection clustering (K-Means + TF-IDF topic naming)."""
    from app.modules.insights.service import InsightsService

    return InsightsService(db).collection_clusters(collection_id, user["id"])


@router.get("/{collection_id}/export")
def export(
    collection_id: str,
    db: ReadDbSession,
    user: CurrentUserDep,
    format: Annotated[CitationFormat, Query()] = "bib",
) -> Response:
    service = CollectionService(db)
    body = service.export(collection_id, user["id"], "BibTeX" if format == "bib" else format)
    media = "application/x-bibtex" if format == "bib" else "text/plain"
    filename = f"collection-{collection_id}.{format}"
    return Response(
        content=body,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


__all__ = ["router"]
