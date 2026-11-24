"""Publishing endpoints (manuscript submission pipeline)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, UploadFile, status

from app.core.deps import CurrentUserDep, DbSession, ReadDbSession
from app.core.errors import ValidationAppError
from app.schemas import ManuscriptCreate, ManuscriptOut, ManuscriptStatusPatch

router = APIRouter(prefix="/manuscripts", tags=["publishing"])


@router.get("", response_model=list[ManuscriptOut])
def list_manuscripts(db: ReadDbSession, user: CurrentUserDep) -> Any:
    from app.modules.publishing.service import PublishingService

    return PublishingService(db).list(user["id"])


@router.post("", response_model=ManuscriptOut, status_code=status.HTTP_201_CREATED)
def create_manuscript(payload: ManuscriptCreate, db: DbSession, user: CurrentUserDep) -> Any:
    from app.modules.publishing.service import PublishingService

    return PublishingService(db).create(user, payload)


@router.get("/{manuscript_id}", response_model=ManuscriptOut)
def get_manuscript(manuscript_id: str, db: ReadDbSession, user: CurrentUserDep) -> Any:
    from app.modules.publishing.service import PublishingService

    service = PublishingService(db)
    return service.out(service.get(manuscript_id, user["id"]))


@router.post("/{manuscript_id}/file", response_model=ManuscriptOut)
def upload_file(
    manuscript_id: str,
    db: DbSession,
    user: CurrentUserDep,
    file: UploadFile = File(...),
) -> Any:
    from app.modules.publishing.service import PublishingService

    # FastAPI has already spooled the body, so the sync handle is available.
    data = file.file.read()
    return PublishingService(db).attach_file(
        manuscript_id, user, file.filename or "manuscript.pdf", data
    )


@router.post("/{manuscript_id}/ai-prereview")
def ai_prereview(manuscript_id: str, db: DbSession, user: CurrentUserDep) -> Any:
    from app.modules.publishing.service import PublishingService

    return PublishingService(db).prereview(manuscript_id, user)


@router.post("/{manuscript_id}/citation-gaps")
def citation_gaps(
    manuscript_id: str,
    db: ReadDbSession,
    user: CurrentUserDep,
    draftText: str = "",
) -> dict[str, Any]:
    """7.4: Detect methods mentioned without citations and suggest KG/corpus papers."""
    from app.modules.publishing.service import PublishingService

    return PublishingService(db).citation_gaps(manuscript_id, user, draft_text=draftText)


@router.patch("/{manuscript_id}/status", response_model=ManuscriptOut)
def set_status(
    manuscript_id: str,
    payload: ManuscriptStatusPatch,
    db: DbSession,
    user: CurrentUserDep,
) -> Any:
    """Reviewer-only states are enforced in the service against `user.role`."""
    from app.modules.publishing.service import PublishingService

    return PublishingService(db).set_status(manuscript_id, user, str(payload.status))


__all__ = ["router", "ValidationAppError"]
