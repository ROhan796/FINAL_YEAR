"""User profile and verification-request endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends

from app.core.deps import CurrentUserDep, DbSession, ReadDbSession, require
from app.core.errors import NotFoundError
from app.models.user import User, UserProfile
from app.modules.auth.repository import AuthRepository
from app.modules.auth.service import AuthService
from app.schemas import ProfileUpdate, UserProfileOut, VerificationRequestIn, VerificationRequestOut
from app.schemas.common import OkResponse

router = APIRouter(prefix="/users", tags=["users"])


def _load(db: Any, user_id: str) -> tuple[User, UserProfile | None]:
    repo = AuthRepository(db)
    user = repo.get(user_id)
    if user is None:
        raise NotFoundError(f"User '{user_id}' not found")
    return user, repo.profile(user_id)


@router.get("/me", response_model=UserProfileOut)
def me(db: ReadDbSession, user: CurrentUserDep) -> Any:
    row, profile = _load(db, user["id"])
    return AuthService.profile_out(row, profile)


@router.get("/me/velocity")
def my_velocity(db: ReadDbSession, user: CurrentUserDep) -> dict[str, Any]:
    """7.7: Self-reflective Research Velocity Dashboard metrics."""
    from app.modules.insights.service import InsightsService

    return InsightsService(db).research_velocity(user["id"])


@router.patch("/me", response_model=UserProfileOut)
def update_me(db: DbSession, user: CurrentUserDep, payload: ProfileUpdate) -> Any:
    """SettingsView sends name/title/institution/bio — `name` maps to User.full_name."""
    row, profile = _load(db, user["id"])
    if payload.name is not None:
        row.full_name = payload.name
    if profile is None:
        profile = UserProfile(user_id=row.id)
        db.add(profile)
        db.flush()
    for field in ("title", "institution", "field", "bio", "avatar", "interests"):
        value = getattr(payload, field)
        if value is not None:
            setattr(profile, field, value)
    db.commit()
    db.refresh(row)
    return AuthService.profile_out(row, profile)


@router.get("/{user_id}", response_model=UserProfileOut)
def get_user(user_id: str, db: ReadDbSession, user: CurrentUserDep) -> Any:
    row, profile = _load(db, user_id)
    return AuthService.profile_out(row, profile)


@router.patch("/{user_id}/verified", response_model=UserProfileOut)
def set_verified(
    user_id: str,
    db: DbSession,
    _: Annotated[Any, Depends(require("decide_verification"))],
    verified: bool = Body(..., embed=True),
) -> Any:
    row, profile = _load(db, user_id)
    row.verified = verified
    db.commit()
    db.refresh(row)
    return AuthService.profile_out(row, profile)


verification_router = APIRouter(prefix="/verification-requests", tags=["verification"])


@verification_router.post("", response_model=VerificationRequestOut, status_code=201)
def request_verification(
    payload: VerificationRequestIn, db: DbSession, user: CurrentUserDep
) -> Any:
    return AuthService(db).request_verification(user["id"], payload)


@verification_router.get("/{request_id}", response_model=VerificationRequestOut)
def get_request(request_id: str, db: ReadDbSession, user: CurrentUserDep) -> Any:
    return AuthService(db).get_verification(request_id)


@verification_router.post("/{request_id}/approve", response_model=VerificationRequestOut)
def approve(
    request_id: str,
    db: DbSession,
    _: Annotated[Any, Depends(require("decide_verification"))],
) -> Any:
    return AuthService(db).decide(request_id, True)


@verification_router.post("/{request_id}/reject", response_model=VerificationRequestOut)
def reject(
    request_id: str,
    db: DbSession,
    _: Annotated[Any, Depends(require("decide_verification"))],
) -> Any:
    return AuthService(db).decide(request_id, False)


__all__ = ["router", "verification_router"]
