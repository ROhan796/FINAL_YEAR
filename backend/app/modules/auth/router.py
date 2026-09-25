"""Auth endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body

from app.core.deps import DbSession, ReadDbSession
from app.core.errors import NotFoundError
from app.core.config import settings
from app.modules.auth.service import AuthService
from app.schemas import (
    TokenPair,
    UserCreate,
    UserLogin,
    VerificationRequestIn,
    VerificationRequestOut,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenPair, status_code=201)
def register(payload: UserCreate, db: DbSession) -> Any:
    return AuthService(db).register(payload)


@router.post("/login", response_model=TokenPair)
def login(payload: UserLogin, db: DbSession) -> Any:
    return AuthService(db).login(payload.email, payload.password)


@router.post("/refresh")
def refresh(db: DbSession, refreshToken: str = Body(embed=True)) -> Any:  # noqa: N803 - contract name
    return AuthService(db).refresh(refreshToken)


@router.post("/dev-login", response_model=TokenPair)
def dev_login(db: DbSession) -> Any:
    """Seeded researcher token. Disabled in production."""
    return AuthService(db).dev_login()


@router.get("/config")
def auth_config() -> Any:
    """Lets the frontend decide whether to show a login screen."""
    return {
        "devAutoLogin": settings.dev_auto_login,
        "passwordAuth": not settings.is_production or bool(settings.jwt_secret),
        "environment": settings.app_env,
    }


__all__ = ["router", "NotFoundError", "VerificationRequestOut"]
