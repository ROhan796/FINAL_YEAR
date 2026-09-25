"""Authentication, JWT issuing and role-based capability checks.

The shipped frontend has no login screen, so `DEV_AUTO_LOGIN` short-circuits to the
seeded researcher. Flip it off in staging/prod and the same code path requires a token.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings
from app.core.errors import ForbiddenError, UnauthorizedError

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Role(StrEnum):
    READER = "reader"
    RESEARCHER = "researcher"
    VERIFIED_RESEARCHER = "verified_researcher"
    EDITOR = "editor"
    REVIEWER = "reviewer"
    ADMIN = "admin"


#: Capability -> roles allowed. One place to audit what each portal can do.
CAPABILITIES: dict[str, set[Role]] = {
    "search": {Role.READER, Role.RESEARCHER, Role.VERIFIED_RESEARCHER, Role.EDITOR, Role.REVIEWER, Role.ADMIN},
    "browse": {Role.READER, Role.RESEARCHER, Role.VERIFIED_RESEARCHER, Role.EDITOR, Role.REVIEWER, Role.ADMIN},
    "save_papers": {Role.READER, Role.RESEARCHER, Role.VERIFIED_RESEARCHER, Role.EDITOR, Role.REVIEWER, Role.ADMIN},
    "follow": {Role.READER, Role.RESEARCHER, Role.VERIFIED_RESEARCHER, Role.EDITOR, Role.REVIEWER, Role.ADMIN},
    "comment": {Role.READER, Role.RESEARCHER, Role.VERIFIED_RESEARCHER, Role.EDITOR, Role.REVIEWER, Role.ADMIN},
    "create_collections": {Role.READER, Role.RESEARCHER, Role.VERIFIED_RESEARCHER, Role.EDITOR, Role.REVIEWER, Role.ADMIN},
    "create_projects": {Role.RESEARCHER, Role.VERIFIED_RESEARCHER, Role.EDITOR, Role.REVIEWER, Role.ADMIN},
    # Publication gating mirrors the frontend's `user.verified` interstitial.
    "post_to_feed": {Role.VERIFIED_RESEARCHER, Role.EDITOR, Role.ADMIN},
    "submit_manuscript": {Role.VERIFIED_RESEARCHER, Role.EDITOR, Role.ADMIN},
    "review_manuscript": {Role.REVIEWER, Role.EDITOR, Role.ADMIN},
    "decide_verification": {Role.ADMIN},
    "moderate": {Role.ADMIN},
    "manage_ingestion": {Role.ADMIN},
}

DEV_USER_ID = "current-user"

DEV_CLAIMS: dict[str, Any] = {
    "sub": DEV_USER_ID,
    "role": Role.VERIFIED_RESEARCHER.value,
    "email": "amara.nwosu@research-inst.org",
    "name": "Dr. Amara Nwosu",
}


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(plain, hashed)
    except ValueError:
        return False


def create_access_token(
    subject: str,
    *,
    role: str,
    email: str = "",
    expires_minutes: int | None = None,
) -> str:
    minutes = expires_minutes or settings.access_token_expire_minutes
    payload = {
        "sub": subject,
        "role": role,
        "email": email,
        "type": "access",
        "iat": datetime.now(UTC),
        "exp": datetime.now(UTC) + timedelta(minutes=minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_refresh_token(subject: str, *, role: str) -> str:
    payload = {
        "sub": subject,
        "role": role,
        "type": "refresh",
        "iat": datetime.now(UTC),
        "exp": datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise UnauthorizedError("Invalid or expired token") from exc


def ensure_capability(role: str | Role, capability: str) -> None:
    try:
        allowed = CAPABILITIES[capability]
    except KeyError as exc:
        raise ValueError(f"Unknown capability: {capability}") from exc
    if Role(role) not in allowed:
        raise ForbiddenError(f"Role '{role}' cannot perform '{capability}'")
