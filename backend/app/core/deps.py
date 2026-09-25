"""Shared FastAPI dependencies: session, current user, capability guards, cache."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Annotated, Any

from fastapi import Depends, Header, Query, Request
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import ForbiddenError, UnauthorizedError
from app.core.security import DEV_CLAIMS, Role, decode_token, ensure_capability
from app.core.pagination import Cursor, PageParams, clamp_limit
from app.db.session import get_db, get_read_db

CurrentUser = dict[str, Any]

#: Keys present on every resolved caller. erified gates manuscript submission.
_USER_KEYS = ('id', 'role', 'email', 'name', 'verified')
DbSession = Annotated[Session, Depends(get_db)]
ReadDbSession = Annotated[Session, Depends(get_read_db)]


def _extract_bearer(authorization: str | None, x_api_key: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    if x_api_key:
        return x_api_key.strip()
    return None


def get_current_user(
    db: DbSession,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> CurrentUser:
    """Resolve the caller.

    Production requires a JWT. In `DEV_AUTO_LOGIN` the seeded researcher is
    returned so the current frontend (which has no login screen) works unchanged.
    """
    token = _extract_bearer(authorization, x_api_key)

    if token:
        claims = decode_token(token)
        if claims.get("type") != "access":
            raise UnauthorizedError("Expected an access token")
        return _hydrate(db, {
            "id": str(claims.get("sub", "")),
            "role": str(claims.get("role", Role.READER)),
            "email": str(claims.get("email", "")),
            "name": str(claims.get("name", "")),
        })

    if settings.dev_auto_login and not settings.is_production:
        return _hydrate(db, {
            "id": DEV_CLAIMS["sub"], "role": DEV_CLAIMS["role"],
            "email": DEV_CLAIMS["email"], "name": DEV_CLAIMS["name"],
        })

    raise UnauthorizedError("Missing bearer token")


def _hydrate(db: Session, user: dict[str, Any]) -> CurrentUser:
    """Fill profile-derived claims (erified, display name) from the row.

    A token minted before a verification decision must pick the new value up
    immediately, so this is read per-request rather than baked into the JWT.
    """
    from app.models.user import User

    role = user.get("role", Role.READER)
    row = db.get(User, user["id"])
    user["verified"] = bool(row.verified) if row is not None else (
        role in (Role.VERIFIED_RESEARCHER, Role.EDITOR, Role.REVIEWER, Role.ADMIN)
    )
    if row is not None and row.full_name:
        user["name"] = row.full_name
    if row is not None and row.email:
        user["email"] = row.email
    return user


def get_optional_user(
    db: DbSession,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> CurrentUser | None:
    token = _extract_bearer(authorization, x_api_key)
    if not token:
        if settings.dev_auto_login and not settings.is_production:
            return _hydrate(db, {
                "id": DEV_CLAIMS["sub"], "role": DEV_CLAIMS["role"],
                "email": DEV_CLAIMS["email"], "name": DEV_CLAIMS["name"],
            })
        return None
    claims = decode_token(token)
    return _hydrate(db, {
        "id": str(claims.get("sub", "")),
        "role": str(claims.get("role", Role.READER)),
        "email": str(claims.get("email", "")),
        "name": str(claims.get("name", "")),
    })


CurrentUserDep = Annotated[CurrentUser, Depends(get_current_user)]
OptionalUserDep = Annotated[CurrentUser | None, Depends(get_optional_user)]


def require(capability: str) -> Callable[[CurrentUser], CurrentUser]:
    """Dependency factory: `Depends(require('post_to_feed'))`."""

    def _guard(user: CurrentUserDep) -> CurrentUser:
        ensure_capability(user["role"], capability)
        return user

    return _guard


class Pagination:
    """`?cursor=&limit=` dependency."""

    def __init__(self, default: int = 20, maximum: int = 100) -> None:
        self.default = default
        self.maximum = maximum

    def __call__(
        self,
        cursor: Annotated[str | None, Query()] = None,
        limit: Annotated[int | None, Query(ge=1, le=200)] = None,
    ) -> PageParams:
        return PageParams(limit=clamp_limit(limit, self.default, self.maximum), cursor=Cursor.decode(cursor))


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def iterator_session() -> Iterator[Session]:  # pragma: no cover - scripts
    yield from get_db()
