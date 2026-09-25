"""Authentication: register, login, refresh, dev-login, verification requests."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import ConflictError, NotFoundError, RateLimitError, UnauthorizedError
from app.core.security import (
    DEV_CLAIMS,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.modules.auth.repository import AuthRepository
from app.models.social import ResearcherProfile
from app.models.user import User, UserProfile, VerificationRequest
from app.schemas import TokenPair, UserProfileOut, UserCreate, VerificationRequestIn

#: In-process fallback when Redis is unavailable. Bounded so it cannot grow unbounded.
_attempts: dict[str, list[float]] = {}
_MAX_ATTEMPTS = 5
_WINDOW_SECONDS = 60


def _rate_limit(email: str) -> None:
    now = time.monotonic()
    hits = [t for t in _attempts.get(email, []) if now - t < _WINDOW_SECONDS]
    if len(hits) >= _MAX_ATTEMPTS:
        raise RateLimitError("Too many login attempts. Try again shortly.")
    hits.append(now)
    _attempts[email] = hits
    if len(_attempts) > 5000:  # pragma: no cover - prevents unbounded growth
        _attempts.clear()


class AuthService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo = AuthRepository(db)

    def register(self, payload: UserCreate) -> TokenPair:
        if self.repo.by_email(payload.email):
            raise ConflictError("An account with that email already exists")

        user = User(
            email=payload.email.lower().strip(),
            password_hash=hash_password(payload.password),
            full_name=payload.name,
            role="researcher",
            verified=False,
        )
        self.repo.add(user)
        self.db.flush()

        self.repo.add_profile(
            UserProfile(
                user_id=user.id,
                title=payload.title,
                institution=payload.institution,
                field=payload.field,
            )
        )
        self.repo.add_researcher(
            ResearcherProfile(
                user_id=user.id,
                name=payload.name,
                title=payload.title,
                institution=payload.institution,
                field=payload.field,
            )
        )
        self.db.commit()
        self.db.refresh(user)
        return self._tokens(user)

    def login(self, email: str, password: str) -> TokenPair:
        _rate_limit(email.lower().strip())
        user = self.repo.by_email(email)
        # Single message for both failure modes — no account enumeration.
        if user is None or not user.password_hash or not verify_password(password, user.password_hash):
            raise UnauthorizedError("Invalid email or password")
        if not user.is_active:
            raise UnauthorizedError("Account is disabled")
        user.last_login_at = datetime.now(UTC)
        self.db.commit()
        return self._tokens(user)

    def refresh(self, refresh_token: str) -> dict[str, str]:
        claims = decode_token(refresh_token)
        if claims.get("type") != "refresh":
            raise UnauthorizedError("Expected a refresh token")
        user = self.repo.get(str(claims.get("sub", "")))
        if user is None:
            raise UnauthorizedError("Account no longer exists")
        return {"accessToken": create_access_token(user.id, role=user.role, email=user.email)}

    def dev_login(self) -> TokenPair:
        if settings.is_production:
            raise NotFoundError("Not available in production")
        user = self.repo.get(DEV_CLAIMS["sub"])
        if user is None:
            user = User(
                id=DEV_CLAIMS["sub"],
                email=DEV_CLAIMS["email"],
                full_name=DEV_CLAIMS["name"],
                role="verified_researcher",
                verified=True,
                password_hash=None,
            )
            self.repo.add(user)
            self.db.flush()
            self.repo.add_profile(
                UserProfile(
                    user_id=user.id,
                    title="Principal AI Researcher",
                    institution="Independent Research Institute",
                    field="Computational Linguistics & AI Reasoning",
                    bio="Independent researcher focusing on multimodal reasoning architectures, adaptive transformer routing, and causal physics engines.",
                    avatar="https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400&auto=format&fit=crop&q=80",
                    interests=["Multimodal Transformers", "Causal AI", "Symbolic Reasoning", "Open Weights"],
                )
            )
            self.repo.add_researcher(
                ResearcherProfile(
                    user_id=user.id,
                    name=user.full_name,
                    title="Principal AI Researcher",
                    institution="Independent Research Institute",
                    field="Computational Linguistics & AI Reasoning",
                    verified=True,
                    avatar="https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400&auto=format&fit=crop&q=80",
                    interests=["Multimodal Transformers", "Causal AI", "Symbolic Reasoning"],
                    publications_count=42,
                    citation_count=1893,
                    h_index=21,
                )
            )
            self.db.commit()
            self.db.refresh(user)
        return self._tokens(user)

    def _tokens(self, user: User) -> TokenPair:
        profile = self.repo.profile(user.id)
        return TokenPair(
            accessToken=create_access_token(user.id, role=user.role, email=user.email),
            refreshToken=create_refresh_token(user.id, role=user.role),
            user=self.profile_out(user, profile),
        )

    @staticmethod
    def profile_out(user: User, profile: UserProfile | None) -> UserProfileOut:
        return UserProfileOut(
            id=user.id,
            name=user.full_name,
            title=profile.title if profile else "",
            institution=profile.institution if profile else "",
            field=profile.field if profile else "",
            bio=profile.bio if profile else "",
            avatar=profile.avatar if profile else "",
            verified=user.verified,
            interests=list(profile.interests) if profile and profile.interests else [],
            email=user.email,
        )

    # ── verification ────────────────────────────────────────────────────

    def request_verification(self, user_id: str, payload: VerificationRequestIn) -> dict[str, Any]:
        if self.repo.open_verification(user_id):
            raise ConflictError("A verification request is already pending")
        req = VerificationRequest(
            user_id=user_id,
            institutional_email=payload.institutionalEmail.lower(),
            orcid=payload.orcid,
            status="pending",
        )
        self.repo.add_verification(req)
        self.db.commit()
        self.db.refresh(req)
        return self.verification_out(req)

    def get_verification(self, request_id: str) -> dict[str, Any]:
        req = self.repo.verification(request_id)
        if req is None:
            raise NotFoundError(f"Verification request '{request_id}' not found")
        return self.verification_out(req)

    def decide(self, request_id: str, approve: bool) -> dict[str, Any]:
        req = self.repo.verification(request_id)
        if req is None:
            raise NotFoundError(f"Verification request '{request_id}' not found")
        req.status = "approved" if approve else "rejected"
        req.decided_at = datetime.now(UTC)
        user = self.repo.get(req.user_id)
        if user is not None and approve:
            user.verified = True
        self.db.commit()
        self.db.refresh(req)
        if user is not None:
            self._notify_decision(user, approve)
        return self.verification_out(req)

    @staticmethod
    def verification_out(req: VerificationRequest) -> dict[str, Any]:
        return {
            "id": req.id,
            "status": req.status,
            "institutionalEmail": req.institutional_email,
            "orcid": req.orcid,
            "createdAt": req.created_at,
            "decidedAt": req.decided_at,
        }

    def _notify_decision(self, user: User, approved: bool) -> None:
        try:
            from app.modules.notifications.service import create_notification

            create_notification(
                self.db,
                user.id,
                "verification",
                "Verification decision",
                "Your researcher verification was approved." if approved else "Your verification request was not approved.",
                link="/settings",
            )
        except Exception:  # noqa: BLE001
            pass


__all__ = ["AuthService", "TokenPair", "UserCreate", "decode_token"]
