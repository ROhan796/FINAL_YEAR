"""Account data access."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.social import ResearcherProfile
from app.models.user import User, UserProfile, VerificationRequest


class AuthRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def by_email(self, email: str) -> User | None:
        return self.db.scalar(select(User).where(User.email == email.lower().strip()))

    def get(self, user_id: str) -> User | None:
        return self.db.get(User, user_id)

    def profile(self, user_id: str) -> UserProfile | None:
        return self.db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))

    def add(self, user: User) -> User:
        self.db.add(user)
        return user

    def add_profile(self, profile: UserProfile) -> UserProfile:
        self.db.add(profile)
        return profile

    def add_researcher(self, researcher: ResearcherProfile) -> ResearcherProfile:
        self.db.add(researcher)
        return researcher

    def open_verification(self, user_id: str) -> VerificationRequest | None:
        return self.db.scalar(
            select(VerificationRequest).where(
                VerificationRequest.user_id == user_id,
                VerificationRequest.status == "pending",
            )
        )

    def verification(self, request_id: str) -> VerificationRequest | None:
        return self.db.get(VerificationRequest, request_id)

    def add_verification(self, req: VerificationRequest) -> VerificationRequest:
        self.db.add(req)
        return req
