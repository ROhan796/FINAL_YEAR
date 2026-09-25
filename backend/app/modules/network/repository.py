"""Researcher directory + follow graph data access."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session

from app.models.paper import Paper
from app.models.social import Follow, ResearcherProfile
from app.models.user import User, UserProfile


class NetworkRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    # ── directory ───────────────────────────────────────────────────────

    def _profiles(self, field: str | None, q: str | None) -> Select[Any]:
        stmt = select(ResearcherProfile)
        if field:
            # Substring match — NetworkView filters on `r.field` contains chip.
            stmt = stmt.where(ResearcherProfile.field.ilike(f"%{field}%"))
        if q:
            like = f"%{q}%"
            stmt = stmt.where(
                or_(
                    ResearcherProfile.name.ilike(like),
                    ResearcherProfile.institution.ilike(like),
                    ResearcherProfile.title.ilike(like),
                )
            )
        return stmt

    def list_researchers(
        self, field: str | None, q: str | None, limit: int, offset: int
    ) -> tuple[list[ResearcherProfile], int]:
        stmt = self._profiles(field, q)
        total = int(self.db.scalar(select(func.count()).select_from(stmt.subquery())) or 0)
        rows = list(
            self.db.scalars(
                stmt.order_by(ResearcherProfile.verified.desc(), ResearcherProfile.follower_count.desc()).offset(offset).limit(limit)
            ).all()
        )
        return rows, total

    def synthesise_users(self, field: str | None, q: str | None, limit: int) -> list[ResearcherProfile]:
        """Every account is followable even before a ResearcherProfile exists."""
        stmt = select(User, UserProfile).join(UserProfile, UserProfile.user_id == User.id)
        if field:
            stmt = stmt.where(UserProfile.field.ilike(f"%{field}%"))
        if q:
            stmt = stmt.where(UserProfile.institution.ilike(f"%{q}%"))
        out: list[ResearcherProfile] = []
        for user, profile in self.db.execute(stmt.limit(limit)).all():
            out.append(
                ResearcherProfile(
                    id=user.id,
                    user_id=user.id,
                    name=user.full_name,
                    title=profile.title,
                    institution=profile.institution,
                    field=profile.field,
                    bio=profile.bio,
                    avatar=profile.avatar,
                    verified=user.verified,
                    interests=list(profile.interests or []),
                    publications_count=profile.publications_count,
                    citation_count=profile.citation_count,
                    h_index=profile.h_index,
                    follower_count=self.follower_count(user.id),
                )
            )
        return out

    def get_profile(self, researcher_id: str) -> ResearcherProfile | None:
        row = self.db.get(ResearcherProfile, researcher_id)
        if row is not None:
            return row
        user = self.db.get(User, researcher_id)
        if user is None:
            return None
        profile = self.db.scalar(select(UserProfile).where(UserProfile.user_id == user.id))
        return ResearcherProfile(
            id=user.id,
            user_id=user.id,
            name=user.full_name,
            title=profile.title if profile else "",
            institution=profile.institution if profile else "",
            field=profile.field if profile else "",
            bio=profile.bio if profile else "",
            avatar=profile.avatar if profile else "",
            verified=user.verified,
            interests=list(profile.interests) if profile and profile.interests else [],
            follower_count=self.follower_count(user.id),
        )

    def publications(self, name: str, limit: int = 10) -> list[Paper]:
        # JSONB containment is Postgres-only; fall back to a Python-side filter.
        try:
            rows = self.db.scalars(
                select(Paper).where(Paper.authors.contains([name])).order_by(Paper.citation_count.desc()).limit(limit)
            ).all()
            if rows:
                return list(rows)
        except Exception:  # noqa: BLE001
            self.db.rollback()
        return [
            p
            for p in self.db.scalars(
                select(Paper).order_by(Paper.citation_count.desc()).limit(200)
            ).all()
            if any(name.lower() in str(a).lower() for a in (p.authors or []))
        ][:limit]

    # ── follow graph ────────────────────────────────────────────────────

    def follower_count(self, researcher_id: str) -> int:
        return int(
            self.db.scalar(
                select(func.count()).select_from(Follow).where(Follow.followee_id == researcher_id)
            )
            or 0
        )

    def is_following(self, follower_id: str, followee_id: str) -> bool:
        return (
            self.db.scalar(
                select(Follow.id).where(
                    Follow.follower_id == follower_id, Follow.followee_id == followee_id
                )
            )
            is not None
        )

    def follow(self, follower_id: str, followee_id: str) -> bool:
        if follower_id == followee_id or self.is_following(follower_id, followee_id):
            return False
        self.db.add(Follow(follower_id=follower_id, followee_id=followee_id))
        return True

    def unfollow(self, follower_id: str, followee_id: str) -> bool:
        row = self.db.scalar(
            select(Follow).where(
                Follow.follower_id == follower_id, Follow.followee_id == followee_id
            )
        )
        if row is None:
            return False
        self.db.delete(row)
        return True

    def followers_of(self, researcher_id: str, limit: int = 100) -> list[ResearcherProfile]:
        ids = list(
            self.db.scalars(
                select(Follow.follower_id).where(Follow.followee_id == researcher_id).limit(limit)
            ).all()
        )
        return [p for i in ids if (p := self.get_profile(i)) is not None]

    def following_of(self, researcher_id: str, limit: int = 100) -> list[ResearcherProfile]:
        ids = list(
            self.db.scalars(
                select(Follow.followee_id).where(Follow.follower_id == researcher_id).limit(limit)
            ).all()
        )
        return [p for i in ids if (p := self.get_profile(i)) is not None]

    def followed_ids(self, user_id: str) -> set[str]:
        return set(
            self.db.scalars(select(Follow.followee_id).where(Follow.follower_id == user_id)).all()
        )
