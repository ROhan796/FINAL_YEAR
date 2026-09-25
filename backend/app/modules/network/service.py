"""Network business rules: directory, profiles, follow/unfollow."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.core.pagination import Page, PageParams
from app.modules.network.repository import NetworkRepository
from app.schemas import PaperOut, ResearcherOut


class NetworkService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo = NetworkRepository(db)

    def list_researchers(
        self, viewer_id: str, field: str | None, q: str | None, page: PageParams
    ) -> Page[ResearcherOut]:
        rows, total = self.repo.list_researchers(field, q, page.limit + 1, 0)
        if not rows and not field and not q:
            rows = self.repo.synthesise_users(None, None, page.limit + 1)
            total = len(rows)

        has_more = len(rows) > page.limit
        visible = rows[: page.limit]
        followed = self.repo.followed_ids(viewer_id)
        items = [self._out(r, viewer_id, followed) for r in visible]
        next_cursor = None
        if has_more and visible:
            from app.core.pagination import Cursor

            next_cursor = Cursor(k=total, i=visible[-1].id).encode()
        return Page[ResearcherOut](items=items, nextCursor=next_cursor, total=total)

    def get(self, researcher_id: str, viewer_id: str) -> ResearcherOut:
        profile = self.repo.get_profile(researcher_id)
        if profile is None:
            raise NotFoundError(f"Researcher '{researcher_id}' not found")
        followed = self.repo.followed_ids(viewer_id)
        out = self._out(profile, viewer_id, followed)
        out.publications = [
            PaperOut.model_validate(p, from_attributes=True)
            for p in self.repo.publications(profile.name, limit=10)
        ]
        return out

    def toggle_follow(self, viewer_id: str, researcher_id: str, want: bool) -> dict[str, object]:
        if self.repo.get_profile(researcher_id) is None:
            raise NotFoundError(f"Researcher '{researcher_id}' not found")
        changed = self.repo.follow(viewer_id, researcher_id) if want else self.repo.unfollow(viewer_id, researcher_id)
        self.db.commit()
        count = self.repo.follower_count(researcher_id)
        if changed and want:
            self._notify(researcher_id, viewer_id)
        return {"ok": True, "isFollowed": self.repo.is_following(viewer_id, researcher_id), "followersCount": count}

    def followers(self, researcher_id: str, viewer_id: str) -> list[ResearcherOut]:
        followed = self.repo.followed_ids(viewer_id)
        return [self._out(p, viewer_id, followed) for p in self.repo.followers_of(researcher_id)]

    def following(self, researcher_id: str, viewer_id: str) -> list[ResearcherOut]:
        followed = self.repo.followed_ids(viewer_id)
        return [self._out(p, viewer_id, followed) for p in self.repo.following_of(researcher_id)]

    @staticmethod
    def _out(profile: object, viewer_id: str, followed: set[str]) -> ResearcherOut:
        p = profile  # duck-typed: ResearcherProfile instance
        return ResearcherOut(
            id=p.id,
            name=p.name,
            title=p.title,
            institution=p.institution,
            field=p.field,
            bio=p.bio,
            avatar=p.avatar,
            verified=p.verified,
            publicationsCount=p.publications_count,
            citationCount=p.citation_count,
            hIndex=p.h_index,
            followersCount=p.follower_count,
            interests=list(p.interests or []),
            publications=[],
            isFollowed=p.id in followed,
        )

    def _notify(self, researcher_id: str, actor_id: str) -> None:
        try:
            from sqlalchemy import select

            from app.models.user import User
            from app.modules.notifications.service import create_notification

            actor = self.db.get(User, actor_id)
            create_notification(
                self.db,
                researcher_id,
                "collaboration",
                "New follower",
                f"{actor.full_name if actor else 'A researcher'} started following you.",
                link="/network",
                actor_id=actor_id,
            )
        except Exception:  # noqa: BLE001 - notification failure must not break follow
            pass


__all__ = ["NetworkService"]
