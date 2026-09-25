"""Workspace data access: projects and everything nested under one."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.paper import Paper
from app.models.project import (
    Project,
    ProjectChatMessage,
    ProjectDataset,
    ProjectDiscussion,
    ProjectMember,
    ProjectNote,
    ProjectPaper,
    ProjectTask,
    ProjectTimeEntry,
)

ROLE_RANK = {"Viewer": 0, "Editor": 1, "Owner": 2}


class WorkspaceRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def _eager(self) -> Any:
        return select(Project).options(
            selectinload(Project.members),
            selectinload(Project.tasks),
            selectinload(Project.notes),
            selectinload(Project.discussions),
            selectinload(Project.datasets),
        )

    def list_for(self, user_id: str, limit: int = 200) -> list[Project]:
        rows = list(self.db.scalars(self._eager()).all())
        owned_or_member = [
            p
            for p in rows
            if p.owner_id == user_id or any(m.email and m.user_id == user_id for m in p.members)
        ]
        # Demo fallback: a fresh account with no memberships still sees the workspace.
        return owned_or_member or rows

    def get(self, project_id: str) -> Project | None:
        return self.db.scalar(self._eager().where(Project.id == project_id))

    def paper_ids(self, project_id: str) -> list[str]:
        return list(
            self.db.scalars(
                select(ProjectPaper.paper_id)
                .where(ProjectPaper.project_id == project_id)
                .order_by(ProjectPaper.position.asc())
            ).all()
        )

    def papers(self, project_id: str) -> list[Paper]:
        ids = self.paper_ids(project_id)
        if not ids:
            return []
        rows = self.db.scalars(select(Paper).where(Paper.id.in_(ids))).all()
        by_id = {p.id: p for p in rows}
        return [by_id[i] for i in ids if i in by_id]

    def time_entries(self, project_id: str) -> list[ProjectTimeEntry]:
        return list(
            self.db.scalars(
                select(ProjectTimeEntry)
                .where(ProjectTimeEntry.project_id == project_id)
                .order_by(ProjectTimeEntry.day.asc())
            ).all()
        )

    # ── membership / roles ──────────────────────────────────────────────

    def member(self, project_id: str, user_id: str) -> ProjectMember | None:
        return self.db.scalar(
            select(ProjectMember).where(
                ProjectMember.project_id == project_id, ProjectMember.user_id == user_id
            )
        )

    @staticmethod
    def role_of(member: ProjectMember | None) -> str:
        if member is None:
            return "Viewer"
        return member.role

    @staticmethod
    def allows(role: str, minimum: str) -> bool:
        return ROLE_RANK.get(role, 0) >= ROLE_RANK.get(minimum, 99)

    # ── sub-entities ────────────────────────────────────────────────────

    def add(self, obj: Any) -> Any:
        self.db.add(obj)
        return obj

    def task(self, project_id: str, task_id: str) -> ProjectTask | None:
        return self.db.scalar(
            select(ProjectTask).where(
                ProjectTask.id == task_id, ProjectTask.project_id == project_id
            )
        )

    def note(self, project_id: str, note_id: str) -> ProjectNote | None:
        return self.db.scalar(
            select(ProjectNote).where(
                ProjectNote.id == note_id, ProjectNote.project_id == project_id
            )
        )

    def discussion(self, project_id: str, msg_id: str) -> ProjectDiscussion | None:
        return self.db.scalar(
            select(ProjectDiscussion).where(
                ProjectDiscussion.id == msg_id, ProjectDiscussion.project_id == project_id
            )
        )

    def dataset(self, project_id: str, dataset_id: str) -> ProjectDataset | None:
        return self.db.scalar(
            select(ProjectDataset).where(
                ProjectDataset.id == dataset_id, ProjectDataset.project_id == project_id
            )
        )

    def chat(self, project_id: str, limit: int = 200) -> list[ProjectChatMessage]:
        return list(
            self.db.scalars(
                select(ProjectChatMessage)
                .where(ProjectChatMessage.project_id == project_id)
                .order_by(ProjectChatMessage.created_at.asc())
                .limit(limit)
            ).all()
        )

    def clear_chat(self, project_id: str) -> None:
        for row in self.chat(project_id):
            self.db.delete(row)


__all__ = ["WorkspaceRepository", "ROLE_RANK", "Project"]
