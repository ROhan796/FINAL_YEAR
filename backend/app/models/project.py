"""Workspace projects and everything nested under one: members, papers, notes,
tasks, datasets, documents, discussions, chat thread, time entries."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import JSONB_TYPE, Base, TimestampMixin, UUIDPrimaryKey


class Project(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "projects"

    owner_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    field: Mapped[str] = mapped_column(String(255), default="", nullable=False, index=True)
    # Active | Under Review | Archived
    status: Mapped[str] = mapped_column(String(40), default="Active", nullable=False)
    owner_name: Mapped[str] = mapped_column(String(200), default="", nullable=False)

    members: Mapped[list[ProjectMember]] = relationship(
        back_populates="project", cascade="all, delete-orphan", lazy="selectin"
    )
    tasks: Mapped[list[ProjectTask]] = relationship(
        back_populates="project", cascade="all, delete-orphan", lazy="selectin"
    )
    notes: Mapped[list[ProjectNote]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="ProjectNote.created_at.desc()"
    )
    discussions: Mapped[list[ProjectDiscussion]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="ProjectDiscussion.created_at"
    )
    datasets: Mapped[list[ProjectDataset]] = relationship(
        back_populates="project", cascade="all, delete-orphan", lazy="selectin"
    )


class ProjectMember(UUIDPrimaryKey, Base):
    __tablename__ = "project_members"
    __table_args__ = (UniqueConstraint("project_id", "email", name="uq_project_member"),)

    project_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    avatar: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    # Owner | Editor | Viewer  (Owner > Editor > Viewer)
    role: Mapped[str] = mapped_column(String(24), default="Viewer", nullable=False)

    project: Mapped[Project] = relationship(back_populates="members")


class ProjectPaper(UUIDPrimaryKey, Base):
    __tablename__ = "project_papers"
    __table_args__ = (UniqueConstraint("project_id", "paper_id", name="uq_project_paper"),)

    project_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    paper_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    added_by: Mapped[str | None] = mapped_column(String(64))
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class ProjectNote(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "project_notes"

    project_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    author_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("users.id", ondelete="SET NULL"))
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    paper_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("papers.id", ondelete="SET NULL"), index=True
    )
    paper_title: Mapped[str | None] = mapped_column(Text)
    author_name: Mapped[str] = mapped_column(String(200), default="", nullable=False)

    project: Mapped[Project] = relationship(back_populates="notes")


class ProjectTask(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "project_tasks"

    project_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    assignee_name: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    assignee_avatar: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    assignee_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("users.id", ondelete="SET NULL"))
    due_date: Mapped[date | None] = mapped_column(Date)
    # Todo | In Progress | Done
    status: Mapped[str] = mapped_column(String(24), default="Todo", nullable=False, index=True)
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    project: Mapped[Project] = relationship(back_populates="tasks")


class ProjectDataset(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "project_datasets"

    project_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    format: Mapped[str] = mapped_column(String(60), default="", nullable=False)
    size: Mapped[str] = mapped_column(String(40), default="", nullable=False)
    link: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    uploaded_by: Mapped[str | None] = mapped_column(String(64))

    project: Mapped[Project] = relationship(back_populates="datasets")


class ProjectDocument(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "project_documents"

    project_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    # Draft | AI Pre-Reviewed | Submitted | Published
    status: Mapped[str] = mapped_column(String(40), default="Draft", nullable=False)
    ai_score: Mapped[int | None] = mapped_column(Integer)
    manuscript_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("manuscripts.id", ondelete="SET NULL"), index=True
    )


class ProjectDiscussion(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "project_discussions"

    project_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    author_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("users.id", ondelete="SET NULL"))
    author_name: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    author_avatar: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    project: Mapped[Project] = relationship(back_populates="discussions")


class ProjectChatMessage(UUIDPrimaryKey, TimestampMixin, Base):
    """Server-owned AI assistant history — survives tab switches and replica moves."""

    __tablename__ = "project_chat_messages"

    project_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("users.id", ondelete="SET NULL"))
    # user | assistant
    sender: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    cached: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    __table_args__ = (Index("ix_project_chat_project_created", "project_id", "created_at"),)


class ProjectTimeEntry(UUIDPrimaryKey, Base):
    """Replaces the hardcoded '31 hrs' + chart data in the Overview tab."""

    __tablename__ = "project_time_entries"

    project_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("users.id", ondelete="SET NULL"))
    day: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    hours: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    note: Mapped[str | None] = mapped_column(Text)
