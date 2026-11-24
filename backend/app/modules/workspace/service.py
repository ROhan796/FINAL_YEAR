"""Workspace business rules: projects, roles, tasks, notes, analytics, AI thread."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from app.core.errors import ForbiddenError, NotFoundError
from app.core.pagination import Page, PageParams
from app.models.project import (
    Project,
    ProjectChatMessage,
    ProjectDataset,
    ProjectDiscussion,
    ProjectMember,
    ProjectNote,
    ProjectPaper,
    ProjectTask,
)
from app.modules.workspace.repository import WorkspaceRepository
from app.schemas import (
    ActivityPoint,
    ChatTurn,
    DatasetOut,
    DiscussionOut,
    MemberOut,
    NoteOut,
    ProjectAnalytics,
    ProjectCreate,
    TaskOut,
    WorkspaceProjectOut,
)

AVATAR_PLACEHOLDER = "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=400&auto=format&fit=crop&q=80"


def _date_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _humanize(dt: datetime | None) -> str:
    if dt is None:
        return "Just now"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    seconds = max(0, int((datetime.now(UTC) - dt).total_seconds()))
    if seconds < 60:
        return "Just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


class WorkspaceService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo = WorkspaceRepository(db)

    # ── projects ────────────────────────────────────────────────────────

    def list_projects(self, user_id: str, page: PageParams) -> Page[WorkspaceProjectOut]:
        rows = self.repo.list_for(user_id)
        visible = rows[: page.limit]
        return Page[WorkspaceProjectOut](
            items=[self._out(p, user_id) for p in visible],
            nextCursor=None,
            total=len(rows),
        )

    def get_project(self, project_id: str, user_id: str) -> Project:
        project = self.repo.get(project_id)
        if project is None:
            raise NotFoundError(f"Project '{project_id}' not found")
        _ = user_id
        return project

    def _require(self, project_id: str, user_id: str, minimum: str) -> Project:
        project = self.get_project(project_id, user_id)
        role = self._role(project, user_id)
        if not WorkspaceRepository.allows(role, minimum):
            raise ForbiddenError(f"Role '{role}' cannot perform this action (needs {minimum})")
        return project

    @staticmethod
    def _role(project: Project, user_id: str) -> str:
        member = next((m for m in project.members if m.user_id == user_id), None)
        if member is not None:
            return member.role
        return "Owner" if project.owner_id == user_id else "Viewer"

    def create_project(self, user_id: str, author: dict[str, Any], payload: ProjectCreate) -> WorkspaceProjectOut:
        project = Project(
            owner_id=user_id,
            name=payload.name,
            description=payload.description,
            field=payload.field,
            owner_name=author.get("name", ""),
        )
        self.db.add(project)
        self.db.flush()  # project.id now exists, so the member FK resolves

        member = ProjectMember(
            project_id=project.id,
            user_id=user_id,
            name=author.get("name", ""),
            email=author.get("email", ""),
            avatar=author.get("avatar", AVATAR_PLACEHOLDER),
            role="Owner",
        )
        project.members.append(member)
        self.db.add(member)
        self.db.commit()
        self.db.refresh(project)
        return self._out(project, user_id)

    def delete_project(self, project_id: str, user_id: str) -> None:
        project = self._require(project_id, user_id, "Owner")
        self.db.delete(project)
        self.db.commit()

    def invite(self, project_id: str, user_id: str, email: str, role: str) -> WorkspaceProjectOut:
        project = self._require(project_id, user_id, "Owner")
        if any(m.email == email for m in project.members):
            from app.core.errors import ConflictError

            raise ConflictError("That person is already a member")
        self.repo.add(
            ProjectMember(
                project_id=project.id,
                name=email.split("@")[0].replace(".", " ").title(),
                email=email,
                avatar=AVATAR_PLACEHOLDER,
                role=role,
            )
        )
        self.db.commit()
        self._notify_invite(project, email, user_id)
        self.db.refresh(project)
        return self._out(project, user_id)

    def remove_member(self, project_id: str, user_id: str, email: str) -> WorkspaceProjectOut:
        project = self._require(project_id, user_id, "Owner")
        member = next((m for m in project.members if m.email == email), None)
        if member is None:
            raise NotFoundError("Member not found")
        self.db.delete(member)
        self.db.commit()
        self.db.refresh(project)
        return self._out(project, user_id)

    # ── papers ──────────────────────────────────────────────────────────

    def add_paper(self, project_id: str, user_id: str, paper_id: str) -> WorkspaceProjectOut:
        project = self._require(project_id, user_id, "Editor")
        if paper_id not in self.repo.paper_ids(project.id):
            from app.models.paper import Paper

            if self.db.get(Paper, paper_id) is None:
                raise NotFoundError(f"Paper '{paper_id}' not found")
            self.repo.add(ProjectPaper(project_id=project.id, paper_id=paper_id, added_by=user_id))
            self.db.commit()
        self.db.refresh(project)
        return self._out(project, user_id)

    def remove_paper(self, project_id: str, user_id: str, paper_id: str) -> WorkspaceProjectOut:
        from sqlalchemy import delete

        project = self._require(project_id, user_id, "Editor")
        self.db.execute(
            delete(ProjectPaper).where(
                ProjectPaper.project_id == project.id, ProjectPaper.paper_id == paper_id
            )
        )
        self.db.commit()
        self.db.refresh(project)
        return self._out(project, user_id)

    # ── tasks ───────────────────────────────────────────────────────────

    def create_task(
        self, project_id: str, user_id: str, title: str, due_date: str | None, assignee: str | None
    ) -> TaskOut:
        project = self._require(project_id, user_id, "Editor")
        member = next((m for m in project.members if m.user_id == user_id), None)
        parsed: date | None = None
        if due_date:
            try:
                parsed = date.fromisoformat(due_date)
            except ValueError:
                parsed = None
        task = ProjectTask(
            project_id=project.id,
            title=title,
            due_date=parsed,
            assignee_name=assignee or (member.name if member else project.owner_name),
            assignee_avatar=member.avatar if member else AVATAR_PLACEHOLDER,
            assignee_id=user_id,
            status="Todo",
            position=len(project.tasks),
        )
        self.repo.add(task)
        self.db.commit()
        self.db.refresh(task)
        return TaskOut(
            id=task.id,
            title=task.title,
            assigneeName=task.assignee_name,
            assigneeAvatar=task.assignee_avatar,
            dueDate=_date_str(task.due_date) or "2026-08-30",
            status=task.status,  # type: ignore[arg-type]
        )

    def patch_task(self, project_id: str, user_id: str, task_id: str, **fields: Any) -> TaskOut:
        self._require(project_id, user_id, "Editor")
        task = self.repo.task(project_id, task_id)
        if task is None:
            raise NotFoundError("Task not found")
        if fields.get("status") is not None:
            task.status = fields["status"]
        if fields.get("title"):
            task.title = fields["title"]
        if fields.get("dueDate"):
            try:
                task.due_date = date.fromisoformat(fields["dueDate"])
            except (ValueError, TypeError):
                pass
        self.db.commit()
        self.db.refresh(task)
        return TaskOut(
            id=task.id,
            title=task.title,
            assigneeName=task.assignee_name,
            assigneeAvatar=task.assignee_avatar,
            dueDate=_date_str(task.due_date) or "2026-08-30",
            status=task.status,  # type: ignore[arg-type]
        )

    def delete_task(self, project_id: str, user_id: str, task_id: str) -> None:
        self._require(project_id, user_id, "Editor")
        task = self.repo.task(project_id, task_id)
        if task is None:
            raise NotFoundError("Task not found")
        self.db.delete(task)
        self.db.commit()

    # ── notes ───────────────────────────────────────────────────────────

    def create_note(
        self, project_id: str, user_id: str, title: str, content: str, paper_id: str | None
    ) -> NoteOut:
        project = self._require(project_id, user_id, "Editor")
        paper_title = None
        if paper_id:
            from app.models.paper import Paper

            paper = self.db.get(Paper, paper_id)
            if paper is None:
                raise NotFoundError(f"Paper '{paper_id}' not found")
            paper_title = paper.title
        member = next((m for m in project.members if m.user_id == user_id), None)
        note = ProjectNote(
            project_id=project.id,
            author_id=user_id,
            title=title,
            content=content,
            paper_id=paper_id,
            paper_title=paper_title,
            author_name=member.name if member else project.owner_name,
        )
        self.repo.add(note)
        self.db.commit()
        self.db.refresh(note)
        return NoteOut(
            id=note.id,
            title=note.title,
            content=note.content,
            paperId=note.paper_id,
            paperTitle=note.paper_title,
            updatedAt=_date_str(note.created_at),
            authorName=note.author_name,
        )

    def patch_note(self, project_id: str, user_id: str, note_id: str, **fields: Any) -> NoteOut:
        self._require(project_id, user_id, "Editor")
        note = self.repo.note(project_id, note_id)
        if note is None:
            raise NotFoundError("Note not found")
        if fields.get("title"):
            note.title = fields["title"]
        if fields.get("content"):
            note.content = fields["content"]
        self.db.commit()
        self.db.refresh(note)
        return NoteOut(
            id=note.id,
            title=note.title,
            content=note.content,
            paperId=note.paper_id,
            paperTitle=note.paper_title,
            updatedAt=_date_str(note.created_at),
            authorName=note.author_name,
        )

    def delete_note(self, project_id: str, user_id: str, note_id: str) -> None:
        self._require(project_id, user_id, "Editor")
        note = self.repo.note(project_id, note_id)
        if note is None:
            raise NotFoundError("Note not found")
        self.db.delete(note)
        self.db.commit()

    # ── discussions ─────────────────────────────────────────────────────

    def add_discussion(self, project_id: str, user_id: str, content: str) -> DiscussionOut:
        project = self._require(project_id, user_id, "Editor")
        member = next((m for m in project.members if m.user_id == user_id), None)
        msg = ProjectDiscussion(
            project_id=project.id,
            author_id=user_id,
            author_name=member.name if member else project.owner_name,
            author_avatar=member.avatar if member else AVATAR_PLACEHOLDER,
            content=content.strip(),
        )
        self.repo.add(msg)
        self.db.commit()
        self.db.refresh(msg)
        return DiscussionOut(
            id=msg.id,
            authorName=msg.author_name,
            authorAvatar=msg.author_avatar,
            content=msg.content,
            timestamp=_humanize(msg.created_at),
        )

    def delete_discussion(self, project_id: str, user_id: str, msg_id: str) -> None:
        self._require(project_id, user_id, "Editor")
        msg = self.repo.discussion(project_id, msg_id)
        if msg is None:
            raise NotFoundError("Message not found")
        if msg.author_id not in (user_id, None) and self._role(self.get_project(project_id, user_id), user_id) != "Owner":
            raise ForbiddenError("You can only delete your own messages")
        self.db.delete(msg)
        self.db.commit()

    # ── datasets ────────────────────────────────────────────────────────

    def add_dataset(
        self, project_id: str, user_id: str, name: str, description: str, fmt: str, size: str, link: str
    ) -> DatasetOut:
        project = self._require(project_id, user_id, "Editor")
        ds = ProjectDataset(
            project_id=project.id, name=name, description=description,
            format=fmt, size=size, link=link, uploaded_by=user_id,
        )
        self.repo.add(ds)
        self.db.commit()
        self.db.refresh(ds)
        return self._out_dataset(ds)

    def delete_dataset(self, project_id: str, user_id: str, dataset_id: str) -> None:
        self._require(project_id, user_id, "Editor")
        ds = self.repo.dataset(project_id, dataset_id)
        if ds is None:
            raise NotFoundError("Dataset not found")
        self.db.delete(ds)
        self.db.commit()

    # ── analytics (replaces the hardcoded 31 hrs / chart) ───────────────

    def analytics(self, project_id: str, user_id: str) -> ProjectAnalytics:
        project = self.get_project(project_id, user_id)
        entries = self.repo.time_entries(project.id)
        by_day: dict[date, float] = {}
        for e in entries:
            by_day[e.day] = by_day.get(e.day, 0.0) + e.hours

        papers = self.repo.paper_ids(project.id)
        today = datetime.now(UTC).date()
        chart: list[ActivityPoint] = []
        for i in range(6, -1, -1):
            day = today - timedelta(days=i)
            chart.append(
                ActivityPoint(
                    date=day.strftime("%a"),
                    papersAdded=0,
                    notesCreated=sum(1 for n in project.notes if _date_str(n.created_at) == day.isoformat()),
                    tasksCompleted=sum(
                        1 for t in project.tasks if t.status == "Done" and _date_str(t.updated_at) == day.isoformat()
                    ),
                )
            )
        completed = [t for t in project.tasks if t.status == "Done"]
        return ProjectAnalytics(
            totalHours=round(sum(by_day.values()), 1),
            papersAdded=len(papers),
            notesCreated=len(project.notes),
            tasksCompleted=len(completed),
            tasksTotal=len(project.tasks),
            chart=chart,
        )

    # ── AI thread (server-owned history) ────────────────────────────────

    def chat_turns(self, project_id: str, user_id: str) -> list[ChatTurn]:
        self.get_project(project_id, user_id)
        return [
            ChatTurn(sender=row.sender, text=row.content)  # type: ignore[arg-type]
            for row in self.repo.chat(project_id)
        ]

    def ask(self, project_id: str, user_id: str, message: str) -> dict[str, Any]:
        project = self.get_project(project_id, user_id)
        papers = self.repo.papers(project.id)

        reply = ""
        cached = False
        try:
            from app.ai.service import chat_assistant

            reply = chat_assistant(
                message=message,
                project_context={
                    "id": project.id,
                    "name": project.name,
                    "field": project.field,
                    "description": project.description,
                    "paperIds": [p.id for p in papers],
                    "paperTitles": [p.title for p in papers],
                },
                conversation_history=[
                    {"sender": r.sender, "text": r.content} for r in self.repo.chat(project.id)[-10:]
                ],
                db=self.db,
            )
            cached = True
        except Exception:  # noqa: BLE001 - assistant degradation must not 500
            reply = (
                f'Regarding "{message}": start by checking whether the literature in '
                f'"{project.name}" already covers this angle, then record the open question in Notes.'
            )

        self.repo.add(ProjectChatMessage(project_id=project.id, user_id=user_id, sender="user", content=message))
        self.repo.add(
            ProjectChatMessage(project_id=project.id, user_id=user_id, sender="assistant", content=reply, cached=cached)
        )
        self.db.commit()
        return {"reply": reply, "cached": cached}

    def ask_stream(self, project_id: str, user_id: str, message: str):
        """Yield SSE deltas from RAG-grounded chat_assistant_stream and persist both turns."""
        from app.ai.service import _collect_rag_chunks, chat_assistant_stream

        project = self.get_project(project_id, user_id)
        papers = self.repo.papers(project.id)
        proj_ctx = {
            "id": project.id,
            "name": project.name,
            "field": project.field,
            "description": project.description,
            "paperIds": [p.id for p in papers],
            "paperTitles": [p.title for p in papers],
        }
        history = [{"sender": r.sender, "text": r.content} for r in self.repo.chat(project.id)[-10:]]
        rag_chunks = _collect_rag_chunks(proj_ctx, None, message, db=self.db)

        buffered: list[str] = []
        yield {"ragSources": [c.get("metadata", {}) for c in rag_chunks[:5]]}
        for piece in chat_assistant_stream(
            message=message,
            project_context=proj_ctx,
            conversation_history=history,
            db=self.db,
        ):
            buffered.append(piece)
            yield {"delta": piece}

        full_reply = "".join(buffered).strip()
        self.repo.add(ProjectChatMessage(project_id=project.id, user_id=user_id, sender="user", content=message))
        self.repo.add(
            ProjectChatMessage(project_id=project.id, user_id=user_id, sender="assistant", content=full_reply, cached=False)
        )
        self.db.commit()
        yield {"done": True}

    def clear_chat(self, project_id: str, user_id: str) -> None:
        self.get_project(project_id, user_id)
        self.repo.clear_chat(project_id)
        self.db.commit()

    # ── Reading Room (7.6: sentence-level collaborative paper annotation) ─

    def reading_annotations(
        self, project_id: str, user_id: str, paper_id: str | None = None
    ) -> list[dict[str, Any]]:
        from sqlalchemy import select

        from app.models.project import ReadingAnnotation

        self.get_project(project_id, user_id)
        stmt = select(ReadingAnnotation).where(ReadingAnnotation.project_id == project_id)
        if paper_id:
            stmt = stmt.where(ReadingAnnotation.paper_id == paper_id)
        rows = self.db.scalars(stmt.order_by(ReadingAnnotation.sentence_index, ReadingAnnotation.created_at)).all()
        return [
            {
                "id": r.id,
                "projectId": r.project_id,
                "paperId": r.paper_id,
                "chunkId": r.chunk_id,
                "sentenceIndex": r.sentence_index,
                "highlightText": r.highlight_text,
                "comment": r.comment,
                "authorName": r.author_name,
                "authorAvatar": r.author_avatar,
                "timestamp": _humanize(r.created_at),
            }
            for r in rows
        ]

    def add_reading_annotation(
        self,
        project_id: str,
        user_id: str,
        *,
        paper_id: str,
        sentence_index: int,
        highlight_text: str,
        comment: str,
        chunk_id: str | None = None,
    ) -> dict[str, Any]:
        from app.models.project import ReadingAnnotation

        project = self._require(project_id, user_id, "Editor")
        member = next((m for m in project.members if m.user_id == user_id), None)
        ann = ReadingAnnotation(
            project_id=project_id,
            paper_id=paper_id,
            chunk_id=chunk_id,
            sentence_index=sentence_index,
            highlight_text=highlight_text[:1000],
            comment=comment.strip(),
            author_id=user_id,
            author_name=member.name if member else project.owner_name,
            author_avatar=member.avatar if member else AVATAR_PLACEHOLDER,
        )
        self.db.add(ann)
        self.db.commit()
        self.db.refresh(ann)
        return {
            "id": ann.id,
            "projectId": ann.project_id,
            "paperId": ann.paper_id,
            "chunkId": ann.chunk_id,
            "sentenceIndex": ann.sentence_index,
            "highlightText": ann.highlight_text,
            "comment": ann.comment,
            "authorName": ann.author_name,
            "authorAvatar": ann.author_avatar,
            "timestamp": _humanize(ann.created_at),
        }

    def delete_reading_annotation(self, project_id: str, user_id: str, ann_id: str) -> None:
        from app.models.project import ReadingAnnotation

        self._require(project_id, user_id, "Editor")
        ann = self.db.get(ReadingAnnotation, ann_id)
        if ann is None or ann.project_id != project_id:
            raise NotFoundError("Annotation not found")
        self.db.delete(ann)
        self.db.commit()

    # ── serialisation ───────────────────────────────────────────────────

    def _out(self, project: Project, viewer_id: str) -> WorkspaceProjectOut:
        paper_ids = self.repo.paper_ids(project.id)
        member_out = [
            MemberOut(name=m.name, role=m.role, email=m.email, avatar=m.avatar)  # type: ignore[arg-type]
            for m in project.members
        ]
        tasks = [
            TaskOut(
                id=t.id, title=t.title, assigneeName=t.assignee_name,
                assigneeAvatar=t.assignee_avatar, dueDate=_date_str(t.due_date) or "2026-08-30",
                status=t.status,  # type: ignore[arg-type]
            )
            for t in sorted(project.tasks, key=lambda x: x.position)
        ]
        notes = [
            NoteOut(
                id=n.id, title=n.title, content=n.content, paperId=n.paper_id,
                paperTitle=n.paper_title, updatedAt=_date_str(n.created_at), authorName=n.author_name,
            )
            for n in project.notes
        ]
        discussions = [
            DiscussionOut(
                id=d.id, authorName=d.author_name, authorAvatar=d.author_avatar,
                content=d.content, timestamp=_humanize(d.created_at),
            )
            for d in project.discussions
        ]
        datasets = [self._out_dataset(d) for d in project.datasets]
        milestones = [
            {"id": t.id, "title": t.title, "date": t.dueDate, "completed": t.status == "Done"}
            for t in tasks[:6]
        ]
        completed = sum(1 for t in tasks if t.status == "Done")
        return WorkspaceProjectOut(
            id=project.id,
            name=project.name,
            description=project.description,
            field=project.field,
            status=project.status,  # type: ignore[arg-type]
            ownerName=project.owner_name,
            members=member_out,
            paperIds=paper_ids,
            notes=notes,
            tasks=tasks,
            milestones=milestones,
            datasets=datasets,
            documents=[],
            discussions=discussions,
            activityHistory=[
                ActivityPoint(date="Now", papersAdded=len(paper_ids), notesCreated=len(notes), tasksCompleted=completed)
            ],
        )

    @staticmethod
    def _out_dataset(d: ProjectDataset) -> DatasetOut:
        return DatasetOut(
            id=d.id,
            name=d.name,
            description=d.description,
            size=d.size,
            format=d.format,
            link=d.link,
            uploadedDate=_date_str(d.created_at),
        )

    def _notify_invite(self, project: Project, email: str, actor_id: str) -> None:
        try:
            from sqlalchemy import select

            from app.models.user import User
            from app.modules.notifications.service import create_notification

            invitee = self.db.scalar(select(User).where(User.email == email.lower()))
            if invitee is None:
                return
            actor = self.db.get(User, actor_id)
            create_notification(
                self.db,
                invitee.id,
                "collaboration",
                "Project invitation",
                f"{actor.full_name if actor else 'A collaborator'} added you to {project.name}.",
                link=f"/workspace?project={project.id}",
                actor_id=actor_id,
                entity_id=project.id,
            )
        except Exception:  # noqa: BLE001
            pass


__all__ = ["WorkspaceService"]
