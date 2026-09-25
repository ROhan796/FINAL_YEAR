"""Workspace endpoints: projects and every entity nested under one."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query

from app.core.deps import CurrentUserDep, DbSession, PageParams, Pagination, ReadDbSession
from app.schemas import (
    ChatReply,
    ChatTurn,
    DatasetCreate,
    DatasetOut,
    DiscussionCreate,
    DiscussionOut,
    MemberInvite,
    NoteCreate,
    NoteOut,
    OkResponse,
    ProjectAnalytics,
    ProjectCreate,
    TaskCreate,
    TaskOut,
    TaskPatch,
    WorkspaceProjectOut,
)
from app.schemas.common import PageResponse
from app.modules.workspace.service import WorkspaceService

router = APIRouter(prefix="/projects", tags=["workspace"])


@router.get("", response_model=PageResponse[WorkspaceProjectOut])
def list_projects(
    db: DbSession,
    user: CurrentUserDep,
    page: Annotated[PageParams, Depends(Pagination(default=20))],
) -> Any:
    return WorkspaceService(db).list_projects(user["id"], page)


@router.post("", response_model=WorkspaceProjectOut, status_code=201)
def create_project(payload: ProjectCreate, db: DbSession, user: CurrentUserDep) -> Any:
    return WorkspaceService(db).create_project(user["id"], user, payload)


# ── a single project ────────────────────────────────────────────────────────


@router.get("/{project_id}", response_model=WorkspaceProjectOut)
def get_project(project_id: str, db: ReadDbSession, user: CurrentUserDep) -> Any:
    service = WorkspaceService(db)
    return service._out(service.get_project(project_id, user["id"]), user["id"])


@router.delete("/{project_id}", response_model=OkResponse)
def delete_project(project_id: str, db: DbSession, user: CurrentUserDep) -> Any:
    WorkspaceService(db).delete_project(project_id, user["id"])
    return OkResponse(ok=True)


# ── members ─────────────────────────────────────────────────────────────────


@router.post("/{project_id}/members", response_model=WorkspaceProjectOut)
def invite_member(
    project_id: str, payload: MemberInvite, db: DbSession, user: CurrentUserDep
) -> Any:
    return WorkspaceService(db).invite(project_id, user["id"], str(payload.email), str(payload.role))


@router.delete("/{project_id}/members/{email}", response_model=WorkspaceProjectOut)
def remove_member(project_id: str, email: str, db: DbSession, user: CurrentUserDep) -> Any:
    return WorkspaceService(db).remove_member(project_id, user["id"], email)


# ── papers ──────────────────────────────────────────────────────────────────


@router.post("/{project_id}/papers", response_model=WorkspaceProjectOut)
def add_paper(project_id: str, db: DbSession, user: CurrentUserDep, paperId: str = Body(embed=True)) -> Any:  # noqa: N803
    return WorkspaceService(db).add_paper(project_id, user["id"], paperId)


@router.delete("/{project_id}/papers/{paper_id}", response_model=WorkspaceProjectOut)
def remove_paper(project_id: str, paper_id: str, db: DbSession, user: CurrentUserDep) -> Any:
    return WorkspaceService(db).remove_paper(project_id, user["id"], paper_id)


# ── tasks ───────────────────────────────────────────────────────────────────


@router.get("/{project_id}/tasks", response_model=list[TaskOut])
def list_tasks(project_id: str, db: ReadDbSession, user: CurrentUserDep) -> Any:
    service = WorkspaceService(db)
    project = service.get_project(project_id, user["id"])
    return service._out(project, user["id"]).tasks


@router.post("/{project_id}/tasks", response_model=TaskOut, status_code=201)
def create_task(project_id: str, payload: TaskCreate, db: DbSession, user: CurrentUserDep) -> Any:
    return WorkspaceService(db).create_task(
        project_id, user["id"], payload.title, payload.dueDate, payload.assigneeName
    )


@router.patch("/{project_id}/tasks/{task_id}", response_model=TaskOut)
def patch_task(
    project_id: str, task_id: str, payload: TaskPatch, db: DbSession, user: CurrentUserDep
) -> Any:
    return WorkspaceService(db).patch_task(project_id, user["id"], task_id, **payload.model_dump())


@router.delete("/{project_id}/tasks/{task_id}", response_model=OkResponse)
def delete_task(project_id: str, task_id: str, db: DbSession, user: CurrentUserDep) -> Any:
    WorkspaceService(db).delete_task(project_id, user["id"], task_id)
    return OkResponse(ok=True)


# ── notes ───────────────────────────────────────────────────────────────────


@router.get("/{project_id}/notes", response_model=list[NoteOut])
def list_notes(project_id: str, db: ReadDbSession, user: CurrentUserDep) -> Any:
    service = WorkspaceService(db)
    project = service.get_project(project_id, user["id"])
    return service._out(project, user["id"]).notes


@router.post("/{project_id}/notes", response_model=NoteOut, status_code=201)
def create_note(project_id: str, payload: NoteCreate, db: DbSession, user: CurrentUserDep) -> Any:
    return WorkspaceService(db).create_note(
        project_id, user["id"], payload.title, payload.content, payload.paperId
    )


@router.patch("/{project_id}/notes/{note_id}", response_model=NoteOut)
def patch_note(
    project_id: str, note_id: str, db: DbSession, user: CurrentUserDep,
    title: str | None = Body(None, embed=True),
    content: str | None = Body(None, embed=True),
) -> Any:
    return WorkspaceService(db).patch_note(
        project_id, user["id"], note_id, title=title, content=content
    )


@router.delete("/{project_id}/notes/{note_id}", response_model=OkResponse)
def delete_note(project_id: str, note_id: str, db: DbSession, user: CurrentUserDep) -> Any:
    WorkspaceService(db).delete_note(project_id, user["id"], note_id)
    return OkResponse(ok=True)


# ── discussions ─────────────────────────────────────────────────────────────


@router.get("/{project_id}/discussions", response_model=list[DiscussionOut])
def list_discussions(project_id: str, db: ReadDbSession, user: CurrentUserDep) -> Any:
    service = WorkspaceService(db)
    project = service.get_project(project_id, user["id"])
    return service._out(project, user["id"]).discussions


@router.post("/{project_id}/discussions", response_model=DiscussionOut, status_code=201)
def create_discussion(
    project_id: str, payload: DiscussionCreate, db: DbSession, user: CurrentUserDep
) -> Any:
    return WorkspaceService(db).add_discussion(project_id, user["id"], payload.content)


@router.delete("/{project_id}/discussions/{msg_id}", response_model=OkResponse)
def delete_discussion(project_id: str, msg_id: str, db: DbSession, user: CurrentUserDep) -> Any:
    WorkspaceService(db).delete_discussion(project_id, user["id"], msg_id)
    return OkResponse(ok=True)


# ── datasets ────────────────────────────────────────────────────────────────


@router.get("/{project_id}/datasets", response_model=list[DatasetOut])
def list_datasets(project_id: str, db: ReadDbSession, user: CurrentUserDep) -> Any:
    service = WorkspaceService(db)
    project = service.get_project(project_id, user["id"])
    return service._out(project, user["id"]).datasets


@router.post("/{project_id}/datasets", response_model=DatasetOut, status_code=201)
def create_dataset(
    project_id: str, payload: DatasetCreate, db: DbSession, user: CurrentUserDep
) -> Any:
    return WorkspaceService(db).add_dataset(
        project_id, user["id"], payload.name, payload.description,
        payload.format, payload.size, payload.link,
    )


@router.delete("/{project_id}/datasets/{dataset_id}", response_model=OkResponse)
def delete_dataset(project_id: str, dataset_id: str, db: DbSession, user: CurrentUserDep) -> Any:
    WorkspaceService(db).delete_dataset(project_id, user["id"], dataset_id)
    return OkResponse(ok=True)


# ── analytics + AI thread ───────────────────────────────────────────────────


@router.get("/{project_id}/analytics", response_model=ProjectAnalytics)
def analytics(project_id: str, db: ReadDbSession, user: CurrentUserDep) -> Any:
    return WorkspaceService(db).analytics(project_id, user["id"])


@router.get("/{project_id}/ai-thread", response_model=list[ChatTurn])
def ai_thread(project_id: str, db: ReadDbSession, user: CurrentUserDep) -> Any:
    return WorkspaceService(db).chat_turns(project_id, user["id"])


@router.post("/{project_id}/ai-thread", response_model=ChatReply)
def ai_ask(
    project_id: str, db: DbSession, user: CurrentUserDep, message: str = Body(embed=True)
) -> Any:
    return WorkspaceService(db).ask(project_id, user["id"], message)


@router.delete("/{project_id}/ai-thread", response_model=OkResponse)
def ai_clear(project_id: str, db: DbSession, user: CurrentUserDep) -> Any:
    WorkspaceService(db).clear_chat(project_id, user["id"])
    return OkResponse(ok=True)


__all__ = ["router"]
