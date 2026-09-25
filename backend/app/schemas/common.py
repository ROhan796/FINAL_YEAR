"""Shared response primitives."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class OkResponse(BaseModel):
    ok: bool = True


class PageResponse(BaseModel, Generic[T]):
    items: list[T]
    nextCursor: str | None = None
    total: int | None = None


class ErrorBody(BaseModel):
    detail: str
    code: str | None = None
    requestId: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    checks: dict[str, str] = Field(default_factory=dict)


class MetaResponse(BaseModel):
    version: str
    environment: str
    features: dict[str, bool]
