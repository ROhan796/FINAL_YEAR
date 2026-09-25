"""Keyset pagination helpers.

OFFSET pagination degrades badly past a few thousand rows; every list endpoint
returns an opaque `nextCursor` instead.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class Cursor(BaseModel):
    """Opaque cursor payload. Keep it small and non-sensitive."""

    k: Any = None  # last sort key (id / created_at / score)
    i: str | None = None  # last id

    def encode(self) -> str:
        raw = json.dumps({"k": self.k, "i": self.i}, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    @classmethod
    def decode(cls, value: str | None) -> Cursor | None:
        if not value:
            return None
        try:
            padded = value + "=" * (-len(value) % 4)
            data = json.loads(base64.urlsafe_b64decode(padded.encode()))
            return cls(k=data.get("k"), i=data.get("i"))
        except (ValueError, TypeError, json.JSONDecodeError):
            return None


class Page(BaseModel, Generic[T]):
    items: list[T]
    nextCursor: str | None = None
    total: int | None = None


@dataclass(frozen=True)
class PageParams:
    limit: int = 20
    cursor: Cursor | None = None

    @property
    def has_more(self) -> bool:
        return self.cursor is not None


def clamp_limit(limit: int | None, default: int = 20, maximum: int = 100) -> int:
    if limit is None or limit <= 0:
        return default
    return min(limit, maximum)


def make_page(items: list[T], limit: int, keyset_last: tuple[Any, str] | None) -> Page[T]:
    """Build a page. `keyset_last` is present only when there is another row."""
    has_more = len(items) > limit
    visible = items[:limit]
    next_cursor = None
    if has_more and keyset_last is not None:
        next_cursor = Cursor(k=keyset_last[0], i=keyset_last[1]).encode()
    return Page(items=visible, nextCursor=next_cursor)
