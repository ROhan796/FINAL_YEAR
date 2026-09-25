"""Business rules for notifications. Owns transactions; router stays thin.

`create_notification` is the public fan-out API other modules import — it is a
plain function (not a method) so a caller never has to construct a service just
to emit an event.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.core.errors import NotFoundError, ValidationAppError
from app.core.pagination import Cursor, Page, PageParams
from app.models.notification import Notification
from app.modules.notifications.repository import NotificationRepository
from app.schemas import NotificationOut

#: Mirrors `NotificationType`. An unknown type would fail response validation
#: for the whole inbox on read, so callers are rejected when the row is written.
NOTIFICATION_TYPES = frozenset(
    {"mention", "comment", "collaboration", "paper_update", "verification"}
)


def humanize_timestamp(value: datetime | None) -> str:
    """`Just now` / `2h ago` style ages.

    The frontend renders the string verbatim, so relative text is what keeps an
    old notification from reading like it happened today.
    """
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    seconds = (datetime.now(UTC) - value).total_seconds()
    if seconds < 60:
        return "Just now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 7:
        return f"{days}d ago"
    weeks = days // 7
    if weeks < 5:
        return f"{weeks}w ago"
    months = days // 30
    if months < 12:
        return f"{months}mo ago"
    return f"{days // 365}y ago"


def create_notification(
    db: Session,
    user_id: str,
    type: str,
    title: str,
    message: str,
    link: str | None = None,
    actor_id: str | None = None,
    entity_id: str | None = None,
) -> Notification:
    """Emit one event for `user_id`. Import this from other modules.

    Commits on its own: notifications are fire-and-forget side effects, so the
    caller invokes it as the last step of its own write and never has to think
    about transaction ownership.
    """
    if type not in NOTIFICATION_TYPES:
        raise ValidationAppError(f"Unknown notification type '{type}'")
    row = NotificationRepository(db).add(
        user_id=user_id,
        type=type,
        title=title,
        message=message,
        link=link,
        actor_id=actor_id,
        entity_id=entity_id,
    )
    db.commit()
    return row


class NotificationService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo = NotificationRepository(db)

    def list_notifications(
        self, user_id: str, unread_only: bool, page: PageParams
    ) -> Page[NotificationOut]:
        rows, total = self.repo.list_notifications(user_id, unread_only, page)
        has_more = len(rows) > page.limit
        visible = rows[:page.limit]
        next_cursor = None
        if has_more and visible:
            last = visible[-1]
            next_cursor = Cursor(k=last.created_at.isoformat(), i=last.id).encode()
        return Page(
            items=[self._out(row) for row in visible],
            nextCursor=next_cursor,
            total=total,
        )

    @staticmethod
    def _out(row: Notification) -> NotificationOut:
        return NotificationOut(
            id=row.id,
            type=row.type,  # type: ignore[arg-type] - validated against the literal
            title=row.title,
            message=row.message,
            timestamp=humanize_timestamp(row.created_at),
            read=row.read,
            link=row.link,
        )

    def unread_count(self, user_id: str) -> int:
        return self.repo.unread_count(user_id)

    def mark_read(self, user_id: str, notification_id: str) -> dict[str, Any]:
        row = self.repo.get(notification_id, user_id)
        if row is None:
            raise NotFoundError(f"Notification '{notification_id}' not found")
        if not row.read:
            self.repo.mark_read(row)
            self.db.commit()
        return {"ok": True}

    def read_all(self, user_id: str) -> dict[str, Any]:
        self.repo.read_all(user_id)
        self.db.commit()
        return {"ok": True}
