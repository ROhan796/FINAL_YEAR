"""SQL-only data access for the notification inbox.

Rule: this layer never imports FastAPI or knows about HTTP. It also never
commits — transaction ownership belongs to the service.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.orm import Session

from app.core.pagination import PageParams
from app.models.notification import Notification


class NotificationRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    # ── reads ───────────────────────────────────────────────────────────

    def list_notifications(
        self, user_id: str, unread_only: bool, page: PageParams
    ) -> tuple[list[Notification], int]:
        base = select(Notification).where(Notification.user_id == user_id)
        if unread_only:
            base = base.where(Notification.read.is_(False))

        total = self.db.scalar(
            select(func.count()).select_from(base.subquery())
        ) or 0

        stmt = base.order_by(Notification.created_at.desc(), Notification.id.desc())
        cursor = page.cursor
        if cursor is not None and cursor.k and cursor.i:
            try:
                since = datetime.fromisoformat(str(cursor.k))
            except ValueError:
                since = None  # corrupt cursor → first page rather than a 500
            if since is not None:
                stmt = stmt.where(
                    or_(
                        Notification.created_at < since,
                        and_(Notification.created_at == since, Notification.id < cursor.i),
                    )
                )

        rows = list(self.db.scalars(stmt.limit(page.limit + 1)).all())
        return rows, int(total)

    def unread_count(self, user_id: str) -> int:
        return int(
            self.db.scalar(
                select(func.count())
                .select_from(Notification)
                .where(Notification.user_id == user_id, Notification.read.is_(False))
            )
            or 0
        )

    def get(self, notification_id: str, user_id: str) -> Notification | None:
        return self.db.scalar(
            select(Notification).where(
                Notification.id == notification_id, Notification.user_id == user_id
            )
        )

    # ── writes ──────────────────────────────────────────────────────────

    def add(
        self,
        *,
        user_id: str,
        type: str,
        title: str,
        message: str,
        link: str | None,
        actor_id: str | None,
        entity_id: str | None,
    ) -> Notification:
        row = Notification(
            user_id=user_id,
            type=type,
            title=title,
            message=message,
            link=link,
            actor_id=actor_id,
            entity_id=entity_id,
        )
        self.db.add(row)
        return row

    def mark_read(self, row: Notification) -> None:
        row.read = True
        self.db.add(row)

    def read_all(self, user_id: str) -> int:
        """Bulk UPDATE so a 10k-row inbox is one statement, not 10k flushes."""
        result = self.db.execute(
            update(Notification)
            .where(Notification.user_id == user_id, Notification.read.is_(False))
            .values(read=True)
        )
        return int(result.rowcount or 0)
