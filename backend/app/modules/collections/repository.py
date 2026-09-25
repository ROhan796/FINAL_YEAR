"""Collections data access."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.collection import Collection, CollectionPaper


class CollectionRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def list_for(self, owner_id: str, limit: int = 200) -> list[Collection]:
        stmt = (
            select(Collection)
            .where(Collection.owner_id == owner_id)
            .order_by(Collection.pinned.desc(), Collection.position.asc(), Collection.created_at.desc())
            .limit(limit)
        )
        return list(self.db.scalars(stmt).all())

    def get(self, collection_id: str, owner_id: str | None = None) -> Collection | None:
        stmt = select(Collection).where(Collection.id == collection_id)
        if owner_id:
            stmt = stmt.where(Collection.owner_id == owner_id)
        return self.db.scalar(stmt)

    def paper_ids(self, collection_id: str) -> list[str]:
        rows = self.db.scalars(
            select(CollectionPaper.paper_id)
            .where(CollectionPaper.collection_id == collection_id)
            .order_by(CollectionPaper.position.asc())
        ).all()
        return list(rows)

    def has_paper(self, collection_id: str, paper_id: str) -> bool:
        return (
            self.db.scalar(
                select(CollectionPaper.id).where(
                    CollectionPaper.collection_id == collection_id,
                    CollectionPaper.paper_id == paper_id,
                )
            )
            is not None
        )

    def add_paper(self, collection_id: str, paper_id: str, position: int = 0) -> CollectionPaper:
        row = CollectionPaper(collection_id=collection_id, paper_id=paper_id, position=position)
        self.db.add(row)
        return row

    def remove_paper(self, collection_id: str, paper_id: str) -> bool:
        row = self.db.scalar(
            select(CollectionPaper).where(
                CollectionPaper.collection_id == collection_id,
                CollectionPaper.paper_id == paper_id,
            )
        )
        if row is None:
            return False
        self.db.delete(row)
        return True

    def delete(self, collection: Collection) -> None:
        self.db.delete(collection)

    def add(self, collection: Collection) -> Collection:
        self.db.add(collection)
        return collection
