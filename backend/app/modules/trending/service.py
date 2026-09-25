"""Trending topics.

Existing rows are returned directly; on a cold start the ranking is recomputed
from the corpus and persisted so the next request is fast. Grouping is done in
Python so it behaves identically on PostgreSQL and SQLite.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.pagination import Page, PageParams
from app.models.notification import TrendingTopic
from app.models.paper import Paper
from app.schemas import TrendingTopicOut

NARRATIVE = (
    '"{name}" is accelerating: {count} indexed papers cite {citations:,} references in total, '
    "with {growth:.0f}% month-over-month momentum. Cluster activity centres on {clusters}."
)


class TrendingService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def list(self, page: PageParams) -> Page[TrendingTopicOut]:
        rows = list(
            self.db.scalars(
                select(TrendingTopic).order_by(TrendingTopic.monthly_growth.desc())
            ).all()
        )
        if not rows:
            rows = self._compute_and_persist()
        start = 0
        if page.cursor and page.cursor.i:
            ids = [r.id for r in rows]
            if page.cursor.i in ids:
                start = ids.index(page.cursor.i) + 1
        window = rows[start : start + page.limit + 1]
        has_more = len(window) > page.limit
        visible = window[: page.limit]

        from app.core.pagination import Cursor

        return Page[TrendingTopicOut](
            items=[self._out(t) for t in visible],
            nextCursor=Cursor(k=0, i=visible[-1].id).encode() if (has_more and visible) else None,
            total=len(rows),
        )

    # ── cold-start computation ──────────────────────────────────────────

    def _compute_and_persist(self) -> list[TrendingTopic]:
        papers = list(self.db.scalars(select(Paper)).all())
        if not papers:
            return []

        buckets: dict[str, list[Paper]] = {}
        for p in papers:
            for tag in p.tags or []:
                buckets.setdefault(str(tag), []).append(p)

        now_year = datetime.now(UTC).year
        scored: list[tuple[float, str, list[Paper]]] = []
        for tag, members in buckets.items():
            citations = sum(m.citation_count for m in members)
            # Recency-weight: papers from the last 2 years count double.
            recent = sum(1 for m in members if m.year >= now_year - 2)
            score = citations * math.log1p(len(members)) * (1.0 + recent / max(1, len(members)))
            scored.append((score, tag, members))

        if not scored:
            return []
        scored.sort(key=lambda t: t[0], reverse=True)
        max_score = max(s for s, _, _ in scored) or 1.0

        created: list[TrendingTopic] = []
        for score, tag, members in scored[:20]:
            members_sorted = sorted(members, key=lambda m: m.citation_count, reverse=True)
            clusters = sorted(
                {str(t) for m in members_sorted[:5] for t in (m.tags or []) if str(t) != tag}
            )[:5]
            top_papers = [m.id for m in members_sorted[:5]]
            growth = 100.0 * score / max_score
            topic = TrendingTopic(
                name=tag,
                category=tag,
                monthly_growth=growth,
                citation_count=sum(m.citation_count for m in members),
                paper_count=len(members),
                related_paper_ids=top_papers,
                top_researchers=[a for m in members_sorted[:3] for a in (m.authors or [])][:5],
                ai_narrative=NARRATIVE.format(
                    name=tag, count=len(members), citations=sum(m.citation_count for m in members),
                    growth=growth, clusters=", ".join(clusters[:3]) if clusters else "emerging sub-fields",
                ),
                clusters=clusters,
                computed_at=datetime.now(UTC),
            )
            self.db.merge(topic)
            created.append(topic)
        self.db.commit()
        return created

    @staticmethod
    def _out(t: TrendingTopic) -> TrendingTopicOut:
        return TrendingTopicOut(
            id=t.id,
            name=t.name,
            category=t.category,
            monthlyGrowth=t.monthly_growth,
            citationCount=t.citation_count,
            paperCount=t.paper_count,
            relatedPaperIds=list(t.related_paper_ids or []),
            topResearchers=list(t.top_researchers or []),
            aiNarrative=t.ai_narrative,
            clusters=list(t.clusters or []),
        )


__all__ = ["TrendingService", "Any"]
