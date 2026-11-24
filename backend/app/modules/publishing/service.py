"""Manuscript submission pipeline: create, upload PDF, AI pre-review, status."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import ConflictError, ForbiddenError, NotFoundError, ValidationAppError
from app.models.manuscript import Manuscript
from app.models.user import User
from app.schemas import AIReviewOut, ManuscriptCreate, ManuscriptOut

#: Saved under the working directory (or any configured path) — never a web root.
UPLOAD_DIR = Path(settings.upload_dir)
ALLOWED_PDF_MAGIC = b"%PDF"
#: Frontend `ManuscriptSubmission.submittedAt` is a display string.
_DATE_FMT = "%b %d, %Y"

STATUS_ORDER = ["Draft", "Under AI Review", "Submitted", "Under Review", "Published", "Revision Required"]


class PublishingService:
    def __init__(self, db: Session) -> None:
        self.db = db

    # ── queries ─────────────────────────────────────────────────────────

    def list(self, submitter_id: str) -> list[ManuscriptOut]:
        rows = self.db.scalars(
            select(Manuscript)
            .where(Manuscript.submitter_id == submitter_id)
            .order_by(Manuscript.created_at.desc())
        ).all()
        return [self.out(m) for m in rows]

    def get(self, manuscript_id: str, viewer_id: str) -> Manuscript:
        row = self.db.get(Manuscript, manuscript_id)
        if row is None:
            raise NotFoundError(f"Manuscript '{manuscript_id}' not found")
        # Reviewers/admins see everything; authors only see their own.
        viewer = self.db.get(User, viewer_id)
        is_staff = viewer is not None and viewer.role in ("editor", "reviewer", "admin")
        if row.submitter_id != viewer_id and not is_staff:
            raise NotFoundError(f"Manuscript '{manuscript_id}' not found")
        return row

    # ── mutations ───────────────────────────────────────────────────────

    def create(self, user: dict[str, Any], payload: ManuscriptCreate) -> ManuscriptOut:
        # Contract: full submission requires a verified researcher.
        if not user.get("verified"):
            raise ForbiddenError("Verify your researcher profile before submitting manuscripts")
        dupe = self.db.scalar(
            select(Manuscript).where(
                Manuscript.submitter_id == user["id"], Manuscript.title == payload.title
            )
        )
        if dupe is not None:
            raise ConflictError("You already have a manuscript with that title")
        row = Manuscript(
            submitter_id=user["id"],
            title=payload.title,
            abstract=payload.abstract,
            authors=payload.authors or [user.get("name", "")],
            co_authors=payload.coAuthors,
            venue=payload.venue,
            tags=payload.tags,
            year=payload.year,
            integrity_confirmed=payload.integrityConfirmed,
            status="Draft",
        )
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return self.out(row)

    def attach_file(self, manuscript_id: str, user: dict[str, Any], name: str, data: bytes) -> ManuscriptOut:
        row = self.get(manuscript_id, user["id"])
        if row.submitter_id != user["id"]:
            raise ForbiddenError("Only the submitter can attach a file")
        if len(data) == 0:
            raise ValidationAppError("Uploaded file is empty")
        if len(data) > settings.max_upload_bytes:
            raise ValidationAppError(f"File exceeds {settings.max_upload_bytes // (1024 * 1024)} MB limit")
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name) or f"{uuid.uuid4().hex}.pdf"
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"
        # Validate the actual bytes, not just the extension.
        if not data.lstrip()[:5].startswith(ALLOWED_PDF_MAGIC):
            raise ValidationAppError("Only PDF files are supported")

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        key = f"{uuid.uuid4().hex}-{safe_name}"
        (UPLOAD_DIR / key).write_bytes(data)

        row.file_name = safe_name
        row.file_size = f"{max(1, round(len(data) / 1024))} KB"
        row.file_key = key
        if row.status == "Draft":
            row.status = "Under AI Review"
        self.db.commit()
        self.db.refresh(row)
        return self.out(row)

    def prereview(self, manuscript_id: str, user: dict[str, Any]) -> dict[str, Any]:
        row = self.get(manuscript_id, user["id"])
        if row.submitter_id != user["id"]:
            raise ForbiddenError("Only the submitter can request an AI pre-review")
        if row.status == "Draft":
            row.status = "Under AI Review"

        from app.ai.service import analyze_draft

        review = analyze_draft(
            title=row.title,
            abstract=row.abstract,
            venue=row.venue,
            tags=list(row.tags or []),
            file_name=row.file_name,
            alias=f"manuscript:{row.id}",
        )
        row.ai_review = review
        row.ai_reviewed_at = datetime.now(UTC)
        self.db.commit()
        self.db.refresh(row)
        return {"aiReview": self._review_out(review), "manuscriptId": row.id}

    def citation_gaps(
        self, manuscript_id: str, user: dict[str, Any], draft_text: str = ""
    ) -> dict[str, Any]:
        """7.4: Extract methods/algorithms in draft lacking citations and recommend candidate papers."""
        from sqlalchemy import or_
        from app.models.paper import Paper
        from ml.ner import NERPipeline

        row = self.get(manuscript_id, user["id"])
        text = "\n".join(p for p in [row.title, row.abstract, draft_text] if p)
        entities = NERPipeline().extract_entities(text)
        target_types = {"METHOD", "ALGORITHM", "DATASET", "MATERIAL"}
        seen: set[str] = set()
        gaps: list[dict[str, Any]] = []

        for ent in entities:
            if ent.label not in target_types:
                continue
            key = ent.text.lower()
            if key in seen:
                continue
            seen.add(key)

            # Check if a citation marker ([1], (2022), et al.) appears near the mention
            window = text[max(0, ent.start - 20) : min(len(text), ent.end + 80)]
            if re.search(r"(\[\d+\]|et al\.|\(\d{4}\))", window):
                continue

            like = f"%{ent.text}%"
            cands = list(
                self.db.scalars(
                    select(Paper)
                    .where(or_(Paper.title.ilike(like), Paper.abstract.ilike(like)))
                    .order_by(Paper.citation_count.desc())
                    .limit(4)
                ).all()
            )
            gaps.append(
                {
                    "entity": ent.text,
                    "entityType": ent.label,
                    "message": f"You mention '{ent.text}' ({ent.label}) without a nearby citation.",
                    "candidates": [
                        {
                            "id": p.id,
                            "title": p.title,
                            "year": p.year,
                            "venue": p.venue,
                            "citationCount": p.citation_count,
                        }
                        for p in cands
                    ],
                }
            )

        return {"manuscriptId": row.id, "gapCount": len(gaps), "gaps": gaps[:8]}

    def set_status(self, manuscript_id: str, user: dict[str, Any], status: str) -> ManuscriptOut:
        if status not in STATUS_ORDER:
            raise ValidationAppError(f"Unknown status '{status}'")
        row = self.get(manuscript_id, user["id"])
        is_reviewer = user.get("role") in ("editor", "reviewer", "admin")
        if row.submitter_id != user["id"] and not is_reviewer:
            raise ForbiddenError("Only the submitter or a reviewer can change status")
        # An author may only advance their own draft to a submitted state.
        if not is_reviewer and STATUS_ORDER.index(status) > STATUS_ORDER.index("Submitted"):
            raise ForbiddenError("Only a reviewer can move a manuscript past 'Submitted'")
        row.status = status
        if status in ("Submitted", "Under Review", "Published"):
            row.submitted_at = row.submitted_at or datetime.now(UTC)
        if status == "Published":
            row.decided_at = datetime.now(UTC)
        self.db.commit()
        self.db.refresh(row)
        self._notify_status(row)
        return self.out(row)

    # ── serialisation ───────────────────────────────────────────────────

    @staticmethod
    def _review_out(review: dict[str, Any] | None) -> AIReviewOut | None:
        if not review:
            return None
        return AIReviewOut(
            predictedScore=int(review.get("predictedScore") or 85),
            titleReview=str(review.get("titleReview") or ""),
            executiveCritique=str(review.get("executiveCritique") or ""),
            suggestedKeywords=[str(x) for x in review.get("suggestedKeywords") or []],
            likelyPeerQuestions=[str(x) for x in review.get("likelyPeerQuestions") or []],
            clarityAssessment=str(review.get("clarityAssessment") or ""),
            strengths=[str(x) for x in review.get("strengths") or []],
            weaknesses=[str(x) for x in review.get("weaknesses") or []],
        )

    @classmethod
    def out(cls, row: Manuscript) -> ManuscriptOut:
        stamp = row.submitted_at or row.created_at
        if stamp is not None and stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        return ManuscriptOut(
            id=row.id,
            title=row.title,
            abstract=row.abstract,
            authors=list(row.authors or []),
            coAuthors=list(row.co_authors or []),
            venue=row.venue,
            tags=list(row.tags or []),
            year=row.year,
            fileName=row.file_name,
            fileSize=row.file_size,
            integrityConfirmed=row.integrity_confirmed,
            status=row.status,  # type: ignore[arg-type]
            submittedAt=stamp.strftime(_DATE_FMT) if stamp else "—",
            aiReview=cls._review_out(row.ai_review),
        )

    def _notify_status(self, row: Manuscript) -> None:
        try:
            from app.modules.notifications.service import create_notification

            create_notification(
                self.db,
                row.submitter_id,
                "manuscript",
                "Manuscript status updated",
                f'"{row.title}" is now {row.status}.',
                link=f"/publish?manuscript={row.id}",
            )
        except Exception:  # noqa: BLE001 - notifications must never block a status change
            pass


__all__ = ["PublishingService", "UPLOAD_DIR", "STATUS_ORDER"]
