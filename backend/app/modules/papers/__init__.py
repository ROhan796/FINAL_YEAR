"""Paper catalog: listing, detail, related, save, citation export, AI analysis."""

from __future__ import annotations

from app.models.paper import Paper, PaperTag, RelatedPaper, SavedPaper

__all__ = ["Paper", "PaperTag", "RelatedPaper", "SavedPaper"]
