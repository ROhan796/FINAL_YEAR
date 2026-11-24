"""Semantic Scholar Graph API ingestion adapter.

Uses https://api.semanticscholar.org/graph/v1/paper/search (free tier or API key).
Returns structured `IngestedPaper` records including citation counts and domains.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

import httpx

from ml.ingest.arxiv import IngestedPaper

log = logging.getLogger(__name__)

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "paperId,title,abstract,year,venue,citationCount,authors,externalIds,openAccessPdf,fieldsOfStudy"


def _map_domain(fields: list[str] | None) -> str:
    if not fields:
        return "Computer Science"
    first = fields[0]
    mapping = {
        "Medicine": "Medicine",
        "Biology": "Quantitative Biology",
        "Materials Science": "Materials Science",
        "Physics": "Physics",
        "Mathematics": "Mathematics",
        "Chemistry": "Chemistry",
        "Engineering": "Electrical Engineering",
        "Computer Science": "Computer Science",
    }
    return mapping.get(first, first)


def fetch_semantic_scholar(
    query: str,
    *,
    max_results: int = 50,
    api_key: str = "",
    rate_limit_seconds: float = 1.5,
    timeout: float = 30.0,
) -> list[IngestedPaper]:
    """Search Semantic Scholar Graph API and convert hits to `IngestedPaper`."""
    q = (query or "").strip() or "knowledge graph link prediction"
    headers = {"x-api-key": api_key} if api_key else {}

    collected: list[IngestedPaper] = []
    offset = 0
    page_size = min(100, max_results)

    while len(collected) < max_results:
        limit = min(page_size, max_results - len(collected))
        try:
            resp = httpx.get(
                S2_SEARCH_URL,
                params={
                    "query": q,
                    "offset": str(offset),
                    "limit": str(limit),
                    "fields": _FIELDS,
                },
                headers=headers,
                timeout=timeout,
                follow_redirects=True,
            )
            resp.raise_for_status()
            payload = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("s2_fetch_failed: %s", exc)
            break

        data = payload.get("data") or []
        if not data:
            break

        for item in data:
            title = (item.get("title") or "").strip()
            abstract = (item.get("abstract") or "").strip()
            pid = (item.get("paperId") or "").strip()
            if not pid or not title or not abstract:
                continue

            authors = [a.get("name", "").strip() for a in (item.get("authors") or []) if a.get("name")]
            ext_ids = item.get("externalIds") or {}
            doi = str(ext_ids.get("DOI") or "")
            year = int(item.get("year") or datetime.now(UTC).year)
            oa_pdf = (item.get("openAccessPdf") or {}).get("url") or ""
            fields = list(item.get("fieldsOfStudy") or [])

            collected.append(
                IngestedPaper(
                    external_id=pid,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    year=year,
                    venue=(item.get("venue") or "Semantic Scholar").strip() or "Semantic Scholar",
                    doi=doi,
                    pdf_url=oa_pdf or f"https://www.semanticscholar.org/paper/{pid}",
                    tags=(fields or ["Research"])[:8],
                    domain=_map_domain(fields),
                    published=datetime(year, 1, 1, tzinfo=UTC),
                    citation_count=int(item.get("citationCount") or 0),
                )
            )

        offset += len(data)
        if len(data) < limit or len(collected) >= max_results:
            break
        if rate_limit_seconds > 0:
            time.sleep(rate_limit_seconds)

    log.info("s2_fetched: query=%r -> %d", q, len(collected))
    return collected[:max_results]


__all__ = ["fetch_semantic_scholar", "S2_SEARCH_URL"]
