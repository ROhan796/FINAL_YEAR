"""arXiv ingestion adapter — real HTTP client against the arXiv API.

Uses `http://export.arxiv.org/api/query` (Atom XML), which is free, structured
and rate-limit friendly. We honour a 3 s delay between pages (config via
`settings.ingest_rate_limit_seconds`) and are idempotent: papers are keyed by
(`source='arxiv'`, `external_id=arXiv id`).

Nothing here touches the database — the pipeline layer owns storage.
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

log = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"

DEFAULT_CATEGORY = "cs"  # broad CS category; callers can narrow (cs.LG, cs.CL, q-bio.QM…)


@dataclass
class IngestedPaper:
    """Source-agnostic paper record produced by any ingest adapter."""

    external_id: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    year: int = 2026
    venue: str = "arXiv"
    doi: str = ""
    pdf_url: str = ""
    tags: list[str] = field(default_factory=list)
    domain: str = "Computer Science"
    published: datetime | None = None
    citation_count: int = 0


def _clean(text: str) -> str:
    """Collapse whitespace; arXiv abstracts arrive with newlines/indentation."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def _parse_entry(entry: ET.Element) -> IngestedPaper | None:
    title = _clean(entry.findtext(f"{_ATOM}title", ""))
    abstract = _clean(entry.findtext(f"{_ATOM}summary", ""))
    if not title or not abstract:
        return None

    # id looks like http://arxiv.org/abs/1706.03762v7 → keep 1706.03762
    raw_id = entry.findtext(f"{_ATOM}id", "") or ""
    arxiv_id = raw_id.rstrip("/").split("/abs/")[-1]
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

    authors = [
        _clean(a.findtext(f"{_ATOM}name", ""))
        for a in entry.findall(f"{_ATOM}author")
    ]
    authors = [a for a in authors if a]

    published_raw = entry.findtext(f"{_ATOM}published", "") or ""
    published: datetime | None = None
    year = datetime.now(UTC).year
    try:
        published = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
        year = published.year
    except ValueError:
        pass

    doi = ""
    doi_el = entry.find(f"{_ARXIV_NS}doi")
    if doi_el is not None and doi_el.text:
        doi = doi_el.text.strip()

    pdf_url = ""
    for link in entry.findall(f"{_ATOM}link"):
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            pdf_url = link.get("href", "")
            break
    if not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    # Primary category → domain-ish tag; all categories → tags
    categories = [
        c.get("term", "")
        for c in entry.findall(f"{_ATOM}category")
        if c.get("term")
    ]
    primary_el = entry.find(f"{_ARXIV_NS}primary_category")
    primary = primary_el.get("term", "") if primary_el is not None else (categories[0] if categories else "")
    domain = _domain_for(primary or (categories[0] if categories else ""))

    # arXiv does not expose citation counts — S2 enrichment can fill these later.
    return IngestedPaper(
        external_id=arxiv_id,
        title=title,
        abstract=abstract,
        authors=authors,
        year=year,
        venue="arXiv",
        doi=doi,
        pdf_url=pdf_url,
        tags=categories[:8],
        domain=domain,
        published=published,
        citation_count=0,
    )


def _domain_for(category: str) -> str:
    c = category.lower()
    if c.startswith("q-bio") or c.startswith("q-bm"):
        return "Quantitative Biology"
    if c.startswith("q-fin"):
        return "Quantitative Finance"
    if c.startswith("physics"):
        return "Physics"
    if c.startswith("stat"):
        return "Statistics"
    if c.startswith("eess"):
        return "Electrical Engineering"
    if c.startswith("cs"):
        return "Computer Science"
    if c.startswith("math"):
        return "Mathematics"
    return "Computer Science"


def fetch_arxiv(
    query: str = "",
    *,
    max_results: int = 100,
    category: str | None = None,
    rate_limit_seconds: float = 3.0,
    timeout: float = 30.0,
    start: int = 0,
) -> list[IngestedPaper]:
    """Fetch up to `max_results` papers from arXiv, paging 100 at a time.

    `query` is a free-text search; `category` narrows to e.g. ``cs.LG``.
    Both may be empty to pull the newest submissions in the category.
    """
    collected: list[IngestedPaper] = []
    page_size = 100
    fetched = 0
    first = True

    while fetched < max_results:
        n = min(page_size, max_results - fetched)
        search = query.strip()
        if category:
            cat_clause = f"cat:{category}"
            search = f"({search}) AND {cat_clause}" if search else cat_clause
        if not search:
            search = f"cat:{category or DEFAULT_CATEGORY}"

        params = {
            "search_query": search,
            "start": str(start + fetched),
            "max_results": str(n),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        try:
            resp = httpx.get(ARXIV_API, params=params, timeout=timeout, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("arxiv_fetch_failed: %s", exc)
            break

        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as exc:
            log.warning("arxiv_parse_failed: %s", exc)
            break

        entries = root.findall(f"{_ATOM}entry")
        if not entries:
            break

        for entry in entries:
            paper = _parse_entry(entry)
            if paper:
                collected.append(paper)

        fetched += len(entries)
        first = False
        if len(entries) < n:
            break  # exhausted
        if rate_limit_seconds > 0:
            time.sleep(rate_limit_seconds)

    if not first and rate_limit_seconds > 0:
        pass  # already slept between pages
    log.info("arxiv_fetched: query=%r category=%r → %d", query, category, len(collected))
    return collected


def fetch_arxiv_by_ids(ids: list[str], *, rate_limit_seconds: float = 3.0) -> list[IngestedPaper]:
    """Fetch specific arXiv ids (e.g. ['1706.03762', '1810.04805'])."""
    if not ids:
        return []
    id_list = ",".join(ids)
    try:
        resp = httpx.get(
            ARXIV_API,
            params={"id_list": id_list, "max_results": str(len(ids))},
            timeout=30.0,
            follow_redirects=True,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except (httpx.HTTPError, ET.ParseError) as exc:
        log.warning("arxiv_id_fetch_failed: %s", exc)
        return []
    out = []
    for entry in root.findall(f"{_ATOM}entry"):
        p = _parse_entry(entry)
        if p:
            out.append(p)
    return out


__all__ = ["IngestedPaper", "fetch_arxiv", "fetch_arxiv_by_ids", "ARXIV_API"]
