"""PubMed ingestion adapter — NCBI E-utilities (esearch + efetch XML).

Free, structured, rate-limit friendly (<=3 req/s without API key).
Produces `IngestedPaper` records for `IngestPipeline`.
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

import httpx

from ml.ingest.arxiv import IngestedPaper

log = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _parse_pubmed_article(article: ET.Element) -> IngestedPaper | None:
    medline = article.find("MedlineCitation")
    if medline is None:
        return None
    pmid = _clean(medline.findtext("PMID", ""))
    art = medline.find("Article")
    if art is None or not pmid:
        return None

    title = _clean("".join(art.find("ArticleTitle").itertext())) if art.find("ArticleTitle") is not None else ""
    abstract_parts = [
        _clean("".join(el.itertext()))
        for el in art.findall(".//Abstract/AbstractText")
    ]
    abstract = " ".join(p for p in abstract_parts if p)
    if not title or not abstract:
        return None

    authors: list[str] = []
    for au in art.findall(".//AuthorList/Author"):
        last = _clean(au.findtext("LastName", ""))
        fore = _clean(au.findtext("ForeName", ""))
        coll = _clean(au.findtext("CollectiveName", ""))
        name = f"{fore} {last}".strip() if (fore or last) else coll
        if name:
            authors.append(name)

    year = datetime.now(UTC).year
    year_txt = _clean(art.findtext(".//Journal/JournalIssue/PubDate/Year", ""))
    if not year_txt:
        medline_date = _clean(art.findtext(".//Journal/JournalIssue/PubDate/MedlineDate", ""))
        m = re.search(r"(19|20)\d{2}", medline_date)
        if m:
            year_txt = m.group(0)
    if year_txt.isdigit():
        year = int(year_txt)

    venue = _clean(art.findtext(".//Journal/ISOAbbreviation", "")) or _clean(
        art.findtext(".//Journal/Title", "")
    ) or "PubMed"

    doi = ""
    for aid in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
        if aid.get("IdType") == "doi" and aid.text:
            doi = aid.text.strip()
            break

    tags = [
        _clean(mh.findtext("DescriptorName", ""))
        for mh in medline.findall(".//MeshHeadingList/MeshHeading")
    ]
    tags = [t for t in tags if t][:8]
    if not tags:
        tags = ["Biomedical", "PubMed"]

    return IngestedPaper(
        external_id=pmid,
        title=title,
        abstract=abstract,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        pdf_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        tags=tags,
        domain="Medicine",
        published=datetime(year, 1, 1, tzinfo=UTC),
        citation_count=0,
    )


def fetch_pubmed(
    query: str,
    *,
    max_results: int = 50,
    rate_limit_seconds: float = 0.4,
    timeout: float = 30.0,
) -> list[IngestedPaper]:
    """Search PubMed via E-utilities and fetch structured article XML."""
    q = (query or "").strip()
    if not q:
        q = "graph neural networks biomedical"

    try:
        search_resp = httpx.get(
            ESEARCH_URL,
            params={
                "db": "pubmed",
                "term": q,
                "retmax": str(min(max_results, 200)),
                "retmode": "json",
                "sort": "relevance",
            },
            timeout=timeout,
            follow_redirects=True,
        )
        search_resp.raise_for_status()
        id_list = search_resp.json().get("esearchresult", {}).get("idlist", [])
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("pubmed_esearch_failed: %s", exc)
        return []

    if not id_list:
        return []

    if rate_limit_seconds > 0:
        time.sleep(min(rate_limit_seconds, 1.0))

    collected: list[IngestedPaper] = []
    batch_size = 50
    for i in range(0, len(id_list), batch_size):
        batch = id_list[i : i + batch_size]
        try:
            fetch_resp = httpx.get(
                EFETCH_URL,
                params={"db": "pubmed", "id": ",".join(batch), "retmode": "xml"},
                timeout=timeout,
                follow_redirects=True,
            )
            fetch_resp.raise_for_status()
            root = ET.fromstring(fetch_resp.text)
        except (httpx.HTTPError, ET.ParseError) as exc:
            log.warning("pubmed_efetch_failed: %s", exc)
            continue

        for art in root.findall(".//PubmedArticle"):
            paper = _parse_pubmed_article(art)
            if paper is not None:
                collected.append(paper)

        if rate_limit_seconds > 0 and i + batch_size < len(id_list):
            time.sleep(min(rate_limit_seconds, 1.0))

    log.info("pubmed_fetched: query=%r -> %d", q, len(collected))
    return collected[:max_results]


__all__ = ["fetch_pubmed", "ESEARCH_URL", "EFETCH_URL"]
