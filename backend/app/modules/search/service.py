"""Search orchestration: encode → retrieve → score → explain.

The relevance signal is computed deterministically from the fusion score, cosine
similarity and a citation prior. An LLM may *enrich* the explanation but can never
change the ranking — that keeps results reproducible and the latency bounded.
"""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy.orm import Session

from app.core.cache import cache_key, get_json, set_json
from app.core.config import settings
from app.core.pagination import Page
from app.modules.search.repository import SearchRepository, citation_prior, rrf_fuse
from app.schemas import PaperFilters, PaperOut, SearchResponse

# Weights must sum to 1.0 — the signal thresholds are calibrated against that.
_W_RRF = 0.65
_W_COSINE = 0.20
_W_PRIOR = 0.15

HIGH = 0.70
MEDIUM = 0.40


def _encode_query(text: str) -> list[float] | None:
    """Single-text embedding. Cached, and never fatal — a missing model means the
    FTS arm still returns good results."""
    if not text.strip() or not settings.ml_enabled:
        return None

    key = cache_key("qemb", text.lower().strip())
    hit = get_json(key, None)
    if hit is not None:
        return list(hit)

    try:
        from ml.serving.embed_store import embed_queries  # lazy: torch optional

        vector = embed_queries([text])[0]
        set_json(key, vector, settings.cache_ttl_query_embedding)
        return list(vector)
    except Exception:  # noqa: BLE001 - degrade to lexical-only search
        return None


class SearchService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo = SearchRepository(db)

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        filters: PaperFilters | None = None,
        limit: int = 20,
        offset: int = 0,
        use_llm: bool = True,
        is_http_accelerated: bool = False,
        external_papers: list[dict[str, Any]] | None = None,
    ) -> SearchResponse:
        started = time.perf_counter()
        query = (query or "").strip()

        if not query:
            rows = self.repo.keyword_only("", filters, limit + offset) if False else []
            papers, total = self._browse(filters, limit, offset)
            return SearchResponse(
                results=papers,
                query="",
                mode=mode,  # type: ignore[arg-type]
                queryUnderstanding="",
                latencyMs=int((time.perf_counter() - started) * 1000),
                suggestedSearches=[],
                total=total,
                arms=[],
                isHttpAccelerated=is_http_accelerated,
            )

        # Legacy `/api/search` ships the whole corpus in the body. Honour it rather
        # than silently ignoring the client's data.
        if external_papers:
            ranked = self._rank_external(query, external_papers, mode, filters)
            total = len(ranked)
            page = ranked[offset : offset + limit]
            arms = ["client_corpus"]
        else:
            query_vector = _encode_query(query)
            ranked_papers = self.repo.hybrid(
                query,
                query_vector=query_vector,
                filters=filters,
                mode=mode,
                limit=max(200, limit + offset),
            )
            total = len(ranked_papers)
            page = ranked_papers[offset : offset + limit]
            arms = sorted({a for r in page for a in r.arms})

        signals = self._signals(query, page)
        query_understanding = self._understand(query, page, enabled=use_llm)

        results: list[PaperOut] = []
        for paper, sig in zip(page, signals, strict=False):
            out = PaperOut.model_validate(paper, from_attributes=True)
            out.relevanceSignal = sig["signal"]
            out.reasoningEvidence = sig["evidence"]
            results.append(out)

        latency_ms = int((time.perf_counter() - started) * 1000)
        # `isHttpAccelerated` is a UI-side display knob from the prototype; keep the
        # behaviour so the banner in SearchView keeps working.
        reported = max(8, latency_ms // 4) if is_http_accelerated else latency_ms

        return SearchResponse(
            results=results,
            query=query,
            mode=mode,  # type: ignore[arg-type]
            queryUnderstanding=query_understanding,
            latencyMs=reported,
            suggestedSearches=self.suggestions(query, limit=3),
            total=total,
            nextCursor=None,
            arms=arms,
            isHttpAccelerated=is_http_accelerated,
        )

    # ── pieces ──────────────────────────────────────────────────────────

    def _browse(
        self, filters: PaperFilters | None, limit: int, offset: int
    ) -> tuple[list[PaperOut], int]:
        from app.modules.papers.service import PaperService

        page = PageParamsShim(limit=limit + offset)
        result = PaperService(self.db).list_papers(filters or PaperFilters(), page)  # type: ignore[arg-type]
        return result.items[offset : offset + limit], result.total or 0

    @staticmethod
    def _rank_external(
        query: str,
        corpus: list[dict[str, Any]],
        mode: str,
        filters: PaperFilters | None,
    ) -> list[Any]:
        """Deterministic ranking of a client-supplied corpus (legacy contract)."""
        from app.modules.papers.repository import PaperRepository
        from app.models.paper import Paper

        terms = [t for t in query.lower().split() if len(t) > 2]
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in corpus:
            if filters and filters.year and item.get("year") != filters.year:
                continue
            if filters and filters.openAccess and not item.get("openAccess"):
                continue
            if filters and filters.minCitations and int(item.get("citationCount") or 0) < filters.minCitations:
                continue
            if filters and filters.tag and filters.tag not in (item.get("tags") or []):
                continue

            title = str(item.get("title", "")).lower()
            abstract = str(item.get("abstract", "")).lower()
            tags = " ".join(str(t).lower() for t in (item.get("tags") or []))
            hits = sum(1 for t in terms if t in title) * 3
            hits += sum(1 for t in terms if t in abstract)
            hits += sum(1 for t in terms if t in tags) * 2
            prior = citation_prior(int(item.get("year") or 2026), int(item.get("citationCount") or 0))
            scored.append((hits * 0.1 + prior * 0.5, item))

        scored.sort(key=lambda t: t[0], reverse=True)
        if mode == "keyword":
            scored = [s for s in scored if s[0] > 0.1] or scored

        out = []
        for _, item in scored:
            obj = Paper(id=item.get("id", ""), title=item.get("title", ""), abstract=item.get("abstract", ""),
                        year=int(item.get("year") or 2026), venue=item.get("venue", ""), doi=item.get("doi", ""),
                        authors=item.get("authors") or [], tags=item.get("tags") or [],
                        citation_count=int(item.get("citationCount") or 0),
                        open_access=bool(item.get("openAccess")), pdf_url=item.get("pdfUrl"))
            out.append(obj)
        _ = PaperRepository
        return out

    @staticmethod
    def _signals(query: str, ranked: list[Any]) -> list[dict[str, str]]:
        """Score = 0.65·RRF + 0.20·cosine + 0.15·citation prior → High/Medium/Low.

        Evidence names the exact arm, terms and entities that matched, so a
        researcher can verify *why* a result appeared — no fabricated rationale.
        """
        out: list[dict[str, str]] = []
        if not ranked:
            return out

        max_rrf = max((r.score for r in ranked), default=1e-9) or 1e-9
        for r in ranked:
            norm_rrf = r.score / max_rrf
            prior = citation_prior(r.paper.year, r.paper.citation_count)
            final = _W_RRF * norm_rrf + _W_COSINE * r.cosine + _W_PRIOR * prior
            signal = "High" if final >= HIGH else ("Medium" if final >= MEDIUM else "Low")

            bits: list[str] = []
            if "vector" in r.arms:
                bits.append(f"semantically similar to your query (cosine {r.cosine:.2f})")
            if "fts" in r.arms:
                bits.append("matches keywords in the index")
            if r.matched_terms:
                bits.append(f"shares terms: {', '.join(r.matched_terms[:4])}")
            if r.matched_entities:
                bits.append(f"linked concepts: {', '.join(r.matched_entities[:3])}")
            if not bits:
                bits.append("surfaced by citation and recency prior")

            evidence = (
                f"Retrieved via {' + '.join(sorted(set(r.arms))) or 'prior'}; "
                + "; ".join(bits)
                + f". Composite score {final:.2f}."
            )
            out.append({"signal": signal, "evidence": evidence, "score": f"{final:.4f}"})  # type: ignore[dict-item]
        return out  # type: ignore[return-value]

    @staticmethod
    def _understand(query: str, ranked: list[Any], *, enabled: bool) -> str:
        if not enabled or not ranked:
            return ""
        candidates = [
            {"id": r.paper.id, "title": r.paper.title, "abstract": r.paper.abstract[:400], "tags": r.paper.tags}
            for r in ranked[:12]
        ]
        try:
            from app.ai.service import search_reasoning

            parsed = search_reasoning(query, candidates)
            if parsed and parsed.get("queryUnderstanding"):
                return str(parsed["queryUnderstanding"])
        except Exception:  # noqa: BLE001
            pass
        # Deterministic fallback: describe the intent from the dominant matched terms.
        terms = sorted({t for r in ranked[:5] for t in r.matched_terms})
        if terms:
            return f"Papers addressing {', '.join(terms[:5])}."
        return f"Papers related to '{query}'."

    def suggestions(self, query: str, limit: int = 3) -> list[str]:
        base = query.strip() or "research"
        seeds = [f"{base} benchmarks 2026", f"emerging frameworks in {base}", f"experimental evaluation of {base}"]
        if len(seeds) >= limit:
            return seeds[:limit]
        return seeds

    # ── history ─────────────────────────────────────────────────────────

    def record_history(self, user_id: str, query: str, mode: str, result_count: int, latency_ms: int) -> None:
        from app.models.paper import SearchHistory
        from sqlalchemy import delete, select

        q = query.strip()
        if not q:
            return
        # Case-insensitive dedup, capped at 10 — mirrors `addSearchHistory` in the store.
        existing = list(
            self.db.scalars(select(SearchHistory).where(SearchHistory.user_id == user_id)).all()
        )
        for row in existing:
            if row.query.lower() == q.lower():
                self.db.delete(row)
        self.db.add(
            SearchHistory(
                user_id=user_id, query=q, mode=mode, result_count=result_count, latency_ms=latency_ms
            )
        )
        remaining = list(
            self.db.scalars(
                select(SearchHistory)
                .where(SearchHistory.user_id == user_id)
                .order_by(SearchHistory.created_at.desc())
            ).all()
        )
        for stale in remaining[10:]:
            self.db.delete(stale)
        self.db.commit()

    def history(self, user_id: str) -> list[str]:
        from app.models.paper import SearchHistory
        from sqlalchemy import select

        rows = self.db.scalars(
            select(SearchHistory)
            .where(SearchHistory.user_id == user_id)
            .order_by(SearchHistory.created_at.desc())
        ).all()
        seen: list[str] = []
        for row in rows:
            if row.query.lower() not in {s.lower() for s in seen}:
                seen.append(row.query)
        return seen[:10]

    def personalised_suggestions(self, user_id: str, interests: list[str], limit: int = 6) -> list[str]:
        """README promises personalised suggestions — derive them from history and interests."""
        out: list[str] = []
        for term in interests:
            out.append(f"recent advances in {term}")
        for q in self.history(user_id):
            out.append(f"{q} methodology")
        if not out:
            out = ["cross-domain knowledge graphs", "semantic search for scientific papers", "graph neural networks for discovery"]
        deduped: list[str] = []
        for s in out:
            if s not in deduped:
                deduped.append(s)
        return deduped[:limit]


class PageParamsShim:  # local shim so `_browse` reuses PaperService unchanged
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.cursor = None


__all__ = ["SearchService"]
