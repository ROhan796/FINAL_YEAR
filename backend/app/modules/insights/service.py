"""Novel Research Intelligence Algorithms & Insights Service.

Implements:
- 6.B Research Gap Detection (cross-domain KG entity co-occurrence analysis)
- 6.D Reading-Order Graph & Gateway Papers (PageRank over save sequences)
- 6.E Contrastive Collection Clustering (K-Means + TF-IDF auto-naming)
- 7.2 Research Debt Tracker (unannotated/unread aging saved papers)
- 7.3 Cross-Collection Insights (centroid-bridging papers across collections)
- 7.5 Collaboration Affinity Score (entity + topic + mutual-save overlap)
- 7.7 Research Velocity Dashboard (self-reflective productivity metrics)
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.collection import Collection, CollectionPaper
from app.models.knowledge import KgEdge, KgNode
from app.models.manuscript import Manuscript
from app.models.paper import Paper, PaperChunk, RelatedPaper, SavedPaper
from app.models.project import Project, ProjectNote, ProjectPaper, ProjectTask
from app.models.social import ResearcherProfile
from app.models.user import UserProfile


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class InsightsService:
    def __init__(self, db: Session) -> None:
        self.db = db

    # ── 6.B Research Gap Detection ──────────────────────────────────────

    def research_gaps(self, limit: int = 15) -> list[dict[str, Any]]:
        """Find (Method/Algorithm) x (Disease/Material/Dataset/Phenomenon) pairs
        active in one domain but unexplored in another."""
        nodes = {n.id: n for n in self.db.scalars(select(KgNode)).all()}
        edges = list(self.db.scalars(select(KgEdge)).all())
        if not nodes or not edges:
            return []

        existing_pairs: set[tuple[str, str]] = {(e.src_id, e.dst_id) for e in edges}
        src_types = {"METHOD", "ALGORITHM"}
        dst_types = {"DISEASE", "MATERIAL", "PHENOMENON", "DATASET"}

        methods = [n for n in nodes.values() if n.entity_type in src_types]
        targets = [n for n in nodes.values() if n.entity_type in dst_types]

        # Count out-degree per node to prioritize well-grounded concepts
        degree: dict[str, int] = defaultdict(int)
        for e in edges:
            degree[e.src_id] += 1
            degree[e.dst_id] += 1

        gaps: list[dict[str, Any]] = []
        for m in sorted(methods, key=lambda n: degree[n.id], reverse=True)[:25]:
            for t in sorted(targets, key=lambda n: degree[n.id], reverse=True)[:25]:
                if m.id == t.id or (m.id, t.id) in existing_pairs or (t.id, m.id) in existing_pairs:
                    continue
                if m.domain and t.domain and m.domain == t.domain:
                    continue

                emb_sim = 0.5
                if m.embedding is not None and t.embedding is not None:
                    try:
                        emb_sim = max(
                            0.1,
                            _cos(
                                np.asarray(m.embedding, dtype=np.float32),
                                np.asarray(t.embedding, dtype=np.float32),
                            ),
                        )
                    except Exception:
                        pass

                grounding = min(1.0, (degree[m.id] + degree[t.id] + 2) / 10.0)
                gap_score = round(min(0.99, 0.55 * grounding + 0.45 * emb_sim), 4)
                gaps.append(
                    {
                        "id": f"gap-{m.id[:8]}-{t.id[:8]}",
                        "method": m.canonical_name,
                        "methodDomain": m.domain or "Computer Science",
                        "target": t.canonical_name,
                        "targetType": t.entity_type,
                        "targetDomain": t.domain or "Medicine",
                        "gapScore": gap_score,
                        "summary": (
                            f"{m.canonical_name} is established in {m.domain or 'its source field'} "
                            f"but has no recorded link to {t.canonical_name} ({t.entity_type}) in "
                            f"{t.domain or 'the target field'}."
                        ),
                        "supportingPaperIds": [pid for pid in (m.paper_id, t.paper_id) if pid],
                    }
                )

        gaps.sort(key=lambda g: g["gapScore"], reverse=True)
        return gaps[:limit]

    # ── 6.D Reading-Order Graph & Gateway Papers (PageRank) ─────────────

    def gateway_papers(self, limit: int = 8) -> list[dict[str, Any]]:
        """Build directed reading-order graph from saved_papers timestamps (plus
        related-paper/collection transitions) and compute PageRank gateway scores."""
        papers = list(self.db.scalars(select(Paper)).all())
        if not papers:
            return []
        by_id = {p.id: p for p in papers}
        idx_of = {p.id: i for i, p in enumerate(papers)}
        N = len(papers)

        adj: list[list[int]] = [[] for _ in range(N)]
        transition_counts: dict[tuple[str, str], int] = defaultdict(int)

        # 1. Explicit reading-order signal from user save timestamps
        saves = list(
            self.db.scalars(
                select(SavedPaper).order_by(SavedPaper.user_id, SavedPaper.created_at)
            ).all()
        )
        by_user: dict[str, list[SavedPaper]] = defaultdict(list)
        for s in saves:
            by_user[s.user_id].append(s)
        for user_saves in by_user.values():
            for a, b in zip(user_saves, user_saves[1:]):
                if a.paper_id in idx_of and b.paper_id in idx_of and a.paper_id != b.paper_id:
                    adj[idx_of[a.paper_id]].append(idx_of[b.paper_id])
                    transition_counts[(a.paper_id, b.paper_id)] += 1

        # 2. Structural transitions from related_papers (older/foundational -> newer/applied)
        related = list(self.db.scalars(select(RelatedPaper)).all())
        for rel in related:
            p1 = by_id.get(rel.paper_id)
            p2 = by_id.get(rel.related_to_id)
            if p1 is None or p2 is None or p1.id == p2.id:
                continue
            # Foundational direction: higher-citation or earlier year -> newer follow-up
            if (p1.year, -p1.citation_count) <= (p2.year, -p2.citation_count):
                adj[idx_of[p1.id]].append(idx_of[p2.id])
            else:
                adj[idx_of[p2.id]].append(idx_of[p1.id])

        # PageRank (d = 0.85, 40 iterations)
        d = 0.85
        pr = np.full(N, 1.0 / N, dtype=np.float64)
        for _ in range(40):
            new_pr = np.full(N, (1.0 - d) / N, dtype=np.float64)
            dangling = 0.0
            for u in range(N):
                if not adj[u]:
                    dangling += pr[u]
                else:
                    share = d * pr[u] / len(adj[u])
                    for v in adj[u]:
                        new_pr[v] += share
            new_pr += d * dangling / N
            pr = new_pr

        # Normalize to 0..1 for display
        max_pr = float(pr.max()) or 1.0
        order = np.argsort(pr)[::-1][:limit]
        out: list[dict[str, Any]] = []
        for rank_idx, i in enumerate(order, 1):
            p = papers[int(i)]
            in_deg = sum(1 for u in range(N) if int(i) in adj[u])
            out.append(
                {
                    "rank": rank_idx,
                    "paperId": p.id,
                    "title": p.title,
                    "year": p.year,
                    "venue": p.venue,
                    "domain": p.domain,
                    "citationCount": p.citation_count,
                    "gatewayScore": round(float(pr[int(i)] / max_pr), 4),
                    "inDegree": in_deg,
                    "rationale": (
                        f"PageRank gateway (#{rank_idx}) bridging {in_deg} reading-order transitions "
                        f"in {p.domain or 'research literature'}."
                    ),
                }
            )
        return out

    # ── 6.E Contrastive Collection Clustering (K-Means + TF-IDF) ────────

    def collection_clusters(self, collection_id: str, owner_id: str) -> dict[str, Any]:
        col = self.db.get(Collection, collection_id)
        if col is None or col.owner_id != owner_id:
            return {"collectionId": collection_id, "clusters": []}

        pids = list(
            self.db.scalars(
                select(CollectionPaper.paper_id).where(CollectionPaper.collection_id == collection_id)
            ).all()
        )
        papers = list(self.db.scalars(select(Paper).where(Paper.id.in_(pids))).all()) if pids else []
        if not papers:
            return {"collectionId": collection_id, "clusters": []}

        if len(papers) == 1:
            p = papers[0]
            return {
                "collectionId": collection_id,
                "clusters": [
                    {
                        "clusterId": 0,
                        "label": ", ".join((p.tags or [p.domain or "Core Topic"])[:3]),
                        "paperIds": [p.id],
                        "paperTitles": [p.title],
                        "topTerms": (p.tags or [])[:4],
                    }
                ],
            }

        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        docs = [f"{p.title}. {p.abstract}" for p in papers]
        vec_list: list[np.ndarray] = []
        for p in papers:
            if p.embedding is not None:
                arr = np.asarray(p.embedding, dtype=np.float32)
                if arr.ndim == 1 and arr.size > 0:
                    vec_list.append(arr)
                    continue
            vec_list.append(np.zeros(768, dtype=np.float32))

        X = np.vstack(vec_list)
        if float(np.linalg.norm(X)) == 0.0:
            X = TfidfVectorizer(max_features=128, stop_words="english").fit_transform(docs).toarray()

        k = max(1, min(len(papers), round(math.sqrt(len(papers)))))
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)

        tfidf = TfidfVectorizer(max_features=300, stop_words="english", ngram_range=(1, 2))
        try:
            tf_mat = tfidf.fit_transform(docs).toarray()
            vocab = np.array(tfidf.get_feature_names_out())
        except ValueError:
            tf_mat = None
            vocab = np.array([])

        clusters: list[dict[str, Any]] = []
        for cid in range(k):
            idxs = [idx for idx, lbl in enumerate(labels) if int(lbl) == cid]
            if not idxs:
                continue
            top_terms: list[str] = []
            if tf_mat is not None and len(vocab) > 0:
                centroid_tf = tf_mat[idxs].mean(axis=0)
                top_idx = np.argsort(centroid_tf)[::-1][:3]
                top_terms = [str(vocab[t]) for t in top_idx if centroid_tf[t] > 0]
            if not top_terms:
                top_terms = list({t for i in idxs for t in (papers[i].tags or [])})[:3] or ["Research Theme"]

            clusters.append(
                {
                    "clusterId": cid,
                    "label": " / ".join(w.title() for w in top_terms),
                    "paperIds": [papers[i].id for i in idxs],
                    "paperTitles": [papers[i].title for i in idxs],
                    "topTerms": top_terms,
                }
            )

        return {"collectionId": collection_id, "clusters": clusters}

    # ── 7.2 Research Debt Tracker ───────────────────────────────────────

    def research_debt(self, user_id: str, aging_days: int = 30) -> dict[str, Any]:
        saves = list(
            self.db.scalars(
                select(SavedPaper)
                .where(SavedPaper.user_id == user_id)
                .order_by(SavedPaper.created_at)
            ).all()
        )
        if not saves:
            return {"totalSaved": 0, "debtCount": 0, "overdueCount": 0, "items": []}

        paper_ids = [s.paper_id for s in saves]
        papers = {p.id: p for p in self.db.scalars(select(Paper).where(Paper.id.in_(paper_ids))).all()}

        # Papers that have a workspace note or project link
        user_projects = list(self.db.scalars(select(Project).where(Project.owner_id == user_id)).all())
        proj_ids = [pr.id for pr in user_projects]
        noted_pids = set(
            self.db.scalars(
                select(ProjectNote.paper_id).where(
                    ProjectNote.project_id.in_(proj_ids), ProjectNote.paper_id.is_not(None)
                )
            ).all()
        ) if proj_ids else set()
        linked_pids = set(
            self.db.scalars(
                select(ProjectPaper.paper_id).where(ProjectPaper.project_id.in_(proj_ids))
            ).all()
        ) if proj_ids else set()

        now = datetime.now(UTC)
        items: list[dict[str, Any]] = []
        overdue_count = 0

        for s in saves:
            p = papers.get(s.paper_id)
            if p is None:
                continue
            missing: list[str] = []
            if p.id not in noted_pids:
                missing.append("workspace_note")
            if p.id not in linked_pids:
                missing.append("project_link")
            if not p.ai_summary:
                missing.append("ai_summary")

            if not missing:
                continue

            created = s.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            age_days = max(0, (now - created).days)
            is_overdue = age_days >= aging_days
            if is_overdue:
                overdue_count += 1

            items.append(
                {
                    "paperId": p.id,
                    "title": p.title,
                    "venue": p.venue,
                    "year": p.year,
                    "savedAt": created.date().isoformat(),
                    "ageDays": age_days,
                    "isOverdue": is_overdue,
                    "missing": missing,
                    "debtReason": (
                        f"Saved {age_days}d ago with no "
                        + ", ".join(m.replace("_", " ") for m in missing)
                    ),
                }
            )

        items.sort(key=lambda x: (x["isOverdue"], len(x["missing"]), x["ageDays"]), reverse=True)
        return {
            "totalSaved": len(saves),
            "debtCount": len(items),
            "overdueCount": overdue_count,
            "items": items,
        }

    # ── 7.3 Cross-Collection Insight ────────────────────────────────────

    def cross_collection_insights(self, user_id: str, limit: int = 6) -> list[dict[str, Any]]:
        cols = list(self.db.scalars(select(Collection).where(Collection.owner_id == user_id)).all())
        if len(cols) < 2:
            return []

        all_papers = list(self.db.scalars(select(Paper).where(Paper.embedding.is_not(None))).all())
        if not all_papers:
            return []
        paper_by_id = {p.id: p for p in all_papers}

        centroids: list[tuple[Collection, set[str], np.ndarray]] = []
        for c in cols:
            pids = set(
                self.db.scalars(
                    select(CollectionPaper.paper_id).where(CollectionPaper.collection_id == c.id)
                ).all()
            )
            vecs = [
                np.asarray(paper_by_id[pid].embedding, dtype=np.float32)
                for pid in pids
                if pid in paper_by_id and paper_by_id[pid].embedding is not None
            ]
            if not vecs:
                continue
            cent = np.mean(np.vstack(vecs), axis=0)
            norm = float(np.linalg.norm(cent))
            if norm > 0:
                cent /= norm
            centroids.append((c, pids, cent))

        if len(centroids) < 2:
            return []

        insights: list[dict[str, Any]] = []
        for idx_a in range(len(centroids)):
            for idx_b in range(idx_a + 1, len(centroids)):
                col_a, pids_a, cent_a = centroids[idx_a]
                col_b, pids_b, cent_b = centroids[idx_b]
                both = pids_a & pids_b

                best_bridges: list[tuple[float, Paper]] = []
                for p in all_papers:
                    if p.id in both or p.embedding is None:
                        continue
                    v = np.asarray(p.embedding, dtype=np.float32)
                    s_a = _cos(v, cent_a)
                    s_b = _cos(v, cent_b)
                    bridge = min(s_a, s_b)
                    if bridge > 0.15:
                        best_bridges.append((bridge, p))

                best_bridges.sort(key=lambda t: t[0], reverse=True)
                if best_bridges:
                    top_p = [
                        {
                            "id": p.id,
                            "title": p.title,
                            "year": p.year,
                            "venue": p.venue,
                            "bridgeScore": round(score, 4),
                        }
                        for score, p in best_bridges[:3]
                    ]
                    insights.append(
                        {
                            "collectionA": {"id": col_a.id, "name": col_a.name},
                            "collectionB": {"id": col_b.id, "name": col_b.name},
                            "headline": f"Papers bridging '{col_a.name}' and '{col_b.name}'",
                            "bridgingPapers": top_p,
                        }
                    )

        return insights[:limit]

    # ── 7.5 Collaboration Affinity Score ────────────────────────────────

    def researcher_affinity(self, user_id: str) -> dict[str, dict[str, Any]]:
        profile = self.db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
        user_interests = {str(i).lower() for i in (profile.interests if profile and profile.interests else [])}
        user_field = (profile.field if profile else "").lower()

        saved_pids = list(
            self.db.scalars(select(SavedPaper.paper_id).where(SavedPaper.user_id == user_id)).all()
        )
        saved_papers = list(self.db.scalars(select(Paper).where(Paper.id.in_(saved_pids))).all()) if saved_pids else []
        saved_tags = {str(t).lower() for p in saved_papers for t in (p.tags or [])}
        saved_domains = {p.domain.lower() for p in saved_papers if p.domain}

        user_tokens = user_interests | saved_tags | saved_domains
        if user_field:
            user_tokens.update(w for w in re.findall(r"[a-z]{3,}", user_field))

        researchers = list(self.db.scalars(select(ResearcherProfile)).all())
        result: dict[str, dict[str, Any]] = {}

        for r in researchers:
            r_tokens = {str(i).lower() for i in (r.interests or [])}
            r_field_words = set(re.findall(r"[a-z]{3,}", (r.field or "").lower()))
            r_all = r_tokens | r_field_words

            if not user_tokens or not r_all:
                score = 42
                shared: list[str] = []
            else:
                overlap = sorted(user_tokens & r_all)
                jaccard = len(overlap) / max(1, len(user_tokens | r_all))
                field_bonus = 0.25 if (user_field and r.field and user_field.split()[0] in r.field.lower()) else 0.0
                score = min(98, max(28, int(round((0.35 + 0.65 * min(1.0, jaccard * 3.0 + field_bonus)) * 100))))
                shared = overlap[:4]

            result[r.id] = {
                "researcherId": r.id,
                "affinityScore": score,
                "tier": "High" if score >= 70 else ("Medium" if score >= 45 else "Emerging"),
                "sharedTopics": shared,
                "reason": (
                    f"Shared focus on {', '.join(shared)}"
                    if shared
                    else f"Complementary work in {r.field}"
                ),
            }
        return result

    # ── 7.7 Research Velocity Dashboard ─────────────────────────────────

    def research_velocity(self, user_id: str) -> dict[str, Any]:
        now = datetime.now(UTC)
        months: list[str] = []
        month_buckets: dict[str, dict[str, int]] = {}
        for offset in range(5, -1, -1):
            m_dt = (now.replace(day=1) - timedelta(days=offset * 28)).replace(day=1)
            key = m_dt.strftime("%Y-%m")
            if key not in month_buckets:
                months.append(key)
                month_buckets[key] = {
                    "month": m_dt.strftime("%b"),
                    "papersSaved": 0,
                    "collectionsCreated": 0,
                    "notesWritten": 0,
                    "tasksDone": 0,
                    "manuscriptsSubmitted": 0,
                }

        def _bucket(dt: datetime | None) -> dict[str, int] | None:
            if dt is None:
                return month_buckets[months[-1]]
            k = dt.strftime("%Y-%m")
            return month_buckets.get(k) or month_buckets[months[-1]]

        saves = list(self.db.scalars(select(SavedPaper).where(SavedPaper.user_id == user_id)).all())
        cols = list(self.db.scalars(select(Collection).where(Collection.owner_id == user_id)).all())
        proj_ids = list(self.db.scalars(select(Project.id).where(Project.owner_id == user_id)).all())
        notes = list(self.db.scalars(select(ProjectNote).where(ProjectNote.project_id.in_(proj_ids))).all()) if proj_ids else []
        tasks = list(
            self.db.scalars(
                select(ProjectTask).where(ProjectTask.project_id.in_(proj_ids), ProjectTask.status == "Done")
            ).all()
        ) if proj_ids else []
        manuscripts = list(self.db.scalars(select(Manuscript).where(Manuscript.submitter_id == user_id)).all())

        for s in saves:
            _bucket(s.created_at)["papersSaved"] += 1
        for c in cols:
            _bucket(c.created_at)["collectionsCreated"] += 1
        for n in notes:
            _bucket(n.created_at)["notesWritten"] += 1
        for t in tasks:
            _bucket(t.updated_at)["tasksDone"] += 1
        for m in manuscripts:
            _bucket(m.created_at)["manuscriptsSubmitted"] += 1

        series = [month_buckets[m] for m in months]
        return {
            "totals": {
                "papersSaved": len(saves),
                "collections": len(cols),
                "notes": len(notes),
                "tasksDone": len(tasks),
                "manuscripts": len(manuscripts),
            },
            "series": series,
        }


__all__ = ["InsightsService"]
