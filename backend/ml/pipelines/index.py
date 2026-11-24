"""Indexing pipeline — encode, NER, RE, KG build, and DB persistence.

Layers 2-5 of the CKGAI pipeline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ml.graph import EntityResolver, KGBuilder
from ml.graph.entity_resolution import normalize_name
from ml.ner import NERPipeline
from ml.relation import REPipeline
from ml.serving.embed_store import embed_entities, embed_papers

log = logging.getLogger(__name__)


@dataclass
class IndexResult:
    papers_indexed: int = 0
    entities_extracted: int = 0
    relations_extracted: int = 0
    kg_nodes_created: int = 0
    kg_edges_created: int = 0
    errors: list[str] = field(default_factory=list)


class IndexPipeline:
    """Index papers into the knowledge graph and persist to PostgreSQL/SQLite."""

    def __init__(self, db_session=None, device: str = "cpu"):
        self.db = db_session
        self.ner = NERPipeline(device=device)
        self.re = REPipeline(device=device)
        self.kg_builder = KGBuilder(EntityResolver())

    def index_paper(
        self,
        paper_id: str,
        title: str,
        abstract: str,
        domain: str = "",
    ) -> IndexResult:
        """Index a single paper through NER → RE → KGBuilder → DB persistence."""
        result = IndexResult()

        try:
            text = f"{title}. {abstract}"
            sentences = self._split_sentences(text)

            all_entities = []
            entities_per_sentence = []
            for sent in sentences:
                ents = self.ner.extract_entities(sent)
                entities_per_sentence.append(ents)
                all_entities.extend(ents)
            result.entities_extracted = len(all_entities)

            all_relations = []
            relation_sentences: dict[tuple[str, str, str], str] = {}
            for sent, ents in zip(sentences, entities_per_sentence):
                ent_dicts = [
                    {"text": e.text, "label": e.label, "start": e.start, "end": e.end}
                    for e in ents
                ]
                rels = self.re.extract_relations(sent, ent_dicts)
                for r in rels:
                    relation_sentences[(r.head, r.relation, r.tail)] = sent[:500]
                all_relations.extend(rels)
            result.relations_extracted = len(all_relations)

            rel_dicts = [
                {
                    "head": r.head,
                    "tail": r.tail,
                    "relation": r.relation,
                    "confidence": r.confidence,
                    "sentence": relation_sentences.get((r.head, r.relation, r.tail), ""),
                }
                for r in all_relations
            ]
            ent_dicts_full = [
                {"text": e.text, "label": e.label, "confidence": e.confidence}
                for e in all_entities
            ]

            build_result = self.kg_builder.build_from_extraction(
                entities=ent_dicts_full,
                relations=rel_dicts,
                paper_id=paper_id,
                domain=domain or "Computer Science",
            )

            if self.db is not None and ent_dicts_full:
                created_n, created_e = self._persist_to_db(
                    paper_id=paper_id,
                    domain=domain or "Computer Science",
                    entities=ent_dicts_full,
                    relations=rel_dicts,
                )
                result.kg_nodes_created = created_n
                result.kg_edges_created = created_e
            else:
                result.kg_nodes_created = build_result.nodes_created
                result.kg_edges_created = build_result.edges_created

            result.papers_indexed = 1
        except Exception as exc:  # noqa: BLE001
            result.errors.append(str(exc))
            log.error("index_paper_failed: paper_id=%s error=%s", paper_id, exc)

        return result

    def _persist_to_db(
        self,
        *,
        paper_id: str,
        domain: str,
        entities: list[dict],
        relations: list[dict],
    ) -> tuple[int, int]:
        from sqlalchemy import select

        from app.models.knowledge import KgEdge, KgNode

        nodes_created = 0
        edges_created = 0
        name_to_node: dict[str, KgNode] = {}

        # Deduplicate entities in this paper
        unique_ents: dict[tuple[str, str], dict] = {}
        for ent in entities:
            norm = normalize_name(ent["text"])
            etype = ent.get("label", "METHOD")
            if norm:
                unique_ents.setdefault((norm, etype), ent)

        missing_texts: list[str] = []
        missing_nodes: list[KgNode] = []

        for (norm, etype), ent in unique_ents.items():
            row = self.db.scalar(
                select(KgNode).where(KgNode.normalised_name == norm, KgNode.entity_type == etype)
            )
            if row is None:
                row = KgNode(
                    canonical_name=ent["text"][:512],
                    normalised_name=norm[:512],
                    entity_type=etype[:40],
                    domain=(domain or "Computer Science")[:120],
                    paper_id=paper_id,
                    confidence=float(ent.get("confidence", 0.9)),
                    mention_count=1,
                    aliases=[],
                )
                self.db.add(row)
                self.db.flush()
                nodes_created += 1
                missing_texts.append(f"{row.canonical_name} ({row.entity_type})")
                missing_nodes.append(row)
            else:
                row.mention_count = int(row.mention_count or 0) + 1
                if domain and row.domain and row.domain != domain:
                    pass
                elif domain and not row.domain:
                    row.domain = domain[:120]
            name_to_node[ent["text"]] = row

        if missing_texts:
            try:
                vecs = embed_entities(missing_texts)
                for node, vec in zip(missing_nodes, vecs, strict=False):
                    node.embedding = [float(x) for x in vec]
            except Exception:  # noqa: BLE001
                pass

        for rel in relations:
            src = name_to_node.get(rel["head"])
            dst = name_to_node.get(rel["tail"])
            if src is None or dst is None or src.id == dst.id:
                continue
            existing = self.db.scalar(
                select(KgEdge).where(
                    KgEdge.src_id == src.id,
                    KgEdge.dst_id == dst.id,
                    KgEdge.relation == rel["relation"],
                )
            )
            if existing is not None:
                existing.confidence = max(float(existing.confidence), float(rel.get("confidence", 0.8)))
                continue

            src_dom = src.domain or domain or ""
            dst_dom = dst.domain or domain or ""
            edge = KgEdge(
                src_id=src.id,
                dst_id=dst.id,
                relation=rel["relation"][:60],
                confidence=float(rel.get("confidence", 0.8)),
                provenance_paper_id=paper_id,
                provenance_sentence=rel.get("sentence") or None,
                src_domain=src_dom[:120],
                dst_domain=dst_dom[:120],
                domain_cross=bool(src_dom and dst_dom and src_dom != dst_dom),
            )
            self.db.add(edge)
            edges_created += 1

        self.db.commit()
        return nodes_created, edges_created

    def index_all_missing(self, limit: int = 200) -> IndexResult:
        """Index stored papers into kg_nodes / kg_edges."""
        from sqlalchemy import select

        from app.models.paper import Paper

        if self.db is None:
            return IndexResult(errors=["no db session"])
        papers = list(self.db.scalars(select(Paper).limit(limit)).all())
        total = IndexResult()
        for p in papers:
            r = self.index_paper(p.id, p.title, p.abstract, domain=p.domain)
            total.papers_indexed += r.papers_indexed
            total.entities_extracted += r.entities_extracted
            total.relations_extracted += r.relations_extracted
            total.kg_nodes_created += r.kg_nodes_created
            total.kg_edges_created += r.kg_edges_created
            total.errors.extend(r.errors)
        return total

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if len(s.strip()) > 10]

    def index_batch(self, papers: list[dict]) -> IndexResult:
        total = IndexResult()
        for paper in papers:
            r = self.index_paper(
                paper_id=paper.get("id", ""),
                title=paper.get("title", ""),
                abstract=paper.get("abstract", ""),
                domain=paper.get("domain", ""),
            )
            total.papers_indexed += r.papers_indexed
            total.entities_extracted += r.entities_extracted
            total.relations_extracted += r.relations_extracted
            total.kg_nodes_created += r.kg_nodes_created
            total.kg_edges_created += r.kg_edges_created
            total.errors.extend(r.errors)
        return total

    def warmup(self) -> None:
        self.ner.warmup()
        self.re.warmup()


__all__ = ["IndexPipeline", "IndexResult"]
