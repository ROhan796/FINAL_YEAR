"""Indexing pipeline — encode, NER, RE, KG build, store.

Layers 2-5 of the CKGAI pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ml.ner import NERPipeline
from ml.relation import REPipeline
from ml.graph import KGBuilder, EntityResolver
from ml.serving.embed_store import embed_papers, embed_entities

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
    """Index papers into the knowledge graph.

    Steps:
    1. Preprocess text (normalize, segment, tokenize)
    2. Encode paper embeddings
    3. Run NER on abstracts/text
    4. Run RE on entity pairs
    5. Build knowledge graph
    6. Store embeddings and graph in Postgres
    """

    def __init__(self, db_session=None, device: str = "cpu"):
        self.db = db_session
        self.ner = NERPipeline(device=device)
        self.re = REPipeline(device=device)
        self.kg_builder = KGBuilder(EntityResolver())

    def index_paper(self, paper_id: str, title: str, abstract: str,
                    domain: str = "") -> IndexResult:
        """Index a single paper through the full pipeline."""
        result = IndexResult()

        try:
            # Step 1: Preprocess
            text = f"{title}. {abstract}"
            sentences = self._split_sentences(text)

            # Step 2: Encode paper embedding
            paper_text = f"Title: {title} Abstract: {abstract}"
            embedding = embed_papers([paper_text])[0]

            # Step 3: NER
            all_entities = []
            entities_per_sentence = []
            for sent in sentences:
                ents = self.ner.extract_entities(sent)
                entities_per_sentence.append(ents)
                all_entities.extend(ents)
            result.entities_extracted = len(all_entities)

            # Step 4: RE
            entity_dicts = [
                {"text": e.text, "label": e.label, "start": e.start, "end": e.end}
                for e in all_entities
            ]
            all_relations = []
            for sent, ents in zip(sentences, entities_per_sentence):
                ent_dicts = [
                    {"text": e.text, "label": e.label, "start": e.start, "end": e.end}
                    for e in ents
                ]
                rels = self.re.extract_relations(sent, ent_dicts)
                all_relations.extend(rels)
            result.relations_extracted = len(all_relations)

            # Step 5: Build KG
            rel_dicts = [
                {"head": r.head, "tail": r.tail, "relation": r.relation,
                 "confidence": r.confidence}
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
                domain=domain,
            )
            result.kg_nodes_created = build_result.nodes_created
            result.kg_edges_created = build_result.edges_created
            result.papers_indexed = 1

        except Exception as exc:
            result.errors.append(str(exc))
            log.error("index_paper_failed: paper_id=%s error=%s", paper_id, exc)

        return result

    def _split_sentences(self, text: str) -> list[str]:
        """Simple sentence splitter."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if len(s.strip()) > 10]

    def index_batch(self, papers: list[dict]) -> IndexResult:
        """Index a batch of papers."""
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
