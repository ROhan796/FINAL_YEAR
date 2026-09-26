"""Knowledge graph builder — assembles entities and relationships into a graph.

Takes extracted entities and relations from NER/RE and builds the knowledge
graph in PostgreSQL (source of truth). Each edge stores provenance:
- source paper ID
- source sentence
- extraction confidence
- relation type
- timestamps
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import numpy as np

from ml.graph.entity_resolution import EntityResolver, normalize_name
from ml.graph.schema import KGEdgeData, KGNodeData

log = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Result of a KG build operation."""
    nodes_created: int = 0
    edges_created: int = 0
    nodes_merged: int = 0
    node_ids: dict[str, str] = None  # canonical_name -> node_id

    def __post_init__(self):
        if self.node_ids is None:
            self.node_ids = {}


class KGBuilder:
    """Build knowledge graph from extracted entities and relations.

    Usage:
        builder = KGBuilder(resolver=EntityResolver(threshold=0.92))
        result = builder.build_from_extraction(
            entities=[...],
            relations=[...],
            paper_id="paper-123",
            domain="Computer Science",
        )
    """

    def __init__(self, resolver: EntityResolver | None = None) -> None:
        self.resolver = resolver or EntityResolver()
        self._edges: list[KGEdgeData] = []

    def build_from_extraction(
        self,
        entities: list[dict],
        relations: list[dict],
        paper_id: str = "",
        domain: str = "",
        embedding_fn=None,
    ) -> BuildResult:
        """Build graph from extracted entities and relations.

        Args:
            entities: List of entity dicts with 'text', 'label', optionally 'embedding'.
            relations: List of relation dicts with 'head', 'tail', 'relation', 'confidence'.
            paper_id: Source paper ID for provenance.
            domain: Source paper domain.
            embedding_fn: Optional function to compute embeddings for entity names.
        """
        result = BuildResult()

        # Create KGNodeData for each entity
        name_to_id: dict[str, str] = {}
        for ent in entities:
            node_id = str(uuid.uuid4())
            embedding = None
            if embedding_fn:
                try:
                    embedding = embedding_fn(ent["text"])[0].tolist()
                except Exception:
                    pass

            node_data = KGNodeData(
                canonical_name=ent["text"],
                normalised_name=normalize_name(ent["text"]),
                entity_type=ent.get("label", ""),
                domain=domain,
                embedding=embedding,
                paper_id=paper_id or None,
                confidence=ent.get("confidence", 1.0),
            )

            resolved_id = self.resolver.resolve(node_data, node_id)
            name_to_id[ent["text"]] = resolved_id

            if resolved_id == node_id:
                result.nodes_created += 1
            else:
                result.nodes_merged += 1

        # Create edges from relations
        for rel in relations:
            head_id = name_to_id.get(rel["head"])
            tail_id = name_to_id.get(rel["tail"])
            if not head_id or not tail_id:
                continue

            edge = KGEdgeData(
                src_id=head_id,
                dst_id=tail_id,
                relation=rel["relation"],
                confidence=rel.get("confidence", 1.0),
                provenance_paper_id=paper_id or None,
                provenance_sentence=rel.get("sentence"),
                src_domain=domain,
                dst_domain=domain,
                domain_cross=False,
            )
            self._edges.append(edge)
            result.edges_created += 1

        result.node_ids = name_to_id
        return result

    def get_edges(self) -> list[KGEdgeData]:
        return self._edges

    def clear(self) -> None:
        self._edges.clear()
