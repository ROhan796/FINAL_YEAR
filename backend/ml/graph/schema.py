"""Knowledge graph schema — entity and relation type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field

# Entity types (same as NER)
ENTITY_TYPES = (
    "METHOD", "MATERIAL", "DISEASE", "ALGORITHM",
    "DATASET", "PHENOMENON", "METRIC", "CHEMICAL",
)

# Relation types (same as RE)
RELATIONS = (
    "used_in", "derived_from", "applied_to", "improves",
    "treats", "causes", "measures", "contains",
)


@dataclass
class KGNodeData:
    """Data for creating a knowledge graph node."""
    canonical_name: str
    normalised_name: str
    entity_type: str
    domain: str = ""
    embedding: list[float] | None = None
    paper_id: str | None = None
    confidence: float = 1.0
    mention_count: int = 1
    aliases: list[str] = field(default_factory=list)


@dataclass
class KGEdgeData:
    """Data for creating a knowledge graph edge."""
    src_id: str
    dst_id: str
    relation: str
    confidence: float = 1.0
    provenance_paper_id: str | None = None
    provenance_sentence: str | None = None
    src_domain: str = ""
    dst_domain: str = ""
    domain_cross: bool = False
