"""Knowledge Graph construction and management.

PostgreSQL is the source of truth. Neo4j is an optional projected read model.
Entity resolution merges duplicate entities using:
1. Normalization + exact matching
2. Embedding cosine similarity > threshold
3. Ontology lookup (MeSH/UMLS when available)
"""

from __future__ import annotations

from ml.graph.builder import KGBuilder
from ml.graph.entity_resolution import EntityResolver
from ml.graph.repos import KGRepository

__all__ = ["KGBuilder", "EntityResolver", "KGRepository"]
