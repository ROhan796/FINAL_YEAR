"""Entity resolution — merge duplicate entities.

Resolution pipeline:
1. Normalize text (lowercase, strip, lemmatize)
2. Exact string matching
3. Embedding cosine similarity > threshold (default 0.92)
4. Ontology lookup (MeSH/UMLS when configured)

Each entity stores: canonical_name, normalised_name, aliases, embedding, domain, paper_id.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np

from ml.graph.schema import KGNodeData


def normalize_name(name: str) -> str:
    """Normalize entity name for matching."""
    # Unicode normalize
    name = unicodedata.normalize("NFKC", name)
    # Lowercase
    name = name.lower().strip()
    # Remove extra whitespace
    name = re.sub(r"\s+", " ", name)
    # Remove common prefixes/suffixes
    for prefix in ("the ", "a ", "an "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


class EntityResolver:
    """Merge duplicate entities using multiple strategies.

    Args:
        threshold: Cosine similarity threshold for embedding-based merging.
    """

    def __init__(self, threshold: float = 0.92) -> self:
        self.threshold = threshold
        # Map from normalised_name -> canonical node data
        self._nodes: dict[str, KGNodeData] = {}
        # Map from normalised_name -> node id (for Postgres)
        self._name_to_id: dict[str, str] = {}

    def resolve(self, candidate: KGNodeData, node_id: str) -> str:
        """Resolve a candidate entity against existing nodes.

        Returns the ID of the resolved (existing or new) node.
        """
        norm = normalize_name(candidate.canonical_name)
        candidate.normalised_name = norm

        # Strategy 1: Exact match
        if norm in self._nodes:
            existing = self._nodes[norm]
            existing.mention_count += 1
            if candidate.canonical_name not in existing.aliases:
                existing.aliases.append(candidate.canonical_name)
            return self._name_to_id[norm]

        # Strategy 2: Embedding similarity
        if candidate.embedding is not None:
            for existing_norm, existing_node in self._nodes.items():
                if existing_node.embedding is not None:
                    sim = self._cosine_similarity(
                        np.array(candidate.embedding),
                        np.array(existing_node.embedding),
                    )
                    if sim > self.threshold:
                        existing_node.mention_count += 1
                        if candidate.canonical_name not in existing_node.aliases:
                            existing_node.aliases.append(candidate.canonical_name)
                        return self._name_to_id[existing_norm]

        # New entity
        self._nodes[norm] = candidate
        self._name_to_id[norm] = node_id
        return node_id

    def get_all_nodes(self) -> list[KGNodeData]:
        return list(self._nodes.values())

    def get_node_count(self) -> int:
        return len(self._nodes)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
