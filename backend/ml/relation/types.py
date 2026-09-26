"""Relation types for scientific relationship extraction.

8 relation types as specified in the CKGAI document, plus no_relation.
"""

from __future__ import annotations

from dataclasses import dataclass

RELATIONS = (
    "used_in",
    "derived_from",
    "applied_to",
    "improves",
    "treats",
    "causes",
    "measures",
    "contains",
)

RELATION_LABELS = list(RELATIONS) + ["no_relation"]
REL2ID: dict[str, int] = {r: i for i, r in enumerate(RELATION_LABELS)}
ID2REL: dict[int, str] = {i: r for r, i in REL2ID.items()}

NUM_RELATIONS = len(RELATION_LABELS)  # 9

# Type constraints: which entity type pairs can have which relations
TYPE_CONSTRAINTS: dict[str, tuple[tuple[str, str], ...]] = {
    "used_in": (("METHOD", "DOMAIN"), ("ALGORITHM", "DOMAIN"), ("METHOD", "METHOD")),
    "derived_from": (("METHOD", "METHOD"), ("ALGORITHM", "ALGORITHM"), ("METHOD", "ALGORITHM")),
    "applied_to": (("METHOD", "DISEASE"), ("MATERIAL", "DISEASE"), ("METHOD", "PHENOMENON")),
    "improves": (("METHOD", "METRIC"), ("MATERIAL", "METRIC"), ("ALGORITHM", "METRIC")),
    "treats": (("CHEMICAL", "DISEASE"), ("METHOD", "DISEASE"), ("MATERIAL", "DISEASE")),
    "causes": (("CHEMICAL", "PHENOMENON"), ("METHOD", "DISEASE"), ("MATERIAL", "PHENOMENON")),
    "measures": (("METRIC", "PHENOMENON"), ("METRIC", "DISEASE"), ("METHOD", "METRIC")),
    "contains": (("MATERIAL", "CHEMICAL"), ("METHOD", "METHOD"), ("MATERIAL", "MATERIAL")),
}


@dataclass(frozen=True)
class ExtractedRelation:
    """A single extracted relationship."""
    head: str       # head entity text
    tail: str       # tail entity text
    relation: str   # relation type
    confidence: float = 1.0
    head_type: str = ""
    tail_type: str = ""


def are_type_compatible(head_type: str, tail_type: str, relation: str) -> bool:
    """Check if entity type pair is allowed for the given relation.

    Returns True if no constraint is defined (open pairing).
    """
    constraints = TYPE_CONSTRAINTS.get(relation)
    if not constraints:
        return True
    return any(
        (head_type == c[0] or c[0] == "DOMAIN") and
        (tail_type == c[1] or c[1] == "DOMAIN")
        for c in constraints
    )
