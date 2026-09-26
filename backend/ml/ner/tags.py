"""NER entity types and BIO tag definitions.

8 entity types as specified in the CKGAI document:
METHOD, MATERIAL, DISEASE, ALGORITHM, DATASET, PHENOMENON, METRIC, CHEMICAL

BIO tagging: B-TYPE, I-TYPE, O
"""

from __future__ import annotations

from dataclasses import dataclass

ENTITY_TYPES = (
    "METHOD",
    "MATERIAL",
    "DISEASE",
    "ALGORITHM",
    "DATASET",
    "PHENOMENON",
    "METRIC",
    "CHEMICAL",
)

# Generate BIO tags
TAG_NAMES = ["O"]
for _et in ENTITY_TYPES:
    TAG_NAMES.extend([f"B-{_et}", f"I-{_et}"])

TAG2ID: dict[str, int] = {tag: i for i, tag in enumerate(TAG_NAMES)}
ID2TAG: dict[int, str] = {i: tag for tag, i in TAG2ID.items()}

NUM_TAGS = len(TAG_NAMES)  # 17


@dataclass(frozen=True)
class ExtractedEntity:
    """A single extracted entity."""
    text: str
    label: str  # entity type (METHOD, DISEASE, etc.)
    start: int
    end: int
    confidence: float = 1.0


def bio_tags_to_entities(tags: list[str], tokens: list[str]) -> list[ExtractedEntity]:
    """Convert BIO tag sequence to entity spans."""
    entities: list[ExtractedEntity] = []
    current_label: str | None = None
    current_tokens: list[str] = []
    current_start: int = 0

    for i, (tag, token) in enumerate(zip(tags, tokens)):
        if tag.startswith("B-"):
            if current_label is not None:
                entities.append(ExtractedEntity(
                    text=" ".join(current_tokens),
                    label=current_label,
                    start=current_start,
                    end=i,
                ))
            current_label = tag[2:]
            current_tokens = [token]
            current_start = i
        elif tag.startswith("I-") and current_label == tag[2:]:
            current_tokens.append(token)
        else:
            if current_label is not None:
                entities.append(ExtractedEntity(
                    text=" ".join(current_tokens),
                    label=current_label,
                    start=current_start,
                    end=i,
                ))
                current_label = None
                current_tokens = []

    if current_label is not None:
        entities.append(ExtractedEntity(
            text=" ".join(current_tokens),
            label=current_label,
            start=current_start,
            end=len(tokens),
        ))

    return entities
