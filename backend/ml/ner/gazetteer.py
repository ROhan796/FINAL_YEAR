"""Dictionary-based NER fallback.

Provides a gazetteer of known scientific terms for entity extraction
when the transformer model is unavailable.

This is a DEVELOPMENT / PLACEHOLDER component, not a trained model.
"""

from __future__ import annotations

from ml.ner.model import GazetteerNER
from ml.ner.tags import ExtractedEntity

__all__ = ["GazetteerNER", "ExtractedEntity"]
