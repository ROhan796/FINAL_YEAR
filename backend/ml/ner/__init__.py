"""Named Entity Recognition for scientific text.

8 entity types with BIO tagging and optional CRF layer.
When torch is unavailable, provides a gazetteer-based fallback.
"""

from __future__ import annotations

from ml.ner.model import NERPipeline, NERResult

__all__ = ["NERPipeline", "NERResult"]
