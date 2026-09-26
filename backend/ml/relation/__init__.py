"""Relation Extraction for scientific entity pairs.

8 relation types + no_relation.
Entity pair representation: [h_e1 || h_e2 || h_e1⊙h_e2 || h_cls]

TRAINING STATUS: Architecture defined. No trained checkpoint included.
Fallback uses co-occurrence scoring.
"""

from __future__ import annotations

from ml.relation.model import REPipeline, REResult

__all__ = ["REPipeline", "REResult"]
