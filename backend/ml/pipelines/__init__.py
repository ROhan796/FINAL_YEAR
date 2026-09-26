"""ML pipeline orchestration.

Pipelines coordinate the 7 CKGAI layers:
1. Ingest → fetch papers
2. Preprocess → clean, tokenize, segment
3. Encode → generate embeddings
4. NER → extract entities
5. RE → extract relations
6. KG build → construct knowledge graph
7. Index → store in DB

These are designed to run as background jobs via Arq/Redis.
"""

from __future__ import annotations

from ml.pipelines.ingest import IngestPipeline
from ml.pipelines.index import IndexPipeline
from ml.pipelines.discovery import DiscoveryPipeline

__all__ = ["IngestPipeline", "IndexPipeline", "DiscoveryPipeline"]
