"""CKGAI ML Pipeline — Cross-Domain Knowledge Graph AI.

Seven-layer pipeline:
1. Ingest — arXiv, PubMed, Semantic Scholar
2. Preprocess — normalize, segment, tokenize
3. Encode — SciBERT (passage/entity), SPECTER2 (paper-level)
4. NER + RE — entity and relationship extraction
5. Graph — knowledge graph construction + entity resolution
6. GNN — R-GCN + GAT reasoning
7. Discovery — link prediction + cross-domain ranking

Every layer is independently testable. torch is lazy-imported so
the API boots without ML dependencies.
"""

from __future__ import annotations

__version__ = "0.1.0"
