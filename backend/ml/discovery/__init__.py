"""Discovery / link prediction engine.

4 scorers behind one interface:
- Dot Product
- DistMult (default)
- TransE
- MLP

Cross-domain priority: domain(source) != domain(target)
"""

from __future__ import annotations

from ml.discovery.ranker import DiscoveryRanker

__all__ = ["DiscoveryRanker"]
