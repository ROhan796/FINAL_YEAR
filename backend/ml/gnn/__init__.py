"""Graph Neural Network reasoning layer.

Implements the CKGAI spec:
- R-GCN (primary) with basis decomposition
- GAT with multi-head attention
- Optional virtual node for global context
- L=3 layers, hidden=256, bases=30, heads=8

When torch is unavailable, provides numpy-based message passing.
"""

from __future__ import annotations

from ml.gnn.model import CKGAI_GNN

__all__ = ["CKGAI_GNN"]
