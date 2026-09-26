"""Virtual global node — gives every node global context in 1 hop.

Connected to all nodes, so after 1 layer of message passing, every node
has information from the entire graph. Combined with L=3 layers, this
effectively captures long-range dependencies without over-smoothing.
"""

from __future__ import annotations

import numpy as np


class VirtualNode:
    """Virtual global node for context aggregation."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.embedding = np.zeros(dim, dtype=np.float32)

    def aggregate(self, H: np.ndarray) -> np.ndarray:
        """Update virtual node embedding from all node embeddings."""
        self.embedding = H.mean(axis=0)
        return self.embedding

    def broadcast(self, num_nodes: int) -> np.ndarray:
        """Broadcast virtual node embedding to all nodes."""
        return np.tile(self.embedding, (num_nodes, 1))
