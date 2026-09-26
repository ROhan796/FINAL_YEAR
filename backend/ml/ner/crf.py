"""Linear-chain CRF for NER sequence labeling.

Enforces valid BIO transitions (e.g., I-METHOD cannot follow B-DISEASE).
Uses Viterbi algorithm for optimal decoding.

TRAINING STATUS: Architecture defined. Transition scores require training.
When no trained CRF is available, returns the raw softmax predictions.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class LinearChainCRF:
    """Linear-chain Conditional Random Field.

    The CRF computes:
        P(y|x) ∝ exp(Σ_i [ψ(y_{i-1}, y_i) + z_i[y_i]])

    Where:
        ψ(y_{i-1}, y_i) = transition score between consecutive tags
        z_i[y_i] = emission score from the transformer for tag y_i at position i

    TRAINING STATUS: Architecture defined. Without trained transition weights,
    this class passes through the raw emissions.
    """

    def __init__(self, num_tags: int) -> None:
        self.num_tags = num_tags
        # Transition matrix: transitions[i][j] = score of transitioning from tag i to tag j
        self.transitions: np.ndarray | None = None
        self.start_transitions: np.ndarray | None = None
        self.end_transitions: np.ndarray | None = None

    def load_transitions(self, transitions: np.ndarray,
                         start_transitions: np.ndarray | None = None,
                         end_transitions: np.ndarray | None = None) -> None:
        """Load trained transition scores."""
        self.transitions = transitions
        self.start_transitions = start_transitions
        self.end_transitions = end_transitions

    def viterbi_decode(self, emissions: np.ndarray) -> list[int]:
        """Find the optimal tag sequence using Viterbi algorithm.

        Args:
            emissions: (seq_len, num_tags) array of log-emission scores.

        Returns:
            List of optimal tag indices.
        """
        if self.transitions is None:
            # No trained CRF — just take argmax of emissions
            return emissions.argmax(axis=1).tolist()

        seq_len, num_tags = emissions.shape
        # Viterbi forward pass
        trellis = np.zeros((seq_len, num_tags), dtype=np.float64)
        backpointers = np.zeros((seq_len, num_tags), dtype=np.int32)

        # Initialize with start transitions
        if self.start_transitions is not None:
            trellis[0] = self.start_transitions + emissions[0]
        else:
            trellis[0] = emissions[0]

        # Forward pass
        for t in range(1, seq_len):
            for j in range(num_tags):
                candidates = trellis[t - 1] + self.transitions[:, j]
                backpointers[t, j] = np.argmax(candidates)
                trellis[t, j] = candidates[backpointers[t, j]] + emissions[t, j]

        # Backtrack
        best_last = np.argmax(trellis[-1])
        if self.end_transitions is not None:
            best_last = np.argmax(trellis[-1] + self.end_transitions)

        best_path = [best_last]
        for t in range(seq_len - 1, 0, -1):
            best_path.append(backpointers[t, best_path[-1]])
        best_path.reverse()

        return best_path

    def decode(self, emissions: np.ndarray) -> list[int]:
        """Decode optimal tag sequence (Viterbi or argmax fallback)."""
        return self.viterbi_decode(emissions)


__all__ = ["LinearChainCRF"]
