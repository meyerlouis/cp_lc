"""Lattice geometry and label encodings.

Single source of truth for: the canonical cell ordering, label <-> index codes,
and the stacked one-hot embedding e(y). Canonical order = C-order mixed radix
(last head fastest), identical to itertools.product(range(K_1), ..., range(K_M)).
"""
from dataclasses import dataclass
from functools import cached_property
import numpy as np


@dataclass(frozen=True)
class Lattice:
    """Product label space: head m takes values in {0, ..., head_sizes[m]-1}."""
    head_sizes: tuple

    def __post_init__(self):
        sizes = tuple(int(k) for k in self.head_sizes)
        object.__setattr__(self, "head_sizes", sizes)
        if len(sizes) < 1 or any(k < 2 for k in sizes):
            raise ValueError(f"head sizes must all be >= 2, got {sizes}")

    @property
    def n_heads(self) -> int:
        return len(self.head_sizes)

    @property
    def n_cells(self) -> int:
        out = 1
        for k in self.head_sizes:
            out *= k
        return out

    @property
    def onehot_dim(self) -> int:
        return sum(self.head_sizes)

    @cached_property
    def head_offsets(self) -> np.ndarray:
        """Start column of each head's block in the one-hot embedding, shape (n_heads,)."""
        return np.concatenate([[0], np.cumsum(self.head_sizes[:-1])]).astype(np.int64)

    @cached_property
    def cells(self) -> np.ndarray:
        """All label vectors in canonical order, shape (n_cells, n_heads)."""
        return self.decode(np.arange(self.n_cells))

    @cached_property
    def cell_onehots(self) -> np.ndarray:
        """One-hot embedding of every cell, shape (n_cells, onehot_dim)."""
        return self.onehot_of_labels(self.cells)

    # ---------------- codecs ----------------
    def encode(self, labels: np.ndarray) -> np.ndarray:
        """Label vectors (n, n_heads) -> flat cell indices (n,)."""
        labels = np.atleast_2d(np.asarray(labels))
        if labels.shape[1] != self.n_heads:
            raise ValueError(f"expected {self.n_heads} heads, got shape {labels.shape}")
        if (labels < 0).any() or (labels >= np.array(self.head_sizes)).any():
            raise ValueError("label out of range")
        per_head = tuple(labels[:, m] for m in range(self.n_heads))
        return np.ravel_multi_index(per_head, self.head_sizes)

    def decode(self, cell_index: np.ndarray) -> np.ndarray:
        """Flat cell indices (n,) -> label vectors (n, n_heads)."""
        per_head = np.unravel_index(np.asarray(cell_index), self.head_sizes)
        return np.stack(per_head, axis=-1).astype(np.int64)

    # ---------------- embeddings ----------------
    def onehot_of_labels(self, labels: np.ndarray) -> np.ndarray:
        """Label vectors (n, n_heads) -> stacked one-hot e(y), shape (n, onehot_dim)."""
        labels = np.atleast_2d(np.asarray(labels))
        n = labels.shape[0]
        out = np.zeros((n, self.onehot_dim))
        columns = labels + self.head_offsets[None, :]            # (n, n_heads)
        out[np.repeat(np.arange(n), self.n_heads), columns.ravel()] = 1.0
        return out

    def onehot_of_index(self, cell_index: np.ndarray) -> np.ndarray:
        """Flat cell indices (n,) -> e(y), shape (n, onehot_dim)."""
        return self.onehot_of_labels(self.decode(np.asarray(cell_index)))

    # ---------------- test helper ----------------
    def hamming(self, cell_i: int, cell_j: int) -> int:
        return int((self.cells[cell_i] != self.cells[cell_j]).sum())
