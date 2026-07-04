"""Forecasters: producers of per-head probability vectors pi_hat, stacked to
shape (n, onehot_dim). A constant forecaster returns a single row (1, onehot_dim);
that row count is the signal scores use to pick their unconditional fast path.
"""

import numpy as np

from .lattice import Lattice


class MarginalForecaster:
    """Per-head empirical marginals (the unconditional / 'marg' forecaster)."""

    is_constant = True

    def __init__(self, smoothing: float = 0.0):
        self.smoothing = float(smoothing)

    def fit(self, X, labels: np.ndarray, lattice: Lattice):
        labels = np.atleast_2d(np.asarray(labels))
        blocks = []
        for m, size in enumerate(lattice.head_sizes):
            freq = np.bincount(labels[:, m], minlength=size) + self.smoothing
            blocks.append(freq / freq.sum())
        self._probs_row = np.concatenate(blocks)[None, :]  # (1, onehot_dim)
        return self

    def predict_probs(self, X=None) -> np.ndarray:
        return self._probs_row


class LogisticForecaster:
    """One multinomial logistic regression per head (the 'logit' forecaster).
    Heads whose training data misses a class get zero probability there
    (the sklearn classes_ padding), so output is always (n, onehot_dim)."""

    is_constant = False

    def __init__(self, C: float = 1.0, max_iter: int = 2000):
        self.C, self.max_iter = C, max_iter

    def fit(self, X, labels: np.ndarray, lattice: Lattice):
        from sklearn.linear_model import LogisticRegression

        labels = np.atleast_2d(np.asarray(labels))
        self.lattice = lattice
        self._models = []
        for m in range(lattice.n_heads):
            head_labels = labels[:, m]
            if np.unique(head_labels).size < 2:
                # single class in the train fold (subsampling can do this):
                # sklearn refuses to fit, so this head becomes a constant
                # forecaster putting probability 1 on the one observed class.
                self._models.append(int(head_labels[0]))
            else:
                model = LogisticRegression(C=self.C, max_iter=self.max_iter)
                model.fit(X, head_labels)
                self._models.append(model)
        return self

    def predict_probs(self, X) -> np.ndarray:
        n = len(X)
        out = np.zeros((n, self.lattice.onehot_dim))
        for m, model in enumerate(self._models):
            offset = self.lattice.head_offsets[m]
            block = out[:, offset : offset + self.lattice.head_sizes[m]]
            if isinstance(model, int):
                block[:, model] = 1.0
            else:
                block[:, model.classes_] = model.predict_proba(X)
        return out


def residuals(
    lattice: Lattice, probs: np.ndarray, cell_index: np.ndarray
) -> np.ndarray:
    """r = e(y) - pi_hat(x). probs is (n, D) or (1, D) broadcast over the labels."""
    onehots = lattice.onehot_of_index(cell_index)
    return onehots - np.asarray(probs)
