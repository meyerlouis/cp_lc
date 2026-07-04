"""Conformal primitives: THE quantile convention, set evaluation, fold structure.

`conformal_quantile` is the only place ceil((1-alpha)(n+1)) may appear in the repo.
"""
from dataclasses import dataclass, fields
import math
import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal threshold: the rank-th smallest calibration score,
    rank = ceil((1-alpha)(n+1)). If rank > n the finite-sample guarantee is
    unattainable at this (n, alpha) and the exact convention is the trivial
    threshold +inf (deploy everything). Ties allowed (only conservative)."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n = scores.size
    if n < 1:
        raise ValueError("need at least one calibration score")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    rank = math.ceil((1.0 - alpha) * (n + 1))
    if rank > n:
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


# ---------------- prediction-set evaluation ----------------
def prediction_set_mask(lattice_scores: np.ndarray, threshold: float) -> np.ndarray:
    """{y : score(y) <= threshold}. Shape (L,) or (batch, L) -> bool, same shape."""
    return np.asarray(lattice_scores) <= threshold


def prediction_set_sizes(lattice_scores: np.ndarray, threshold: float) -> np.ndarray:
    return prediction_set_mask(lattice_scores, threshold).sum(axis=-1)


def exact_size_and_coverage(lattice_scores: np.ndarray, threshold: float,
                            cell_probs: np.ndarray):
    """Population size and coverage of the set against a known law (L,)."""
    mask = prediction_set_mask(np.asarray(lattice_scores).ravel(), threshold)
    cell_probs = np.asarray(cell_probs, dtype=np.float64).ravel()
    if mask.shape != cell_probs.shape:
        raise ValueError("lattice_scores and cell_probs shape mismatch")
    return int(mask.sum()), float(cell_probs[mask].sum())


def empirical_coverage(point_scores: np.ndarray, threshold: float) -> float:
    return float((np.asarray(point_scores) <= threshold).mean())


# ---------------- folds ----------------
@dataclass(frozen=True)
class Splits:
    """Disjoint fold indices.
    train         : fits forecaster + scores
    sel_threshold : labels for the provisional (stage-1) thresholds
    sel_size      : fresh covariates for stage-1 size estimates (empty if unconditional)
    cal           : final calibration
    test          : held-out evaluation (empty if evaluating exactly against a law)
    """
    train: np.ndarray
    sel_threshold: np.ndarray
    sel_size: np.ndarray
    cal: np.ndarray
    test: np.ndarray

    def __post_init__(self):
        for f in fields(self):
            arr = np.asarray(getattr(self, f.name), dtype=np.int64)
            object.__setattr__(self, f.name, arr)
            if arr.ndim != 1 or (arr.size and arr.min() < 0):
                raise ValueError(f"fold '{f.name}' must be 1-D nonnegative indices")
        all_indices = np.concatenate([getattr(self, f.name) for f in fields(self)])
        if all_indices.size != np.unique(all_indices).size:
            raise ValueError("folds overlap: indices must be pairwise disjoint")

    @property
    def n_total(self) -> int:
        return sum(getattr(self, f.name).size for f in fields(self))


def make_splits(n: int, fractions: dict, rng: np.random.Generator) -> Splits:
    """Partition range(n) by `fractions` (keys among the five fold names, values
    summing to 1). Largest-remainder rounding so counts sum exactly to n."""
    fold_names = ["train", "sel_threshold", "sel_size", "cal", "test"]
    frac = {name: float(fractions.get(name, 0.0)) for name in fold_names}
    if abs(sum(frac.values()) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {sum(frac.values())}")
    target = {name: frac[name] * n for name in fold_names}
    counts = {name: int(math.floor(target[name])) for name in fold_names}
    deficit = n - sum(counts.values())
    by_remainder = sorted(fold_names, key=lambda k: target[k] - counts[k], reverse=True)
    for name in by_remainder[:deficit]:
        counts[name] += 1
    perm = rng.permutation(n)
    folds, position = {}, 0
    for name in fold_names:
        folds[name] = np.sort(perm[position:position + counts[name]])
        position += counts[name]
    return Splits(**folds)
