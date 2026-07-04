"""Closed-form theoretical quantities (session record): oracle sizes, the
phase-transition lower envelope, and the blindness constant. Used as figure
overlays and as bounds the simulations must respect in tests."""
import math
import numpy as np


def n_alpha(support_size: int, alpha: float) -> int:
    """Oracle deterministic set size on a uniform support: ceil((1-alpha) t)."""
    return math.ceil((1.0 - alpha) * support_size)


def oracle_size(cell_probs: np.ndarray, alpha: float) -> int:
    """Smallest deterministic set with mass >= 1 - alpha under a known law."""
    sorted_probs = np.sort(np.asarray(cell_probs))[::-1]
    return int(np.searchsorted(np.cumsum(sorted_probs), 1 - alpha - 1e-12) + 1)


def phase_lower_envelope(budget: int, support_size: int, alpha: float) -> float:
    """Lower bound on E|C| / n_cells for ANY procedure valid over all uniform
    t-subset laws, with total label budget N:  max((1 - 1/t)^N - alpha, 0)."""
    return max((1.0 - 1.0 / support_size) ** budget - alpha, 0.0)


def blindness_constant(head_sizes, ridge: float) -> float:
    """The constant Mahalanobis score value on any pairwise-uniform law:
    sum_m (K_m - 1) / (1 + ridge * K_m)."""
    return sum((k - 1) / (1.0 + ridge * k) for k in head_sizes)
