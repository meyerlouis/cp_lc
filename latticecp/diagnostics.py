"""Label-only diagnostics: the order-2 maximum-entropy projection (IPF), the
higher-order divergence Delta3, and its sampling machinery (G^2, dof, debias).

Conventions: natural log everywhere; Delta3 = KL(P || P_leq2) in nats.
"""
from dataclasses import dataclass
import itertools
import numpy as np
from scipy.stats import chi2
from .lattice import Lattice


@dataclass(frozen=True)
class IPFResult:
    projection: np.ndarray        # cell probabilities of P_leq2, shape (n_cells,)
    n_iterations: int
    marginal_gap: float           # max |pair marginal of Q - target| at exit


def ipf_order2(cell_probs: np.ndarray, lattice: Lattice,
               max_iterations: int = 2000, tolerance: float = 1e-10) -> IPFResult:
    """Iterative proportional fitting onto the all-two-way log-linear family:
    the entropy maximizer among laws sharing every pairwise marginal of P."""
    target = np.asarray(cell_probs, dtype=np.float64).reshape(lattice.head_sizes)
    if abs(target.sum() - 1.0) > 1e-8 or (target < 0).any():
        raise ValueError("cell_probs must be a probability vector")
    heads = range(lattice.n_heads)
    pairs = list(itertools.combinations(heads, 2))
    other_axes = {pair: tuple(m for m in heads if m not in pair) for pair in pairs}
    target_marginals = {pair: target.sum(axis=other_axes[pair], keepdims=True)
                        for pair in pairs}

    current = np.full_like(target, 1.0 / lattice.n_cells)
    gap = np.inf
    for iteration in range(1, max_iterations + 1):
        for pair in pairs:
            marginal = current.sum(axis=other_axes[pair], keepdims=True)
            ratio = np.divide(target_marginals[pair], marginal,
                              out=np.zeros_like(marginal), where=marginal > 1e-300)
            current = current * ratio
        gap = max(abs(current.sum(axis=other_axes[pair], keepdims=True)
                      - target_marginals[pair]).max() for pair in pairs)
        if gap < tolerance:
            break
    return IPFResult(current.ravel(), iteration, float(gap))


def delta3(cell_probs: np.ndarray, lattice: Lattice, **ipf_kwargs) -> float:
    """Delta3 = KL(P || P_leq2) >= 0; zero iff P is order-2 log-linear."""
    P = np.asarray(cell_probs, dtype=np.float64).ravel()
    Q = ipf_order2(P, lattice, **ipf_kwargs).projection
    positive = P > 0
    return float((P[positive] * np.log(P[positive]
                                       / np.maximum(Q[positive], 1e-300))).sum())


def delta3_from_counts(cell_counts: np.ndarray, lattice: Lattice, **kw) -> float:
    """Plug-in estimate from label counts (upward-biased at sparse cells:
    see delta3_debiased)."""
    counts = np.asarray(cell_counts, dtype=np.float64)
    return delta3(counts / counts.sum(), lattice, **kw)


def dof(lattice: Lattice) -> int:
    """Degrees of freedom of saturated minus all-two-way log-linear model."""
    sizes = lattice.head_sizes
    main = sum(k - 1 for k in sizes)
    pairwise = sum((sizes[i] - 1) * (sizes[j] - 1)
                   for i, j in itertools.combinations(range(len(sizes)), 2))
    return lattice.n_cells - 1 - main - pairwise


def g2(cell_counts: np.ndarray, lattice: Lattice, **kw) -> float:
    """Deviance of the all-two-way model: G^2 = 2 n Delta3_hat ~ chi2(dof) under
    the null Delta3 = 0."""
    n = int(np.asarray(cell_counts).sum())
    return 2.0 * n * delta3_from_counts(cell_counts, lattice, **kw)


def g2_pvalue(cell_counts: np.ndarray, lattice: Lattice, **kw) -> float:
    return float(chi2.sf(g2(cell_counts, lattice, **kw), dof(lattice)))


def delta3_debiased(cell_counts: np.ndarray, lattice: Lattice, **kw) -> float:
    """Mean-corrected plug-in: subtract the null expectation dof/(2n), floor at 0.
    Report alongside the raw value, never instead of it."""
    n = int(np.asarray(cell_counts).sum())
    raw = delta3_from_counts(cell_counts, lattice, **kw)
    return max(0.0, raw - dof(lattice) / (2.0 * n))
