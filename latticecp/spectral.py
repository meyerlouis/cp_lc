"""Interaction-order machinery: ANOVA energy by order and the closed-form
spectrum of the lattice RBF kernel (session record, spectral theorem).
Test/figure-grade code; not on the hot path of any experiment.
"""
import itertools
import numpy as np
from .lattice import Lattice


def anova_energy_by_order(values: np.ndarray, lattice: Lattice) -> np.ndarray:
    """Euclidean energy ||f_S||^2 of the ANOVA components of a lattice function,
    summed by interaction order |S|. Returns shape (n_heads + 1,);
    entries sum to ||values||^2. Cost O(3^n_heads * n_cells): fine for tests."""
    grid = np.asarray(values, dtype=np.float64).reshape(lattice.head_sizes)
    heads = range(lattice.n_heads)

    def mean_over_complement(subset):
        """Average over all heads NOT in `subset`; broadcast back to full grid."""
        axes = tuple(m for m in heads if m not in subset)
        return grid.mean(axis=axes, keepdims=True) if axes else grid

    energy = np.zeros(lattice.n_heads + 1)
    for subset in itertools.chain.from_iterable(
            itertools.combinations(heads, r) for r in range(lattice.n_heads + 1)):
        component = np.zeros_like(grid)
        for r in range(len(subset) + 1):                      # Moebius inversion
            for sub in itertools.combinations(subset, r):
                component = component + (-1) ** (len(subset) - r) * mean_over_complement(sub)
        energy[len(subset)] += (component ** 2).sum()
    return energy


def rbf_kernel_eigenvalues(lengthscale: float, lattice: Lattice) -> np.ndarray:
    """All n_cells eigenvalues (with multiplicity) of the RBF Gram over the full
    one-hot lattice: lambda_S = (1-q)^|S| * prod_{m not in S} (1 + (K_m - 1) q),
    q = exp(-1/lengthscale^2); eigenspace dim = prod_{m in S} (K_m - 1)."""
    q = np.exp(-1.0 / lengthscale ** 2)
    sizes = lattice.head_sizes
    eigenvalues = []
    for subset in itertools.chain.from_iterable(
            itertools.combinations(range(lattice.n_heads), r)
            for r in range(lattice.n_heads + 1)):
        value = (1 - q) ** len(subset)
        multiplicity = 1
        for m in range(lattice.n_heads):
            if m in subset:
                multiplicity *= sizes[m] - 1
            else:
                value *= 1 + (sizes[m] - 1) * q
        eigenvalues += [value] * multiplicity
    return np.sort(np.array(eigenvalues))


def order_decay_ratio(lengthscale: float, head_size: int) -> float:
    """rho = (1-q)/(1+(K-1)q): per-order eigenvalue decay (equal alphabets)."""
    q = np.exp(-1.0 / lengthscale ** 2)
    return (1 - q) / (1 + (head_size - 1) * q)
