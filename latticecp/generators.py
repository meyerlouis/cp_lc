"""Synthetic laws with known ground truth.

Every generator returns a Law (unconditional) or ConditionalLaw whose .meta dict
carries the closed-form truths that exist (delta3_true, support, oracle sizes),
so tests compare the codebase against THEOREMS, never code against code.
"""
from dataclasses import dataclass, field
import math
import numpy as np
from .lattice import Lattice


@dataclass
class Law:
    lattice: Lattice
    cell_probs: np.ndarray
    meta: dict = field(default_factory=dict)

    def sample_labels(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.lattice.n_cells, size=n, p=self.cell_probs)

    def oracle_size(self, alpha: float) -> int:
        """Smallest deterministic set with probability >= 1 - alpha."""
        sorted_probs = np.sort(self.cell_probs)[::-1]
        return int(np.searchsorted(np.cumsum(sorted_probs), 1 - alpha - 1e-12) + 1)


class ConditionalLaw:
    """E6-style x-dependent law: per-head linear logits plus a three-way tilt
    tau * 1{y_3 = (y_1 + y_2) mod K}. X ~ N(0, I_2)."""

    def __init__(self, head_size: int, tau: float, weight_seed: int = 0):
        self.lattice = Lattice((head_size,) * 3)
        self.tau = float(tau)
        weight_rng = np.random.default_rng(weight_seed)
        self.weights = [weight_rng.normal(0.0, 1.0, (2, head_size)) for _ in range(3)]
        cells = self.lattice.cells
        self._tilt = (cells[:, 2] == (cells[:, 0] + cells[:, 1]) % head_size).astype(float)
        self.meta = {"tau": tau, "weight_seed": weight_seed}

    def cond_probs(self, X: np.ndarray) -> np.ndarray:
        """Conditional cell probabilities, shape (n, n_cells)."""
        cells = self.lattice.cells
        logits = sum((X @ self.weights[m])[:, cells[:, m]] for m in range(3))
        logits = logits + self.tau * self._tilt[None, :]
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        return probs / probs.sum(axis=1, keepdims=True)

    def sample(self, n: int, rng: np.random.Generator):
        X = rng.normal(size=(n, 2))
        cumulative = self.cond_probs(X).cumsum(axis=1)
        cell_index = (cumulative < rng.random(n)[:, None]).sum(axis=1)
        return X, cell_index


# ---------------- pairwise-uniform hard family ----------------
def _is_prime(k: int) -> bool:
    return k >= 2 and all(k % d for d in range(2, int(k ** 0.5) + 1))


def affine_world(head_size: int, n_heads: int, shifts) -> Law:
    """Uniform on T_v = {y : y_m = y_1 + (m-2) y_2 + v_m mod K, m >= 3}.
    Pairwise-uniform; delta3 = (M-2) log K exactly. Requires K prime >= M-1."""
    K, M = head_size, n_heads
    if not _is_prime(K) or K < M - 1 or M < 3:
        raise ValueError("affine worlds need K prime >= M-1 and M >= 3")
    shifts = np.asarray(shifts)
    if shifts.shape != (M - 2,):
        raise ValueError(f"need {M - 2} shifts")
    lattice = Lattice((K,) * M)
    cells = lattice.cells
    in_world = np.ones(lattice.n_cells, dtype=bool)
    for m in range(3, M + 1):
        in_world &= cells[:, m - 1] == (cells[:, 0] + (m - 2) * cells[:, 1]
                                        + shifts[m - 3]) % K
    probs = in_world / in_world.sum()
    return Law(lattice, probs, meta={
        "delta3_true": (M - 2) * math.log(K),
        "support": np.flatnonzero(in_world),
        "oracle_size_fn": lambda alpha: math.ceil((1 - alpha) * K * K)})


def latin_mixture(head_size: int, leak: float) -> Law:
    """(1-leak) * Unif(latin square y3 = y1 + y2 mod K)  +  leak * Unif(lattice).
    Pairwise-uniform, so delta3 has the closed form stored in meta."""
    K = head_size
    lattice = Lattice((K, K, K))
    cells = lattice.cells
    on_square = cells[:, 2] == (cells[:, 0] + cells[:, 1]) % K
    probs = leak / lattice.n_cells + np.where(on_square, (1 - leak) / (K * K), 0.0)
    heavy, light = probs.max(), probs.min()
    delta3_true = (K * K * heavy * math.log(heavy * lattice.n_cells)
                   + (lattice.n_cells - K * K) * light
                   * (math.log(light * lattice.n_cells) if light > 0 else 0.0))
    return Law(lattice, probs, meta={"delta3_true": delta3_true,
                                     "support": np.flatnonzero(probs > 0)})


def parity(n_heads: int, flip_prob: float) -> Law:
    """Binary heads; last head = XOR of the others, flipped w.p. flip_prob.
    Pairwise-uniform; delta3 = log 2 - binary_entropy(flip_prob) (nats)."""
    lattice = Lattice((2,) * n_heads)
    cells = lattice.cells
    parity_holds = cells[:, -1] == cells[:, :-1].sum(axis=1) % 2
    half = lattice.n_cells // 2
    probs = np.where(parity_holds, (1 - flip_prob) / half, flip_prob / half)
    entropy = (-flip_prob * math.log(flip_prob) - (1 - flip_prob) * math.log(1 - flip_prob)
               if 0 < flip_prob < 1 else 0.0)
    return Law(lattice, probs, meta={"delta3_true": math.log(2) - entropy,
                                     "heavy_support": np.flatnonzero(parity_holds)})


# ---------------- null (delta3 = 0) laws ----------------
def independent_dirichlet(head_sizes: tuple, concentration: float,
                          rng: np.random.Generator) -> Law:
    """Product of per-head Dirichlet marginals: delta3 = 0 exactly."""
    lattice = Lattice(head_sizes)
    marginals = [rng.dirichlet(np.full(k, concentration)) for k in head_sizes]
    probs = np.ones(lattice.n_cells)
    for m, marginal in enumerate(marginals):
        probs *= marginal[lattice.cells[:, m]]
    return Law(lattice, probs, meta={"delta3_true": 0.0})


def pairwise_ising(n_heads: int, strength: float, rng: np.random.Generator) -> Law:
    """Binary heads, log P = sum_ij J_ij s_i s_j + sum_i h_i s_i: order <= 2 by
    construction, so delta3 = 0 exactly."""
    lattice = Lattice((2,) * n_heads)
    spins = 2.0 * lattice.cells - 1.0
    couplings = rng.normal(0.0, strength, (n_heads, n_heads))
    couplings = np.triu(couplings, 1)
    fields = rng.normal(0.0, strength, n_heads)
    log_probs = np.einsum("li,ij,lj->l", spins, couplings, spins) + spins @ fields
    probs = np.exp(log_probs - log_probs.max())
    return Law(lattice, probs / probs.sum(), meta={"delta3_true": 0.0})


# ---------------- the insufficiency construction (record Thm 3.2) ----------------
def insufficiency_law(temperature: float = 0.25, lam: float = 0.49,
                      seed: int = 0) -> Law:
    """P propto exp(temperature * (f2 + lam * g)) on 3x3x3 with f2 additive taking
    values 0..26 (gaps 1) and g pure 3-way, |g|_inf <= 1, lam < 1/2: the ordering
    equals f2's ordering yet delta3 > 0."""
    lattice = Lattice((3, 3, 3))
    cells = lattice.cells
    f2 = (np.array([0, 1, 2])[cells[:, 0]] + np.array([0, 3, 6])[cells[:, 1]]
          + np.array([0, 9, 18])[cells[:, 2]]).astype(float)
    grid = np.random.default_rng(seed).normal(size=(3, 3, 3))
    for axis in range(3):                       # sequential centering -> pure 3-way
        grid = grid - grid.mean(axis=axis, keepdims=True)
    g = grid.ravel() / np.abs(grid).max()
    log_probs = temperature * (f2 + lam * g)
    probs = np.exp(log_probs - log_probs.max())
    return Law(lattice, probs / probs.sum(),
               meta={"f2_ordering": f2, "pure3_component": g})


# ---------------- phase-transition families ----------------
def uniform_subset(lattice: Lattice, support_size: int,
                   rng: np.random.Generator) -> Law:
    support = rng.choice(lattice.n_cells, support_size, replace=False)
    probs = np.zeros(lattice.n_cells)
    probs[support] = 1.0 / support_size
    return Law(lattice, probs, meta={"support": np.sort(support),
                                     "delta3_true": None})


def soft_subset(lattice: Lattice, support_size: int, heavy_mass: float,
                rng: np.random.Generator) -> Law:
    """heavy_mass spread on a random support, the rest uniform on the complement
    (parity-like heavy/light structure on a random support)."""
    heavy = rng.choice(lattice.n_cells, support_size, replace=False)
    probs = np.full(lattice.n_cells, (1 - heavy_mass) / (lattice.n_cells - support_size))
    probs[heavy] = heavy_mass / support_size
    return Law(lattice, probs, meta={"heavy_support": np.sort(heavy)})
