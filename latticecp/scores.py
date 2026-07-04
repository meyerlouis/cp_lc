"""The nonconformity score family.

Design:
  * Pure math lives in module functions (mahalanobis_lattice_scores,
    gp_variance_scores, ...): efficiency and theorem tests target these.
  * Score classes are thin: fit() stores state, score_lattice(probs) -> (batch, L),
    and score_points() is a GATHER from score_lattice rows, so calibration scores
    and deployed sets share one code path by construction.
  * Unconditional fast/exact paths trigger on probs having a single row.
    GP-variance scores then use the count-weighted distinct-atom compression:
    EXACT full-training-support, cost <= min(n_train, n_cells)^3, no cap anywhere.
  * Conditional GP-variance uses a seeded support subsample (Nystrom cap),
    optionally bagged; CountScore is exact at any scale and must stay in the
    family whenever kernels are capped (build_family warns otherwise).
"""

import warnings
from dataclasses import dataclass, field

import numpy as np

from .lattice import Lattice

# ====================== pure math ======================


def squared_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, shape (len(A), len(B))."""
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(d2, 0.0)


def rbf_kernel(lengthscale: float):
    def kernel(A, B):
        return np.exp(-squared_distances(A, B) / (2.0 * lengthscale**2))

    return kernel


def additive_kernel(lengthscale: float, precision: np.ndarray):
    """Nested kernel: normalized linear (Mahalanobis inner product) + RBF."""
    dim = precision.shape[0]
    rbf = rbf_kernel(lengthscale)

    def kernel(A, B):
        return (A @ precision @ B.T) / dim + rbf(A, B)

    return kernel


def mahalanobis_lattice_scores(
    lattice: Lattice, probs: np.ndarray, covariance: np.ndarray, ridge: float
) -> np.ndarray:
    """Scores r' P r for every cell, r = e(y) - probs[x], P = (cov + ridge I)^-1
    (pseudo-inverse if ridge == 0). Expanded quadratic form so the per-x cost is
    one matmul: e'Pe - 2 p'Pe + p'Pp.  Returns (batch, n_cells)."""
    probs = np.atleast_2d(probs)
    dim = covariance.shape[0]
    if ridge > 0:
        precision = np.linalg.inv(covariance + ridge * np.eye(dim))
    else:
        precision = np.linalg.pinv(covariance)
    E = lattice.cell_onehots  # (L, D)
    cell_quad = np.einsum("ld,de,le->l", E, precision, E)  # e'Pe, (L,)
    cross = (probs @ precision) @ E.T  # p'Pe, (b, L)
    probs_quad = np.einsum("bd,de,be->b", probs, precision, probs)
    return cell_quad[None, :] - 2.0 * cross + probs_quad[:, None]


def gp_variance_scores(
    queries: np.ndarray,
    support: np.ndarray,
    support_weights: np.ndarray,
    kernel,
    ridge: float,
    solve=None,
) -> np.ndarray:
    """GP-variance score 1 - k' W^(1/2) (W^(1/2) K W^(1/2) + ridge I)^-1 W^(1/2) k.

    With unit weights this is the standard score over `support` points. With
    weights = atom counts it is EXACTLY the score whose support is the full
    training multiset (each atom repeated count times) -- the compression
    identity, verified to 1e-12 in tests. Pass a precomputed `solve` (from
    gp_variance_solver) to amortize the inversion across query batches."""
    if solve is None:
        solve = gp_variance_solver(support, support_weights, kernel, ridge)
    k = kernel(queries, support) * np.sqrt(support_weights)[None, :]
    return 1.0 - (k * solve(k.T).T).sum(axis=1)


def gp_variance_solver(
    support: np.ndarray, support_weights: np.ndarray, kernel, ridge: float
):
    """Factorize (W^(1/2) K W^(1/2) + ridge I) once; return a solve(rhs) closure."""
    sqrt_w = np.sqrt(np.asarray(support_weights, dtype=np.float64))
    gram = kernel(support, support) * sqrt_w[:, None] * sqrt_w[None, :]
    gram[np.diag_indices_from(gram)] += ridge
    inverse = np.linalg.inv(gram)
    return lambda rhs: inverse @ rhs


# ====================== score classes ======================


class Score:
    """Interface: fit(...) then score_lattice(probs (b, D)) -> (b, n_cells).
    score_points is a gather from score_lattice -- single code path."""

    name: str = "score"

    def fit(self, lattice: Lattice, train_residuals, train_cell_index, train_probs):
        raise NotImplementedError

    def score_lattice(self, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def score_points(self, probs: np.ndarray, cell_index: np.ndarray) -> np.ndarray:
        lattice_scores = self.score_lattice(probs)
        cell_index = np.asarray(cell_index)
        if lattice_scores.shape[0] == 1:  # constant forecaster
            return lattice_scores[0, cell_index]
        if lattice_scores.shape[0] != cell_index.size:
            raise ValueError("probs batch and cell_index length mismatch")
        return lattice_scores[np.arange(cell_index.size), cell_index]


class LACScore(Score):
    """Sum over heads of (1 - p_head(y_head)); order <= 1 baseline.
    Lattice scores = n_heads - probs @ E', a single matmul."""

    name = "lac"

    def fit(self, lattice, train_residuals, train_cell_index, train_probs):
        self.lattice = lattice
        return self

    def score_lattice(self, probs):
        probs = np.atleast_2d(probs)
        return self.lattice.n_heads - probs @ self.lattice.cell_onehots.T


class MahalanobisScore(Score):
    """Katsios / covariance score: r' (Cov + ridge I)^-1 r."""

    name = "mahalanobis"

    def __init__(self, ridge: float = 1e-2):
        self.ridge = ridge

    def fit(self, lattice, train_residuals, train_cell_index, train_probs):
        self.lattice = lattice
        self.covariance = np.cov(np.asarray(train_residuals).T)
        return self

    def score_lattice(self, probs):
        return mahalanobis_lattice_scores(
            self.lattice, probs, self.covariance, self.ridge
        )


class CountScore(Score):
    """ridge / (train_count(y) + ridge): the exact ell -> 0 member, O(n + L)."""

    name = "count"

    def __init__(self, ridge: float = 1e-2):
        self.ridge = ridge

    def fit(self, lattice, train_residuals, train_cell_index, train_probs):
        counts = np.bincount(np.asarray(train_cell_index), minlength=lattice.n_cells)
        self._row = (self.ridge / (counts + self.ridge))[None, :]
        return self

    def score_lattice(self, probs):
        batch = np.atleast_2d(probs).shape[0]
        return np.broadcast_to(self._row, (batch, self._row.shape[1]))


class GPVarianceScore(Score):
    """GP-variance score for a given kernel. Two fitting paths:
    * unconditional (probs has one row): EXACT count-weighted compression over
      the distinct seen atoms -- full training support, no cap;
    * conditional: seeded support subsample of size <= max_support (Nystrom).
    Scoring is chunked over the query batch to respect max_elements memory."""

    def __init__(
        self,
        name: str,
        kernel_builder,
        ridge: float = 1e-2,
        max_support: int = 1024,
        seed: int = 0,
        max_elements: int = 2**26,
    ):
        self.name = name
        self.kernel_builder = kernel_builder  # fit_info -> kernel(A, B)
        self.ridge = ridge
        self.max_support = max_support
        self.seed = seed
        self.max_elements = max_elements

    def fit(self, lattice, train_residuals, train_cell_index, train_probs):
        self.lattice = lattice
        train_probs = np.atleast_2d(train_probs)
        self.unconditional = train_probs.shape[0] == 1
        self.kernel = self.kernel_builder(train_residuals)
        if self.unconditional:
            counts = np.bincount(
                np.asarray(train_cell_index), minlength=lattice.n_cells
            )
            seen = counts > 0
            self.support = lattice.cell_onehots[seen] - train_probs  # atom residuals
            self.support_weights = counts[seen].astype(np.float64)
        else:
            n = len(train_residuals)
            keep = np.random.default_rng(self.seed).permutation(n)[: self.max_support]
            self.support = np.asarray(train_residuals)[np.sort(keep)]
            self.support_weights = np.ones(len(self.support))
        self._solve = gp_variance_solver(
            self.support, self.support_weights, self.kernel, self.ridge
        )
        return self

    def score_lattice(self, probs):
        probs = np.atleast_2d(probs)
        E = self.lattice.cell_onehots
        n_cells = self.lattice.n_cells
        rows_per_chunk = max(1, self.max_elements // (n_cells * len(self.support)))
        out = np.empty((probs.shape[0], n_cells))
        for start in range(0, probs.shape[0], rows_per_chunk):
            chunk = probs[start : start + rows_per_chunk]
            queries = (E[None, :, :] - chunk[:, None, :]).reshape(-1, E.shape[1])
            scores = gp_variance_scores(
                queries,
                self.support,
                self.support_weights,
                self.kernel,
                self.ridge,
                solve=self._solve,
            )
            out[start : start + rows_per_chunk] = scores.reshape(len(chunk), n_cells)
        return out


# ====================== family construction ======================


@dataclass
class FamilyConfig:
    ridge: float = 1e-2
    # Absolute (tiny) lengthscales are used ONLY in unconditional families: there
    # they realize exact counting via the compression identity. Conditionally
    # (continuous residuals) a tiny-ell kernel degenerates into a near-duplicate
    # detector with a bimodal score; its conformal threshold sits on a knife edge
    # and can jump from a few cells to the full lattice between folds (observed
    # in R1: 3 full-lattice blowups, all this member, all conditional).
    ell_absolute: tuple = (0.05,)
    # 4x and 8x added after the lean run's CV pinned at its 4x grid top
    # (the conditional optimum lives at or above 4x median on real views).
    ell_multiples_of_median: tuple = (0.5, 1.0, 2.0, 4.0, 8.0)
    include_sqrt_dim_median: bool = True
    include_additive: bool = False  # dead weight in R1: degenerate stage-1
    #                                      size in 44% of trials, selected 0.3%
    include_count: bool = True
    max_support: int = 1024
    seed: int = 0
    median_subsample: int = 512


def median_pairwise_distance(
    residuals: np.ndarray, subsample: int, rng: np.random.Generator
) -> float:
    keep = rng.permutation(len(residuals))[:subsample]
    d2 = squared_distances(residuals[keep], residuals[keep])
    positive = d2[d2 > 1e-12]
    return float(np.sqrt(np.median(positive))) if positive.size else 1.0


def build_family(
    lattice: Lattice,
    train_residuals,
    train_cell_index,
    train_probs,
    config: FamilyConfig = FamilyConfig(),
) -> list:
    """Fixed-order family: Mahalanobis FIRST (selection tie-break = earliest),
    then the RBF lengthscale grid, additive, count. Fits every member."""
    train_residuals = np.asarray(train_residuals)
    rng = np.random.default_rng(config.seed)
    median = median_pairwise_distance(train_residuals, config.median_subsample, rng)
    conditional = np.atleast_2d(train_probs).shape[0] > 1

    lengthscales = []
    if not conditional:  # see FamilyConfig.ell_absolute note
        lengthscales += [("rbf_abs%g" % v, v) for v in config.ell_absolute]
    lengthscales += [
        ("rbf_%gxmed" % m, m * median) for m in config.ell_multiples_of_median if m > 0
    ]
    if config.include_sqrt_dim_median:
        lengthscales += [("rbf_sqrtDxmed", np.sqrt(lattice.onehot_dim) * median)]

    family = [MahalanobisScore(config.ridge)]
    for name, ell in lengthscales:
        family.append(
            GPVarianceScore(
                name,
                lambda R, e=ell: rbf_kernel(e),
                config.ridge,
                config.max_support,
                config.seed,
            )
        )
    if config.include_additive:

        def additive_builder(R, e=median, ridge=config.ridge):
            precision = np.linalg.inv(
                np.cov(np.asarray(R).T) + ridge * np.eye(R.shape[1])
            )
            return additive_kernel(e, precision)

        family.append(
            GPVarianceScore(
                "additive_med",
                additive_builder,
                config.ridge,
                config.max_support,
                config.seed,
            )
        )
    if config.include_count:
        family.append(CountScore(config.ridge))

    capped = conditional and len(train_residuals) > config.max_support
    if capped and not config.include_count:
        warnings.warn(
            "Capped kernel supports WITHOUT a CountScore member: the "
            "counting end of the family is silently disabled (the E5 "
            "failure mode). Add CountScore or raise max_support."
        )
    for score in family:
        score.fit(lattice, train_residuals, train_cell_index, train_probs)
    return family
