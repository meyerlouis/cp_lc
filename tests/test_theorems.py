"""Theorem tests: the code is compared against THEOREMS, never against
itself. Fast (~1 min), no network. Run:  python tests/test_theorems.py

  A  Affine worlds: IPF certificate equals (M-2) log K exactly.
  B  Mahalanobis constancy: population score spread ~ 0 on
     pairwise-uniform laws; value equals sum (K-1)/(1+ridge*K).
  C  Spectral theorem: closed-form eigenvalues match numpy's eigvalsh
     of the actual Gram, multiplicities included.
  D  Counting limit: the tiny-lengthscale GP-variance score equals
     gamma/(count+gamma) numerically.
  E  Quadratic ceiling: ANOVA energy of the Mahalanobis lattice score
     at orders >= 3 is zero to machine precision.
  F  Tie-robust rank lemma: coverage >= m/(n+1) under heavy ties.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latticecp.conformal import conformal_quantile
from latticecp.diagnostics import delta3
from latticecp.forecasters import MarginalForecaster, residuals
from latticecp.generators import affine_world, parity
from latticecp.lattice import Lattice
from latticecp.scores import (GPVarianceScore, mahalanobis_lattice_scores,
                              rbf_kernel)
from latticecp.spectral import anova_energy_by_order, rbf_kernel_eigenvalues
from latticecp.theory import blindness_constant

RIDGE = 1e-2


def exact_onehot_covariance(law):
    """Population covariance of e(Y) under the law, exactly."""
    E = law.lattice.cell_onehots
    p = law.cell_probs
    mean = p @ E
    return (E * p[:, None]).T @ E - np.outer(mean, mean)


def t_a_worlds():
    for K, M in ((3, 3), (3, 4), (5, 4)):
        law = affine_world(K, M, shifts=np.arange(M - 2))
        d3 = delta3(law.cell_probs, law.lattice, max_iterations=4000,
                    tolerance=1e-12)
        expect = (M - 2) * np.log(K)
        assert abs(d3 - expect) < 1e-6, (K, M, d3, expect)
    print("A worlds certificate = (M-2) log K: ok")


def t_b_constancy():
    for law, sizes in ((affine_world(3, 4, shifts=[0, 1]), (3,) * 4),
                       (parity(5, flip_prob=0.0), (2,) * 5)):
        cov = exact_onehot_covariance(law)
        probs_row = (law.cell_probs @ law.lattice.cell_onehots)[None, :]
        scores = mahalanobis_lattice_scores(law.lattice, probs_row, cov,
                                            RIDGE)[0]
        spread = scores.max() - scores.min()
        assert spread < 1e-9, spread
        expect = blindness_constant(sizes, RIDGE)
        assert abs(scores[0] - expect) < 1e-9, (scores[0], expect)
    print("B Mahalanobis constancy + closed-form constant: ok")


def t_c_spectral():
    lattice = Lattice((3, 3, 3))
    for ell in (0.8, 1.5):
        formula = rbf_kernel_eigenvalues(ell, lattice)
        gram = rbf_kernel(ell)(lattice.cell_onehots, lattice.cell_onehots)
        numeric = np.sort(np.linalg.eigvalsh(gram))
        assert np.allclose(formula, numeric, atol=1e-8), ell
    print("C spectral eigenvalues (formula vs eigvalsh): ok")


def t_d_counting_limit():
    rng = np.random.default_rng(0)
    lattice = Lattice((3, 3, 3))
    labels = rng.integers(0, lattice.n_cells, 400)
    forecaster = MarginalForecaster().fit(None, lattice.decode(labels),
                                          lattice)
    probs_row = forecaster.predict_probs()
    res = residuals(lattice, probs_row, labels)
    score = GPVarianceScore("rbf_tiny", lambda R: rbf_kernel(1e-3),
                            ridge=RIDGE).fit(lattice, res, labels, probs_row)
    row = score.score_lattice(probs_row)[0]
    counts = np.bincount(labels, minlength=lattice.n_cells)
    expect = RIDGE / (counts + RIDGE)
    assert np.allclose(row, expect, atol=1e-9)
    print("D counting limit (ell -> 0): ok")


def t_e_ceiling():
    rng = np.random.default_rng(1)
    lattice = Lattice((3, 3, 3))
    labels = rng.integers(0, lattice.n_cells, 500)
    forecaster = MarginalForecaster().fit(None, lattice.decode(labels),
                                          lattice)
    probs_row = forecaster.predict_probs()
    res = residuals(lattice, probs_row, labels)
    cov = np.cov(res.T)
    row = mahalanobis_lattice_scores(lattice, probs_row, cov, RIDGE)[0]
    energy = anova_energy_by_order(row, lattice)
    high = energy[3:].sum()
    assert high < 1e-12 * energy.sum(), energy
    print("E quadratic ceiling (order >= 3 energy = 0): ok")


def t_f_rank_lemma():
    rng = np.random.default_rng(2)
    n, alpha, trials = 19, 0.10, 4000
    m = int(np.ceil((1 - alpha) * (n + 1)))
    hits = 0
    for _ in range(trials):
        z = rng.integers(0, 4, n + 1).astype(float)  # heavy ties
        q = conformal_quantile(z[:n], alpha)
        hits += z[n] <= q
    coverage = hits / trials
    assert coverage >= m / (n + 1) - 0.02, coverage
    print(f"F tie-robust coverage {coverage:.3f} >= {m/(n+1):.3f}: ok")


if __name__ == "__main__":
    t_a_worlds(); t_b_constancy(); t_c_spectral()
    t_d_counting_limit(); t_e_ceiling(); t_f_rank_lemma()
    print("\nall theorem tests passed")
