"""Behavior tests for the pipeline. Fast set by default; add --full
for the Monte-Carlo validity tests (T1, T3; ~8 extra minutes).
Run:  python tests/test_pipeline.py [--full]

What is tested and why:
  T0  conformal_quantile returns +inf when the finite-sample rank exceeds n
      (the hardening; previously silently returned the max score).
  T1  VALIDITY, unconditional: on a null law, EVERY family member's deployed
      coverage (exact, against the law) is on target at EVERY alpha. This is
      the test that would have caught the lean pipeline's bug A.
  T2  Fold pairing across alphas: one fit, thresholds/sizes monotone in alpha.
  T3  VALIDITY + Delta3(B), conditional: exact per-x coverage on target for
      selected and reference; the correctness-pattern certificate is ~0 when
      the law has no higher-order structure (tau=0) and clearly positive when
      a 3-way tilt survives the per-head forecaster (tau=3).
  T4  LabeledData paths: view-level Delta3 logged on both; Delta3(B) only on
      the conditional path; single-alpha config still returns a 1-element list.

Tolerances are Monte-Carlo, stated inline. Runtime ~2-4 min single-threaded.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latticecp.conformal import conformal_quantile
from latticecp.generators import ConditionalLaw, pairwise_ising
from latticecp.pipeline import LabeledData, TrialConfig, run_trial
from latticecp.scores import FamilyConfig

ALPHAS = (0.2, 0.1, 0.05, 0.02)


def t0_conformal_hardening():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=10)
    # ceil(0.98 * 11) = 11 > 10: guarantee unattainable -> trivial threshold
    assert conformal_quantile(scores, 0.02) == float("inf")
    # sanity: attainable case unchanged (rank = ceil(0.9*11) = 10 -> max here)
    assert np.isfinite(conformal_quantile(scores, 0.10))
    print("T0 conformal hardening: ok")


def t1_unconditional_validity():
    """Null ising law (delta3 = 0 exactly), exact coverage per member per alpha."""
    rng = np.random.default_rng(1)
    law = pairwise_ising(n_heads=6, strength=0.6, rng=rng)
    config = TrialConfig(
        alphas=ALPHAS,
        fold_sizes=dict(train=400, sel_threshold=250, sel_size=0, cal=400, test=0),
    )
    n_trials = 40
    coverage_sum, names = None, None
    selected_cov = {a: [] for a in ALPHAS}
    for _ in range(n_trials):
        results = run_trial(law, config, rng)
        assert len(results) == len(ALPHAS)
        if coverage_sum is None:
            names = results[0].stage1_names
            coverage_sum = {a: np.zeros(len(names)) for a in ALPHAS}
        for res in results:
            coverage_sum[res.alpha] += res.all_coverages
            selected_cov[res.alpha].append(res.coverage_selected)
            assert res.stage1_names[0] == "mahalanobis"
            assert res.delta3B_raw is None          # unconditional: no pattern
            assert res.delta3_view_raw is None      # synthetic: no view sample
    print(f"T1 members: {names}")
    for a in ALPHAS:
        mean_cov = coverage_sum[a] / n_trials
        line = "  ".join(f"{n}:{c:.3f}" for n, c in zip(names, mean_cov))
        print(f"T1 alpha={a}: {line}")
        # conformal is conservative in expectation: mean >= 1 - a (MC slack .015)
        assert (mean_cov >= 1 - a - 0.015).all(), (a, mean_cov)
        sel = np.mean(selected_cov[a])
        assert sel >= 1 - a - 0.015, (a, sel)
    print("T1 unconditional validity (every member, every alpha): ok")


def t2_fold_pairing():
    rng = np.random.default_rng(2)
    law = pairwise_ising(n_heads=6, strength=0.6, rng=rng)
    config = TrialConfig(
        alphas=ALPHAS,
        fold_sizes=dict(train=400, sel_threshold=250, sel_size=0, cal=400, test=0),
    )
    results = run_trial(law, config, rng)
    tight, loose = results[ALPHAS.index(0.02)], results[ALPHAS.index(0.2)]
    assert tight.meta is loose.meta                       # one fit, shared folds
    assert tight.stage1_names == loose.stage1_names
    assert (tight.all_thresholds >= loose.all_thresholds).all()
    assert (tight.all_sizes >= loose.all_sizes).all()     # nested sets in alpha
    print("T2 fold pairing across alphas: ok")


def t3_conditional_validity_and_pattern_certificate():
    config = TrialConfig(
        alphas=(0.1,),
        fold_sizes=dict(train=500, sel_threshold=300, sel_size=250,
                        cal=400, test=400),
        family=FamilyConfig(max_support=256),
    )
    n_trials = 16
    summary = {}
    for tau in (0.0, 3.0):
        rng = np.random.default_rng(3)
        law = ConditionalLaw(head_size=3, tau=tau)
        cov_sel, cov_ref, d3B = [], [], []
        for _ in range(n_trials):
            (res,) = run_trial(law, config, rng)
            cov_sel.append(res.coverage_selected)
            cov_ref.append(res.coverage_reference)
            d3B.append(res.delta3B_debiased)
            assert res.forecast_acc_perhead is not None
        summary[tau] = (np.mean(cov_sel), np.mean(cov_ref), np.mean(d3B))
        print(f"T3 tau={tau}: cov_selected={summary[tau][0]:.3f} "
              f"cov_reference={summary[tau][1]:.3f} "
              f"delta3B_debiased={summary[tau][2]:.4f}")
        # exact per-x coverage, alpha=0.1, MC slack 0.03 over 16 trials
        assert summary[tau][0] >= 0.9 - 0.03, summary[tau]
        assert summary[tau][1] >= 0.9 - 0.03, summary[tau]
    assert summary[0.0][2] < 0.02, "pattern certificate fired on a null law"
    assert summary[3.0][2] > summary[0.0][2] + 0.02, \
        "pattern certificate missed a surviving 3-way tilt"
    print("T3 conditional validity + Delta3(B) sign behavior: ok")


def t4_labeled_data_paths():
    rng = np.random.default_rng(4)
    law = ConditionalLaw(head_size=3, tau=2.0)
    X, cells = law.sample(2500, rng)
    config = TrialConfig(alphas=(0.1, 0.02), family=FamilyConfig(max_support=256))

    conditional = run_trial(LabeledData(law.lattice, X, cells), config, rng)
    assert len(conditional) == 2
    assert conditional[0].delta3_view_debiased is not None
    assert conditional[0].delta3B_debiased is not None
    assert conditional[0].meta["fold_sizes"]["sel_size"] > 0

    unconditional = run_trial(LabeledData(law.lattice, None, cells), config, rng)
    assert unconditional[0].delta3_view_debiased is not None
    assert unconditional[0].delta3B_debiased is None
    # additive columns: present on both paths, Z only conditional
    for res in (conditional[0], unconditional[0]):
        assert res.size_lac is not None and res.coverage_lac is not None
        assert res.size_reference_pooled is not None
    assert conditional[0].delta3Z_debiased is not None
    assert unconditional[0].delta3Z_debiased is None
    assert unconditional[0].meta["fold_sizes"]["sel_size"] == 0  # 4-fold split

    single = run_trial(LabeledData(law.lattice, None, cells),
                       TrialConfig(alphas=(0.1,)), rng)
    assert len(single) == 1
    print("T4 LabeledData paths + diagnostics placement: ok")


if __name__ == "__main__":
    full = "--full" in sys.argv
    started = time.time()
    t0_conformal_hardening()
    t2_fold_pairing()
    t4_labeled_data_paths()
    if full:
        t1_unconditional_validity()
        t3_conditional_validity_and_pattern_certificate()
    else:
        print("(skipped T1/T3 Monte-Carlo validity; run with --full)")
    print(f"\nall tests passed in {time.time() - started:.0f}s")
