"""The canonical experiment flow. Every experiment driver is a loop over
run_trial(task, config, rng); no driver re-implements splitting, fitting,
scoring, selection, or evaluation.

Task types
----------
Law             unconditional synthetic law: exact evaluation against cell_probs.
ConditionalLaw  x-dependent synthetic law: exact per-x evaluation via cond_probs.
LabeledData     provided data (X may be None): empirical test-fold evaluation.

Wiring guarantees
-----------------
* run_trial returns ONE TrialResult PER ALPHA in config.alphas. Everything
  expensive (folds, forecaster, family fits, score matrices) is computed once;
  the alpha loop touches only quantiles, masks, and selection. Results are
  therefore fold-paired across alphas, and each alpha's guarantee is the
  ordinary per-alpha split-conformal guarantee (sharing folds across alphas
  correlates the reported numbers, never the validity).
* two_stage: stage-1 thresholds from the sel_threshold fold, final threshold
  from the cal fold (disjoint by construction) -- the valid procedure.
* naive: selection AND final threshold from the same cal fold. Exists ONLY for
  the A1 ablation; expected to undercover.
* The reference member (index 0 = Mahalanobis) is always recalibrated on the
  same cal fold and evaluated alongside the selection.
* Unconditional flows compute each member's lattice scores ONCE; every other
  score is a gather from those rows. Conditional flows cache each member's
  sel_threshold / cal point scores and sel_size / test lattice scores ONCE.

Diagnostic conventions (all logged, none conflated)
---------------------------------------------------
* delta3_raw / delta3_debiased -- the METHOD's certificate: estimated from
  train + sel_threshold labels only (the data available when the gate/contest
  runs; cal and test never feed any deployed decision).
* delta3_view_raw / delta3_view_debiased -- the VIEW's descriptive certificate:
  estimated from ALL folds of the trial sample (LabeledData tasks only). This
  is the dose-response x-axis: "delta3 = possible gain", the best estimate of
  the law's higher-order structure, with no generalization split.
* delta3B_raw / delta3B_debiased -- the RESIDUAL-LEVEL certificate (conditional
  tasks only): Delta3 of the forecaster's correctness pattern
  B_m = 1{Y_m = argmax_head_m pi_hat(x)} on the 2^M pattern lattice, computed
  on the sel_threshold fold (out-of-sample for the forecaster, pre-cal for the
  procedure). Measures the higher-order error structure the forecaster failed
  to remove -- the quantity available to a conditional kernel.
"""

from dataclasses import dataclass, field

import numpy as np

from .conformal import (
    conformal_quantile,
    empirical_coverage,
    exact_size_and_coverage,
    make_splits,
    prediction_set_mask,
)
from .diagnostics import delta3_debiased, delta3_from_counts
from .forecasters import LogisticForecaster, MarginalForecaster, residuals
from .generators import ConditionalLaw, Law
from .lattice import Lattice
from .scores import FamilyConfig, LACScore, build_family
from .selection import select_min_size


@dataclass
class LabeledData:
    """Pre-sampled or real data. X = None means unconditional (label-only)."""

    lattice: Lattice
    X: object  # (n, d) array, or None
    cell_index: np.ndarray  # (n,) flat cell indices


@dataclass
class TrialConfig:
    alphas: tuple = (0.2, 0.1, 0.05, 0.02)
    # synthetic tasks sample these many points per fold
    fold_sizes: dict = field(
        default_factory=lambda: dict(
            train=600, sel_threshold=300, sel_size=300, cal=600, test=600
        )
    )
    # LabeledData splits its n points by these fractions. The CONDITIONAL path
    # uses all five folds (sel_size supplies fresh covariates for per-x size
    # estimates). The UNCONDITIONAL path never reads sel_size -- the set is a
    # single fixed subset of the lattice whose size is computed exactly -- so we
    # use a 4-fold split and give sel_size's budget to the test fold, where it
    # directly shrinks the seed-to-seed noise of the REPORTED gain/coverage.
    fold_fractions: dict = field(
        default_factory=lambda: dict(
            train=0.4, sel_threshold=0.15, sel_size=0.15, cal=0.2, test=0.1
        )
    )
    fold_fractions_unconditional: dict = field(
        default_factory=lambda: dict(train=0.4, sel_threshold=0.15, cal=0.2, test=0.25)
    )
    family: FamilyConfig = field(default_factory=FamilyConfig)
    selection: str = "two_stage"  # or "naive" (ablation A1 only)
    switch_margin: float = 0.0


@dataclass
class TrialResult:
    alpha: float
    selected_index: int
    selected_name: str
    threshold: float
    size_selected: float
    coverage_selected: float
    size_reference: float  # member 0, recalibrated on the same cal fold
    coverage_reference: float
    stage1_names: list
    stage1_sizes: np.ndarray
    stage1_thresholds: np.ndarray
    # EVERY member deployed (cal-recalibrated) -- the always-on comparison that
    # makes the selector's censoring visible: false positives AND forfeited wins.
    all_thresholds: np.ndarray
    all_sizes: np.ndarray
    all_coverages: np.ndarray
    delta3_raw: float
    delta3_debiased: float
    meta: dict
    # ---- optional diagnostics; None where not applicable (see module docstring)
    delta3_view_raw: float = None        # LabeledData: all folds
    delta3_view_debiased: float = None
    delta3B_raw: float = None            # conditional: correctness pattern
    delta3B_debiased: float = None
    forecast_acc_perhead: float = None   # conditional: mean per-head accuracy
    forecast_acc_joint: float = None     # conditional: all-heads-correct rate
    # ---- additive columns (release run). Each is computed strictly after
    # and independently of every field above: none draws randomness, none
    # enters the contest, none can perturb an existing column.
    size_lac: float = None               # per-head LAC baseline, cal-calibrated
    coverage_lac: float = None           #   (logged only; never in the family)
    size_reference_pooled: float = None  # A6 ablation: incumbent recalibrated
    coverage_reference_pooled: float = None  # on cal + sel_threshold pooled
    delta3Z_raw: float = None            # value-level residual certificate
    delta3Z_debiased: float = None       #   (conditional only; sel fold)


def run_trial(task, config: TrialConfig, rng: np.random.Generator) -> list:
    """Run one trial; return a list of TrialResult, one per alpha in
    config.alphas, sharing folds and fitted objects (fold-paired)."""
    if isinstance(task, Law):
        return _run_law(task, config, rng)
    if isinstance(task, ConditionalLaw):
        return _run_conditional_law(task, config, rng)
    if isinstance(task, LabeledData):
        return _run_labeled_data(task, config, rng)
    raise TypeError(f"unknown task type {type(task)}")


# ====================== unconditional core ======================


def _run_unconditional(lattice, fold_labels, evaluate_row, config, meta,
                       extras=None):
    """fold_labels: dict with train / sel_threshold / cal cell-index arrays.
    evaluate_row(lattice_score_row, threshold) -> (size, coverage)."""
    train_labels = fold_labels["train"]
    forecaster = MarginalForecaster().fit(None, lattice.decode(train_labels), lattice)
    probs_row = forecaster.predict_probs()  # (1, D)
    train_residuals = residuals(lattice, probs_row, train_labels)
    family = build_family(
        lattice, train_residuals, train_labels, probs_row, config.family
    )
    # each member's lattice scores computed ONCE; everything else is a gather
    rows = np.stack([member.score_lattice(probs_row)[0] for member in family])

    selection_fold = "cal" if config.selection == "naive" else "sel_threshold"
    stage1_scores = rows[:, fold_labels[selection_fold]]
    cal_scores = rows[:, fold_labels["cal"]]

    # ---- additive instrumentation (after all existing computation; no rng)
    lac_row = LACScore().fit(lattice, train_residuals, train_labels,
                             probs_row).score_lattice(probs_row)[0]
    lac_cal = lac_row[fold_labels["cal"]]
    pooled_labels = np.concatenate([fold_labels["cal"],
                                    fold_labels["sel_threshold"]])
    pooled_reference = rows[0][pooled_labels]

    # method-side certificate: pre-calibration data only, alpha-independent
    diagnostic_labels = np.concatenate([train_labels, fold_labels["sel_threshold"]])
    counts = np.bincount(diagnostic_labels, minlength=lattice.n_cells)
    d3_raw = delta3_from_counts(counts, lattice)
    d3_deb = delta3_debiased(counts, lattice)

    results = []
    for alpha in config.alphas:
        selection = select_min_size(
            stage1_scores,
            size_at=lambda member, q: (rows[member] <= q).sum(),
            alpha=alpha,
            switch_margin=config.switch_margin,
        )
        all_thresholds = np.array(
            [conformal_quantile(cal_scores[k], alpha) for k in range(len(family))]
        )
        deployed = [evaluate_row(rows[k], all_thresholds[k])
                    for k in range(len(family))]
        all_sizes = np.array([size for size, _ in deployed], dtype=float)
        all_coverages = np.array([coverage for _, coverage in deployed], dtype=float)
        lac_size, lac_cov = evaluate_row(
            lac_row, conformal_quantile(lac_cal, alpha))
        pool_size, pool_cov = evaluate_row(
            rows[0], conformal_quantile(pooled_reference, alpha))
        chosen = selection.chosen
        results.append(TrialResult(
            alpha=alpha,
            selected_index=chosen,
            selected_name=family[chosen].name,
            threshold=float(all_thresholds[chosen]),
            size_selected=float(all_sizes[chosen]),
            coverage_selected=float(all_coverages[chosen]),
            size_reference=float(all_sizes[0]),
            coverage_reference=float(all_coverages[0]),
            stage1_names=[m.name for m in family],
            stage1_sizes=selection.stage1_sizes,
            stage1_thresholds=selection.stage1_thresholds,
            all_thresholds=all_thresholds,
            all_sizes=all_sizes,
            all_coverages=all_coverages,
            delta3_raw=d3_raw,
            delta3_debiased=d3_deb,
            meta=meta,
            size_lac=float(lac_size),
            coverage_lac=float(lac_cov),
            size_reference_pooled=float(pool_size),
            coverage_reference_pooled=float(pool_cov),
            **(extras or {}),
        ))
    return results


def _run_law(law: Law, config, rng):
    sizes = config.fold_sizes
    fold_labels = {
        name: law.sample_labels(sizes[name], rng)
        for name in ("train", "sel_threshold", "cal")
    }
    evaluate = lambda row, q: exact_size_and_coverage(row, q, law.cell_probs)
    meta = dict(task="Law", selection=config.selection, fold_sizes=dict(sizes))
    return _run_unconditional(law.lattice, fold_labels, evaluate, config, meta)


# ====================== conditional core ======================


def _run_conditional(lattice, folds, config, meta, test_cond_probs=None,
                     extras=None):
    """folds: dict name -> (X, cell_index) for all five folds.
    Coverage on test: exact against test_cond_probs (n_test, n_cells) when
    given (ConditionalLaw), else empirical at the test labels."""
    X_train, train_labels = folds["train"]
    forecaster = LogisticForecaster().fit(
        X_train, lattice.decode(train_labels), lattice
    )
    probs = {name: forecaster.predict_probs(X) for name, (X, _) in folds.items()}
    train_residuals = residuals(lattice, probs["train"], train_labels)
    family = build_family(
        lattice, train_residuals, train_labels, probs["train"], config.family
    )

    # ---- residual-level certificate: the forecaster's correctness pattern on
    # the sel_threshold fold (out-of-sample for the forecaster, pre-cal for the
    # procedure); alpha-independent.
    sel_probs = probs["sel_threshold"]
    sel_labels = folds["sel_threshold"][1]
    predicted = np.stack(
        [sel_probs[:, offset:offset + size].argmax(axis=1)
         for offset, size in zip(lattice.head_offsets, lattice.head_sizes)],
        axis=1,
    )
    correct = (lattice.decode(sel_labels) == predicted).astype(np.int64)
    pattern_lattice = Lattice((2,) * lattice.n_heads)
    pattern_counts = np.bincount(
        pattern_lattice.encode(correct), minlength=pattern_lattice.n_cells
    )
    pattern_extras = dict(
        delta3B_raw=delta3_from_counts(pattern_counts, pattern_lattice),
        delta3B_debiased=delta3_debiased(pattern_counts, pattern_lattice),
        forecast_acc_perhead=float(correct.mean()),
        forecast_acc_joint=float(correct.all(axis=1).mean()),
    )
    # value-level residual certificate on the same fold (additive; no rng)
    z_values = np.mod(lattice.decode(sel_labels) - predicted,
                      np.array(lattice.head_sizes)[None, :])
    z_counts = np.bincount(lattice.encode(z_values),
                           minlength=lattice.n_cells)
    pattern_extras["delta3Z_raw"] = delta3_from_counts(z_counts, lattice)
    pattern_extras["delta3Z_debiased"] = delta3_debiased(z_counts, lattice)

    # ---- score matrices, each computed ONCE (the alpha loop only thresholds)
    selection_fold = "cal" if config.selection == "naive" else "sel_threshold"
    stage1_scores = np.stack(
        [member.score_points(probs[selection_fold], folds[selection_fold][1])
         for member in family]
    )
    cal_labels = folds["cal"][1]
    cal_scores = np.stack(
        [member.score_points(probs["cal"], cal_labels) for member in family]
    )
    sel_size_lattice = [member.score_lattice(probs["sel_size"]) for member in family]
    test_lattice = [member.score_lattice(probs["test"]) for member in family]
    test_labels = folds["test"][1]
    n_test = len(test_labels)

    # ---- additive instrumentation (after all existing caches; no rng)
    lac = LACScore().fit(lattice, train_residuals, train_labels,
                         probs["train"])
    lac_cal = lac.score_points(probs["cal"], cal_labels)
    lac_test_lattice = lac.score_lattice(probs["test"])
    pooled_reference = np.concatenate([cal_scores[0], stage1_scores[0]]) \
        if config.selection != "naive" else np.concatenate(
            [cal_scores[0],
             family[0].score_points(probs["sel_threshold"],
                                    folds["sel_threshold"][1])])

    def estimated_size(member, q):
        return prediction_set_mask(sel_size_lattice[member], q).sum(axis=1).mean()

    def deployed_size_and_coverage(member, q):
        return _lattice_size_and_coverage(test_lattice[member], q)

    def _lattice_size_and_coverage(lat, q):
        mask = prediction_set_mask(lat, q)
        mean_size = float(mask.sum(axis=1).mean())
        if test_cond_probs is not None:  # exact per-x coverage against the law
            coverage = float((test_cond_probs * mask).sum(axis=1).mean())
        else:
            point_scores = lat[np.arange(n_test), test_labels]
            coverage = empirical_coverage(point_scores, q)
        return mean_size, coverage

    # method-side certificate: pre-calibration data only, alpha-independent
    diagnostic_labels = np.concatenate([train_labels, folds["sel_threshold"][1]])
    counts = np.bincount(diagnostic_labels, minlength=lattice.n_cells)
    d3_raw = delta3_from_counts(counts, lattice)
    d3_deb = delta3_debiased(counts, lattice)

    merged_extras = dict(pattern_extras)
    merged_extras.update(extras or {})

    results = []
    for alpha in config.alphas:
        selection = select_min_size(
            stage1_scores, estimated_size, alpha, config.switch_margin
        )
        all_thresholds = np.array(
            [conformal_quantile(cal_scores[k], alpha) for k in range(len(family))]
        )
        deployed = [deployed_size_and_coverage(k, all_thresholds[k])
                    for k in range(len(family))]
        all_sizes = np.array([size for size, _ in deployed], dtype=float)
        all_coverages = np.array([coverage for _, coverage in deployed], dtype=float)
        lac_size, lac_cov = _lattice_size_and_coverage(
            lac_test_lattice, conformal_quantile(lac_cal, alpha))
        pool_size, pool_cov = _lattice_size_and_coverage(
            test_lattice[0], conformal_quantile(pooled_reference, alpha))
        chosen = selection.chosen
        results.append(TrialResult(
            alpha=alpha,
            selected_index=chosen,
            selected_name=family[chosen].name,
            threshold=float(all_thresholds[chosen]),
            size_selected=float(all_sizes[chosen]),
            coverage_selected=float(all_coverages[chosen]),
            size_reference=float(all_sizes[0]),
            coverage_reference=float(all_coverages[0]),
            stage1_names=[m.name for m in family],
            stage1_sizes=selection.stage1_sizes,
            stage1_thresholds=selection.stage1_thresholds,
            all_thresholds=all_thresholds,
            all_sizes=all_sizes,
            all_coverages=all_coverages,
            delta3_raw=d3_raw,
            delta3_debiased=d3_deb,
            meta=meta,
            size_lac=float(lac_size),
            coverage_lac=float(lac_cov),
            size_reference_pooled=float(pool_size),
            coverage_reference_pooled=float(pool_cov),
            **merged_extras,
        ))
    return results


def _run_conditional_law(law: ConditionalLaw, config, rng):
    sizes = config.fold_sizes
    folds = {
        name: law.sample(sizes[name], rng)
        for name in ("train", "sel_threshold", "sel_size", "cal", "test")
    }
    test_cond_probs = law.cond_probs(folds["test"][0])
    meta = dict(
        task="ConditionalLaw",
        selection=config.selection,
        fold_sizes=dict(sizes),
        tau=law.meta.get("tau"),
    )
    return _run_conditional(law.lattice, folds, config, meta,
                            test_cond_probs=test_cond_probs)


# ====================== provided-data path ======================


def _run_labeled_data(data: LabeledData, config, rng):
    unconditional = data.X is None
    fractions = (
        config.fold_fractions_unconditional if unconditional else config.fold_fractions
    )
    splits = make_splits(len(data.cell_index), fractions, rng)
    lattice = data.lattice
    meta = dict(
        task="LabeledData",
        selection=config.selection,
        fold_sizes={
            name: getattr(splits, name).size
            for name in ("train", "sel_threshold", "sel_size", "cal", "test")
        },
    )
    # view-level certificate: descriptive, ALL folds ("delta3 = possible gain")
    view_counts = np.bincount(np.asarray(data.cell_index),
                              minlength=lattice.n_cells)
    extras = dict(
        delta3_view_raw=delta3_from_counts(view_counts, lattice),
        delta3_view_debiased=delta3_debiased(view_counts, lattice),
    )

    if unconditional:
        fold_labels = {
            name: data.cell_index[getattr(splits, name)]
            for name in ("train", "sel_threshold", "cal")
        }
        test_labels = data.cell_index[splits.test]

        def evaluate(row, threshold):
            size = int((row <= threshold).sum())  # one deployed set
            return size, empirical_coverage(row[test_labels], threshold)

        return _run_unconditional(lattice, fold_labels, evaluate, config, meta,
                                  extras=extras)

    folds = {
        name: (data.X[getattr(splits, name)], data.cell_index[getattr(splits, name)])
        for name in ("train", "sel_threshold", "sel_size", "cal", "test")
    }
    return _run_conditional(lattice, folds, config, meta, extras=extras)
