"""Stage 05: full analysis of the merged benchmark. Print-only; paste
the whole output back.

    python stages/s05_analyze.py results/results_bench.csv

Sections 1-8 mirror the audited June analysis (validity watchdog, floor
audit, L-shapes, selection map, always-on, native-only). Sections 9-11
read the additive columns:
   9 LAC            per-head baseline from the PRIMARY run (dominated?)
  10 A6 ABLATION    incumbent recalibrated on cal+sel: is the gain a
                    data-split artifact? (predicted: no)
  11 Z FULL-RANGE   value-level residual certificate on all views:
                    partials vs the label certificate (confirms or
                    retires the exploratory tail finding)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.width", 200)

VIEW = ["dataset", "combo", "max_k"]
REGIME_EDGES = [-1, 0.02, 0.05, 0.15, 99]
REGIME_LABELS = ["null<0.02", "0.02-0.05", "0.05-0.15", "tail>=0.15"]

path = sys.argv[1] if len(sys.argv) > 1 else "results/results_bench.csv"
df = pd.read_csv(path)
df = df.drop_duplicates(VIEW + ["forecaster", "seed", "alpha"])

print(f"{'=' * 70}\n1 RAW SHAPE\n{'=' * 70}")
print(f"rows: {len(df)}   datasets: {df.dataset.nunique()}   "
      f"views: {df[VIEW].drop_duplicates().shape[0]}")
print(df.forecaster.value_counts())
print("seeds:", sorted(df.seed.unique()), " alphas:", sorted(df.alpha.unique()))

v = df.groupby(VIEW + ["forecaster", "alpha"], as_index=False).agg(
    {c: "mean" for c in df.columns if df[c].dtype.kind in "fiu"
     and c not in ("seed",)} |
    {"selected_name": lambda s: s.mode().iat[0], "is_native": "first"})
v["regime"] = pd.cut(v.delta3_view_debiased, REGIME_EDGES,
                     labels=REGIME_LABELS)

print(f"\n{'=' * 70}\n2 VALIDITY (target = 1 - alpha)\n{'=' * 70}")
print(df.groupby(["forecaster", "alpha"])[
    ["coverage_selected", "coverage_reference", "coverage_lac",
     "coverage_reference_pooled"]].mean().round(4))
long_cols = VIEW + ["forecaster", "alpha", "seed", "size_reference",
                    "stage1_names", "all_sizes", "all_coverages"]
long = df[long_cols].copy()
long["member"] = long.stage1_names.str.split("|")
long["m_size"] = long.all_sizes.str.split("|")
long["m_cov"] = long.all_coverages.str.split("|")
long = long.explode(["member", "m_size", "m_cov"])
long["m_size"] = long.m_size.astype(float)
long["m_cov"] = long.m_cov.astype(float)
long["m_rel_gain"] = 1.0 - long.m_size / long.size_reference.clip(lower=1e-12)
print("\nper-member watchdog (mean coverage):")
print(long.groupby(["forecaster", "alpha", "member"]).m_cov.mean()
      .unstack().round(4))

print(f"\n{'=' * 70}\n3 FLOOR AUDIT (selected vs Mahalanobis)\n{'=' * 70}")
print(v.groupby(["forecaster", "alpha"]).rel_gain.agg(
    mean="mean", median="median", min="min", max="max",
    negatives=lambda s: int((s < -1e-12).sum()),
    ties=lambda s: int((s.abs() < 1e-12).sum()), n="size").round(4))

print(f"\n{'=' * 70}\n4 L-SHAPE: gain by delta3_view regime\n{'=' * 70}")
for fc in v.forecaster.unique():
    sub = v[v.forecaster == fc]
    print(f"\n--- {fc} ---")
    print(sub.groupby(["regime", "alpha"], observed=True)
          .rel_gain.mean().unstack().round(3))
    print(sub[sub.alpha == 0.1].regime.value_counts().sort_index())

print(f"\n{'=' * 70}\n5 SELECTION by regime (alpha = 0.1)\n{'=' * 70}")
for fc in v.forecaster.unique():
    sub = v[(v.forecaster == fc) & (v.alpha == 0.1)]
    print(f"\n--- {fc} ---")
    print(pd.crosstab(sub.regime, sub.selected_name))

print(f"\n{'=' * 70}\n6 ALWAYS-ON (every member deployed, alpha = 0.1)\n"
      f"{'=' * 70}")
print(long[long.alpha == 0.1].groupby(["forecaster", "member"]).agg(
    mean_gain=("m_rel_gain", "mean"), min_gain=("m_rel_gain", "min"),
    mean_cov=("m_cov", "mean")).round(3))

print(f"\n{'=' * 70}\n7 NATIVE-ONLY tail robustness\n{'=' * 70}")
tail = v[v.regime == "tail>=0.15"]
for fc in v.forecaster.unique():
    sub = tail[tail.forecaster == fc]
    nat = sub[sub.is_native == True]  # noqa: E712
    print(f"{fc}: tail all={sub.rel_gain.mean():+.3f} (n={len(sub)}) | "
          f"native={nat.rel_gain.mean():+.3f} (n={len(nat)})")

print(f"\n{'=' * 70}\n8 UNDER-RATE quick check (selected coverage)\n{'=' * 70}")
df["under"] = df.coverage_selected < (1 - df.alpha - 0.03)
print(df.groupby(["forecaster", "alpha"]).under.mean().round(3))

print(f"\n{'=' * 70}\n9 LAC baseline (primary run; per-head, order-1)\n"
      f"{'=' * 70}")
v9 = v.copy()
v9["lac_vs_mahal"] = v9.size_lac / v9.size_reference.clip(lower=1e-12)
print(v9.groupby(["forecaster", "alpha"]).lac_vs_mahal.agg(
    ["mean", "median"]).round(3))
print("fraction of views where LAC is strictly smaller than Mahalanobis:")
print(v9.groupby(["forecaster", "alpha"])
      .apply(lambda g: float((g.size_lac < g.size_reference - 1e-9).mean()),
             include_groups=False).round(3))

print(f"\n{'=' * 70}\n10 A6 ABLATION: incumbent recalibrated on cal + sel\n"
      f"{'=' * 70}")
v10 = v.copy()
v10["ref_pooled_vs_ref"] = (v10.size_reference_pooled
                            / v10.size_reference.clip(lower=1e-12))
v10["gain_vs_pooled"] = 1.0 - v10.size_selected / \
    v10.size_reference_pooled.clip(lower=1e-12)
print("pooled/plain reference size ratio (1.0 = extra calibration data "
      "buys the incumbent nothing):")
print(v10.groupby(["forecaster", "alpha"]).ref_pooled_vs_ref.agg(
    ["mean", "median"]).round(4))
print("\ntail gain vs plain and vs pooled reference (alpha = 0.1):")
t10 = v10[(v10.regime == "tail>=0.15") & (v10.alpha == 0.1)]
print(t10.groupby("forecaster")[["rel_gain", "gain_vs_pooled"]]
      .mean().round(4))

print(f"\n{'=' * 70}\n11 Z CERTIFICATE, FULL RANGE (logit)\n{'=' * 70}")
lg = v[(v.forecaster == "logit") & (v.alpha == 0.1)].dropna(
    subset=["delta3Z_debiased"]).copy()
if len(lg):
    r = lg[["rel_gain", "delta3_view_debiased", "delta3Z_debiased"]].rank()

    def pcorr(x, y, ctrl):
        rx = x - np.polyval(np.polyfit(ctrl, x, 1), ctrl)
        ry = y - np.polyval(np.polyfit(ctrl, y, 1), ctrl)
        return float(np.corrcoef(rx, ry)[0, 1])

    print(f"n = {len(lg)} views")
    print(f"spearman(gain, label d3) = "
          f"{stats.spearmanr(lg.rel_gain, lg.delta3_view_debiased).statistic:+.3f}")
    print(f"spearman(gain, Z)        = "
          f"{stats.spearmanr(lg.rel_gain, lg.delta3Z_debiased).statistic:+.3f}")
    print(f"spearman(Z, label d3)    = "
          f"{stats.spearmanr(lg.delta3Z_debiased, lg.delta3_view_debiased).statistic:+.3f}")
    print(f"partial (gain, Z | label d3)  = "
          f"{pcorr(r.rel_gain, r.delta3Z_debiased, r.delta3_view_debiased):+.3f}")
    print(f"partial (gain, label d3 | Z)  = "
          f"{pcorr(r.rel_gain, r.delta3_view_debiased, r.delta3Z_debiased):+.3f}")
    print("\nmean gain by (label regime x Z split at median-positive):")
    zpos = lg.delta3Z_debiased > 0
    zmed = lg.loc[zpos, "delta3Z_debiased"].median()
    lg["Zgrp"] = np.where(~zpos, "Z=0",
                          np.where(lg.delta3Z_debiased <= zmed,
                                   "Z low", "Z high"))
    print(lg.pivot_table(index="regime", columns="Zgrp", values="rel_gain",
                         observed=True).round(3))
else:
    print("no logit rows with delta3Z logged")
