"""
Post-hoc audit of results_bench.csv. Pure pandas; no reruns. Run on the cluster:
    python scripts/audit_june.py results/results_bench.csv
Paste the whole output back.

Sections:
  A  WORST VIEWS      every seed-averaged (view, alpha) with selected
                      rel_gain < -0.10: who was selected, coverage, sizes.
  B  UNDER-RATE       fraction of rows under (target - 0.03) binned by
                      n_test, next to the BINOMIAL EXPECTATION under exact
                      coverage -- if observed ~ expected, "under-coverage"
                      is test-fold noise, not a validity problem.
  C  MARGIN CURVE     simulate switch_margin from the logged stage-1 sizes
                      and deployed sizes: mean/tail/min gain and negatives
                      as a function of the margin. (Any margin we ADOPT gets
                      re-validated on fresh seeds per the pre-registration.)
  D  DROP-COUNT       same simulation excluding the count member from the
                      conditional contest (knife-edge policy check).
  E  ORACLE           selected vs per-view best member (selection regret).
  F  FAILURES         distinct failed views from the per-task fail logs.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 250)

VIEW = ["dataset", "combo", "max_k"]
path = sys.argv[1] if len(sys.argv) > 1 else "results/results_bench.csv"
df = pd.read_csv(path)
df = df.drop_duplicates(VIEW + ["forecaster", "seed", "alpha"])

# seed-averaged view table
v = df.groupby(VIEW + ["forecaster", "alpha"], as_index=False).agg(
    rel_gain=("rel_gain", "mean"),
    delta3_view=("delta3_view_debiased", "mean"),
    cov_sel=("coverage_selected", "mean"),
    cov_ref=("coverage_reference", "mean"),
    size_sel=("size_selected", "mean"),
    size_ref=("size_reference", "mean"),
    n_test=("n_test", "mean"),
    lattice=("lattice", "first"),
    top_pick=("selected_name", lambda s: s.mode().iat[0]))

print(f"{'=' * 70}\nA WORST VIEWS (seed-averaged rel_gain < -0.10)\n{'=' * 70}")
bad = v[v.rel_gain < -0.10].sort_values("rel_gain")
cols = ["dataset", "combo", "max_k", "forecaster", "alpha", "lattice",
        "delta3_view", "rel_gain", "size_sel", "size_ref",
        "cov_sel", "cov_ref", "top_pick"]
print(bad[cols].round(3).to_string(index=False) if len(bad) else "none")
print(f"\n{len(bad)} view-alphas below -10%; by selected member:")
print(bad.top_pick.value_counts().to_string() if len(bad) else "")
# single-seed catastrophes (the -126% style rows) even if the mean is fine
raw_bad = df[df.rel_gain < -0.50]
print(f"\nsingle-seed rows below -50%: {len(raw_bad)}")
if len(raw_bad):
    print(raw_bad[["dataset", "combo", "forecaster", "seed", "alpha",
                   "selected_name", "rel_gain", "coverage_selected",
                   "size_selected", "size_reference"]]
          .sort_values("rel_gain").head(20).round(3).to_string(index=False))

print(f"\n{'=' * 70}\nB UNDER-RATE vs n_test (observed vs binomial "
      f"expectation under exact coverage)\n{'=' * 70}")
d = df.copy()
d["under"] = d.coverage_selected < (1 - d.alpha - 0.03)
d["expected_under"] = binom.cdf(
    np.ceil((1 - d.alpha - 0.03) * d.n_test) - 1, d.n_test, 1 - d.alpha)
d["n_test_bin"] = pd.cut(d.n_test, [0, 150, 300, 600, 1200, 10 ** 9],
                         labels=["<=150", "150-300", "300-600",
                                 "600-1200", ">1200"])
tab = d.groupby(["forecaster", "alpha", "n_test_bin"], observed=True).agg(
    observed=("under", "mean"), expected=("expected_under", "mean"),
    mean_cov=("coverage_selected", "mean"), n=("under", "size")).round(3)
print(tab)

# ---- helpers for the policy simulations (per forecaster: fixed member list)
def matrices(group):
    names = group.stage1_names.iloc[0].split("|")
    s1 = np.array([r.split("|") for r in group.stage1_sizes], dtype=float)
    dep = np.array([r.split("|") for r in group.all_sizes], dtype=float)
    return names, s1, dep


def simulate(group, margin=0.0, drop=()):
    names, s1, dep = matrices(group)
    keep = np.array([n not in drop for n in names])
    s1m = np.where(keep[None, :], s1, np.inf)
    chosen = s1m.argmin(axis=1)
    stay = (s1m[:, 0] - s1m[np.arange(len(s1m)), chosen]) <= margin
    chosen = np.where(stay, 0, chosen)
    deployed = dep[np.arange(len(dep)), chosen]
    return 1.0 - deployed / np.maximum(dep[:, 0], 1e-12)


print(f"\n{'=' * 70}\nC SWITCH-MARGIN floor curve (simulated from logged "
      f"stage-1 + deployed sizes)\n{'=' * 70}")
for fc in df.forecaster.unique():
    g = df[(df.forecaster == fc) & (df.alpha == 0.1)].reset_index(drop=True)
    tail = g.delta3_view_debiased >= 0.15
    print(f"\n--- {fc} (alpha=0.1) ---")
    print("margin  mean     tail     min      negatives<-1%")
    for m in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
        gain = simulate(g, margin=m)
        print(f"{m:5.1f}  {gain.mean():+.4f}  {gain[tail].mean():+.4f}  "
              f"{gain.min():+.4f}   {(gain < -0.01).sum()}")

print(f"\n{'=' * 70}\nD DROP-COUNT policy (conditional contest without the "
      f"count member)\n{'=' * 70}")
g = df[(df.forecaster == "logit") & (df.alpha == 0.1)].reset_index(drop=True)
if len(g):
    tail = g.delta3_view_debiased >= 0.15
    for drop in ((), ("count",)):
        gain = simulate(g, drop=drop)
        label = "with count" if not drop else "drop count"
        print(f"{label}: mean {gain.mean():+.4f}  tail {gain[tail].mean():+.4f}"
              f"  min {gain.min():+.4f}  negatives<-1% {(gain < -0.01).sum()}")

print(f"\n{'=' * 70}\nE ORACLE (per-view best member vs selected, alpha=0.1)\n"
      f"{'=' * 70}")
for fc in df.forecaster.unique():
    g = df[(df.forecaster == fc) & (df.alpha == 0.1)].reset_index(drop=True)
    _, _, dep = matrices(g)
    oracle_gain = 1.0 - dep.min(axis=1) / np.maximum(dep[:, 0], 1e-12)
    sel_gain = g.rel_gain.to_numpy()
    reg = pd.cut(g.delta3_view_debiased, [-1, 0.02, 0.05, 0.15, 99],
                 labels=["null", "low", "mid", "tail"])
    out = pd.DataFrame({"regime": reg, "selected": sel_gain,
                        "oracle": oracle_gain}).groupby(
        "regime", observed=True).mean().round(4)
    out["fraction_captured"] = (out.selected / out.oracle.replace(0, np.nan)
                                ).round(3)
    print(f"\n--- {fc} ---"); print(out)

print(f"\n{'=' * 70}\nF FAILURES\n{'=' * 70}")
logs = sorted(glob.glob(str(Path(__file__).resolve().parents[1]
                            / "results_bench" / "fail_*.log")))
seen = {}
for f in logs:
    lines = open(f).read().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("("):
            seen.setdefault(line, lines[i + 1: i + 3])
print(f"{len(seen)} distinct failed view-keys across {len(logs)} logs")
for key, ctx in list(seen.items())[:10]:
    print(f"\n{key}"); print("  " + "\n  ".join(ctx))
