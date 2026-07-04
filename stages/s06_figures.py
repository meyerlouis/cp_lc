"""
Figure 1 (draft) from results_june.csv: selected-procedure gain vs the
view-level certificate, per-dataset balanced (top-S views per dataset by
delta3_view -- each dataset contributes equally; dose-response design, NOT a
prevalence estimate).

Run on the cluster:
    python scripts/plot_june.py results/results_bench.csv
Writes figures/fig1.png (two-panel, alpha=0.1) and
figures/appendix_<forecaster>_grid.png (all four alphas), and prints the
numbers behind every panel.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = 5
ALPHAS = (0.2, 0.1, 0.05, 0.02)
FIG1_ALPHA = 0.1
VIEW = ["dataset", "combo", "max_k"]

path = sys.argv[1] if len(sys.argv) > 1 else "results/results_bench.csv"
root = Path(__file__).resolve().parents[1]
df = pd.read_csv(path)
df = df.drop_duplicates(VIEW + ["forecaster", "seed", "alpha"])

# seed-average -> one row per (view, forecaster, alpha)
v = df.groupby(VIEW + ["forecaster", "alpha"], as_index=False).mean(
    numeric_only=True)


def top_s(sub):
    return (sub.sort_values("delta3_view_debiased", ascending=False)
              .groupby("dataset", group_keys=False).head(S))


def panel(ax, sub, title):
    if not len(sub):
        ax.set_title(f"{title}   (no rows)", fontsize=11)
        return dict(n=0, datasets=0, null=np.nan, n_null=0,
                    tail=np.nan, n_tail=0, worst=np.nan, negatives=0)
    x = sub.delta3_view_debiased.values
    y = sub.rel_gain.values * 100
    ax.axhline(0, color="0.6", lw=0.8, zorder=1)
    ax.scatter(x, y, s=18, alpha=0.45, color="#2c6fbb", edgecolor="none",
               zorder=2)
    null, tail = y[x < 0.02], y[x >= 0.15]
    ax.set_title(f"{title}   (null {null.mean():+.1f}%, "
                 f"tail {tail.mean():+.1f}%)", fontsize=11)
    ax.set_xlim(-0.02, max(0.5, np.nanpercentile(x, 99)))
    ax.grid(axis="y", alpha=0.15)
    return dict(n=len(sub), datasets=sub.dataset.nunique(),
                null=null.mean(), n_null=len(null),
                tail=tail.mean(), n_tail=len(tail),
                worst=y.min(), negatives=int((y < -1e-9).sum()))


plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 11})
outdir = root / "figures"
outdir.mkdir(exist_ok=True)

# ---- Figure 1: two panels at alpha = 0.1
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
stats = {}
for ax, fc, label in zip(axes, ("marg", "logit"),
                         ("marginal forecaster", "conditional forecaster")):
    sub = top_s(v[(v.forecaster == fc) & (v.alpha == FIG1_ALPHA)])
    stats[fc] = panel(ax, sub, label)
    ax.set_xlabel("certified higher-order structure  $\\hat\\Delta_3$ (nats)")
axes[0].set_ylabel("set-size reduction vs Mahalanobis (%)")
fig.suptitle(f"Selected procedure, alpha = {FIG1_ALPHA}: gain vs certificate "
             f"(top-{S} views/dataset, "
             f"{v.dataset.nunique()} datasets, {df.seed.nunique()} seeds)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(outdir / "fig1.png", dpi=160, bbox_inches="tight")
print(f"saved {outdir / 'fig1.png'}")
for fc, s in stats.items():
    print(f"  {fc}: n={s['n']} ({s['datasets']} datasets) | "
          f"null {s['null']:+.1f}% (n={s['n_null']}) | "
          f"tail {s['tail']:+.1f}% (n={s['n_tail']}) | "
          f"worst {s['worst']:+.1f}% | negatives {s['negatives']}")

# ---- appendix grids: all four alphas per forecaster
for fc in ("marg", "logit"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    print(f"\nappendix grid: {fc}")
    for ax, a in zip(axes.ravel(), ALPHAS):
        sub = top_s(v[(v.forecaster == fc) & (v.alpha == a)])
        s = panel(ax, sub, f"alpha = {a}")
        print(f"  alpha={a}: null {s['null']:+.1f}% | tail {s['tail']:+.1f}% "
              f"| worst {s['worst']:+.1f}% | negatives {s['negatives']}")
    for ax in axes[-1]:
        ax.set_xlabel("$\\hat\\Delta_3$ (nats)")
    for ax in axes[:, 0]:
        ax.set_ylabel("set-size reduction vs Mahalanobis (%)")
    fig.suptitle(f"{fc}: selected procedure, all alphas "
                 f"(top-{S} views/dataset)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / f"appendix_{fc}_grid.png", dpi=160,
                bbox_inches="tight")
    print(f"saved {outdir / f'appendix_{fc}_grid.png'}")
