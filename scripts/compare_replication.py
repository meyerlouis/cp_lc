"""Version-robustness replication: compare the canonical benchmark
against an independent run under a different dependency stack (the June
run, sklearn 1.8) and report aggregate stability plus the size and
location of row-level differences.

    python scripts/compare_replication.py OLD.csv NEW.csv

This is NOT the acceptance test (scripts/diff_reproduction.py, which
demands exact equality of a rerun under the frozen lock file). It is
the appendix exhibit: two solver versions, same code, same seeds, same
folds; aggregates must agree to within seed noise, with differences
confined to knife-edge views.
"""
import sys

import numpy as np
import pandas as pd

KEY = ["dataset", "combo", "max_k", "forecaster", "seed", "alpha"]
VIEW = ["dataset", "combo", "max_k"]


def main(old_path, new_path):
    old = pd.read_csv(old_path).drop_duplicates(KEY)
    new = pd.read_csv(new_path).drop_duplicates(KEY)
    m = old.merge(new, on=KEY, suffixes=("_old", "_new"))
    print(f"joined rows: {len(m)} "
          f"(old {len(old)}, new {len(new)})")

    print(f"\n{'=' * 66}\nROW-LEVEL DIFFERENCES (exact standard)\n{'=' * 66}")
    for col in ("rel_gain", "size_selected", "coverage_selected"):
        d = (m[f"{col}_old"] - m[f"{col}_new"]).abs()
        by_fc = m.assign(d=d).groupby("forecaster").d.agg(
            frac_diff=lambda s: float((s > 1e-12).mean()),
            max_diff="max")
        print(f"\n{col}:"); print(by_fc.round(4))
    flips = (m.selected_name_old != m.selected_name_new)
    print(f"\nselection flips: {int(flips.sum())} of {len(m)} "
          f"({flips.mean():.2%}); by forecaster:")
    print(m[flips].forecaster.value_counts().to_string())

    print(f"\n{'=' * 66}\nAGGREGATE STABILITY (the exhibit)\n{'=' * 66}")
    for label, df in (("old", old), ("new", new)):
        df["run"] = label
    both = pd.concat([old, new], ignore_index=True)
    v = both.groupby(VIEW + ["forecaster", "alpha", "run"],
                     as_index=False).agg(
        rel_gain=("rel_gain", "mean"),
        coverage=("coverage_selected", "mean"),
        d3=("delta3_view_debiased", "mean"))
    v["regime"] = pd.cut(v.d3, [-1, 0.02, 0.05, 0.15, 99],
                         labels=["null", "low", "mid", "tail"])
    print("\nmean selected coverage by run (alpha rows):")
    print(v.groupby(["forecaster", "alpha", "run"])["coverage"].mean()
          .unstack().round(4))
    print("\nmean gain by regime and run (alpha = 0.1):")
    g = v[v.alpha == 0.1].groupby(["forecaster", "regime", "run"],
                                  observed=True).rel_gain.mean().unstack()
    g["abs_delta"] = (g["old"] - g["new"]).abs()
    print(g.round(4))
    worst = float(g["abs_delta"].max())
    print(f"\nlargest regime-mean shift between solver versions: "
          f"{worst:.4f} "
          f"({'within seed noise' if worst < 0.01 else 'INSPECT'})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
