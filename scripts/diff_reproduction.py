"""The acceptance test of this repository.

    python scripts/diff_reproduction.py OLD.csv NEW.csv

OLD = the audited results_june.csv from the original run.
NEW = results/results_bench.csv from this repository's stage 03.

Passes iff: (i) the two files contain exactly the same
(dataset, combo, max_k, forecaster, seed, alpha) keys, and (ii) every
column present in OLD has exactly equal values in NEW (floats compared
for exact equality; NaN matches NaN). Columns present only in NEW are
listed as the additive set and do not affect the verdict. Any
discrepancy prints and exits nonzero: the clean repository would then
NOT be the audited experiment, and nothing downstream should be trusted
until the divergence is explained.
"""
import sys

import numpy as np
import pandas as pd

KEY = ["dataset", "combo", "max_k", "forecaster", "seed", "alpha"]


def load(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates(KEY)
    return df.sort_values(KEY).reset_index(drop=True)


def main(old_path, new_path):
    old, new = load(old_path), load(new_path)
    print(f"OLD {old_path}: {len(old)} rows")
    print(f"NEW {new_path}: {len(new)} rows")

    old_keys = set(map(tuple, old[KEY].itertuples(index=False, name=None)))
    new_keys = set(map(tuple, new[KEY].itertuples(index=False, name=None)))
    only_old, only_new = old_keys - new_keys, new_keys - old_keys
    if only_old or only_new:
        print(f"\nKEY MISMATCH: {len(only_old)} keys only in OLD, "
              f"{len(only_new)} only in NEW")
        for k in list(only_old)[:5]:
            print("  only OLD:", k)
        for k in list(only_new)[:5]:
            print("  only NEW:", k)
        sys.exit(1)

    merged = old.merge(new, on=KEY, suffixes=("_old", "_new"))
    shared = [c for c in old.columns if c not in KEY]
    additive = sorted(set(new.columns) - set(old.columns))
    failures = []
    for col in shared:
        a = merged[f"{col}_old"] if f"{col}_old" in merged else merged[col]
        b = merged[f"{col}_new"] if f"{col}_new" in merged else merged[col]
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            av, bv = a.to_numpy(float), b.to_numpy(float)
            equal = (av == bv) | (np.isnan(av) & np.isnan(bv))
            bad = int((~equal).sum())
            worst = float(np.nanmax(np.abs(av - bv))) if bad else 0.0
        else:
            equal = (a.astype(str).fillna("") == b.astype(str).fillna(""))
            bad, worst = int((~equal).sum()), None
        status = "OK " if bad == 0 else "FAIL"
        extra = "" if worst is None else f"  max|diff|={worst:.3e}"
        print(f"  [{status}] {col:32s} mismatches={bad}{extra}")
        if bad:
            failures.append(col)
            rows = merged.loc[~equal, KEY].head(3)
            print(rows.to_string(index=False))

    print(f"\nadditive columns (NEW only, allowed): {additive}")
    if failures:
        print(f"\nREPRODUCTION FAILED on {len(failures)} column(s): "
              f"{failures}")
        sys.exit(1)
    print(f"\nREPRODUCTION EXACT: {len(merged)} rows, "
          f"{len(shared)} shared columns, every value identical.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
