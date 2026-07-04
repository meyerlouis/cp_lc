"""Stage 02: build the benchmark shortlist from a scan table
(round-robin across certificate bins, per-dataset cap; the policy lives
in latticecp.data.shortlist).

    python stages/s02_make_shortlist.py --scan artifacts/master_scan.csv \
        [--out results/shortlist_regen.csv] [--check artifacts/shortlist.csv]

With --check, the regenerated shortlist is compared value-exactly
(key-joined, format-insensitive) against the frozen artifact: this is
the cheap link of the determinism policy, run by `make
verify-artifacts`. Exits nonzero on any difference.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latticecp.data.shortlist import ShortlistConfig, make_shortlist

KEY = ["dataset", "combo", "max_k"]


def compare(regen: pd.DataFrame, frozen: pd.DataFrame) -> bool:
    a = regen.sort_values(KEY).reset_index(drop=True)
    b = frozen.sort_values(KEY).reset_index(drop=True)
    ka = set(map(tuple, a[KEY].astype(str).itertuples(index=False, name=None)))
    kb = set(map(tuple, b[KEY].astype(str).itertuples(index=False, name=None)))
    if ka != kb:
        print(f"KEY MISMATCH: {len(ka - kb)} only in regen, "
              f"{len(kb - ka)} only in frozen")
        return False
    ok = True
    for col in [c for c in b.columns if c in a.columns and c not in KEY]:
        x, y = a[col], b[col]
        if x.dtype.kind in "fc" or y.dtype.kind in "fc":
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            equal = (xv == yv) | (np.isnan(xv) & np.isnan(yv))
        else:
            equal = x.astype(str) == y.astype(str)
        if not equal.all():
            ok = False
            print(f"  [FAIL] {col}: {int((~equal).sum())} mismatches")
    missing = set(b.columns) - set(a.columns)
    if missing:
        ok = False
        print(f"  [FAIL] frozen columns missing from regen: {sorted(missing)}")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scan", default=str(ROOT / "artifacts" / "master_scan.csv"))
    p.add_argument("--out", default=str(ROOT / "results" / "shortlist_regen.csv"))
    p.add_argument("--check", default=None,
                   help="frozen shortlist to compare against")
    args = p.parse_args()

    scan = pd.read_csv(args.scan)
    shortlist = make_shortlist(scan, ShortlistConfig())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shortlist.to_csv(args.out, index=False)
    print(f"in:  {len(scan)} views, {scan.dataset.nunique()} datasets")
    print(f"out: {len(shortlist)} views, {shortlist.dataset.nunique()} "
          f"datasets -> {args.out}")

    if args.check:
        frozen = pd.read_csv(args.check)
        if compare(shortlist, frozen):
            print("DETERMINISM CHECK PASSED: regenerated shortlist matches "
                  "the frozen artifact value-exactly.")
        else:
            print("DETERMINISM CHECK FAILED: see mismatches above.")
            sys.exit(1)
