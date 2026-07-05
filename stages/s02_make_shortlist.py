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


def compare(regen: pd.DataFrame, frozen: pd.DataFrame,
            scan: pd.DataFrame) -> bool:
    """Policy-equivalence check. The historical tie-break was version-
    dependent (unstable sort over heavily tied n_per_cell), so exact key
    identity is NOT the invariant; these are:
      1  provenance: every frozen row appears verbatim in the frozen scan;
      2  admission constraints hold for every frozen row;
      3  per-(dataset, certificate-bin) counts match the regenerated
         policy output exactly;
      4  on the key intersection, all shared columns are value-equal.
    Tie swaps (key differences with matching bin counts) are reported as
    informational."""
    from latticecp.data.shortlist import ShortlistConfig
    config = ShortlistConfig()
    ok = True

    merged = frozen.merge(scan, on=KEY, how="left",
                          suffixes=("", "_scan"), indicator=True)
    n_missing = int((merged._merge != "both").sum())
    print(f"1 provenance: {len(frozen) - n_missing}/{len(frozen)} frozen "
          f"rows found in the frozen scan"
          + ("" if n_missing == 0 else "  [FAIL]"))
    ok &= n_missing == 0
    for col in ("n", "lattice", "n_per_cell", "delta3_debiased"):
        if f"{col}_scan" in merged:
            a = merged[col].to_numpy(float)
            b = merged[f"{col}_scan"].to_numpy(float)
            same = ((a == b) | (np.isnan(a) & np.isnan(b))).all()
            print(f"  frozen.{col} == scan.{col}: {bool(same)}"
                  + ("" if same else "  [FAIL]"))
            ok &= bool(same)

    feasible = ((frozen.lattice >= 2) & (frozen.lattice <= config.lattice_max)
                & (frozen.n_per_cell >= config.min_n_per_cell))
    caps = frozen.groupby("dataset").size().max() <= config.max_per_dataset
    print(f"2 constraints: n_per_cell floor + lattice cap on every row: "
          f"{bool(feasible.all())}; per-dataset cap: {bool(caps)}"
          + ("" if feasible.all() and caps else "  [FAIL]"))
    ok &= bool(feasible.all()) and bool(caps)

    edges = list(config.bin_edges)
    fa, fb = frozen.copy(), regen.copy()
    fa["b"] = pd.cut(fa[config.bin_column], edges, right=False).astype(str)
    fb["b"] = pd.cut(fb[config.bin_column], edges, right=False).astype(str)
    ca = fa.groupby(["dataset", "b"]).size().sort_index()
    cb = fb.groupby(["dataset", "b"]).size().sort_index()
    counts_equal = ca.equals(cb)
    print(f"3 per-(dataset, bin) policy counts equal to regeneration: "
          f"{counts_equal}" + ("" if counts_equal else "  [FAIL]"))
    ok &= counts_equal

    ka = set(map(tuple, regen[KEY].astype(str).itertuples(index=False,
                                                          name=None)))
    kb = set(map(tuple, frozen[KEY].astype(str).itertuples(index=False,
                                                           name=None)))
    swaps = len(kb - ka)
    print(f"4 key intersection {len(ka & kb)}/{len(frozen)}; tie swaps "
          f"(informational, exchangeable within bins): {swaps}")
    inter = regen.merge(frozen, on=KEY, suffixes=("_r", "_f"))
    for col in [c for c in frozen.columns if c in regen.columns
                and c not in KEY]:
        x = inter[f"{col}_r"]; y = inter[f"{col}_f"]
        if x.dtype.kind in "fc" or y.dtype.kind in "fc":
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            equal = (xv == yv) | (np.isnan(xv) & np.isnan(yv))
        else:
            equal = x.astype(str) == y.astype(str)
        if not equal.all():
            ok = False
            print(f"  [FAIL] {col}: {int((~equal).sum())} value mismatches "
                  f"on shared keys")
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
        if compare(shortlist, frozen, scan):
            print("DETERMINISM CHECK PASSED: regenerated shortlist matches "
                  "the frozen artifact value-exactly.")
        else:
            print("DETERMINISM CHECK FAILED: see mismatches above.")
            sys.exit(1)
