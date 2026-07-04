"""Stage 01 (EXPENSIVE; optional regeneration): scan every catalog
dataset for candidate label views and score each with the certificate.

    python stages/s01_scan_views.py [--workers 8] [--limit 5]
                                    [--out results/master_scan_regen.csv]

Writes one row per (dataset, combo, max_k) with the certificate
diagnostics and the search provenance (exhaustive | greedy | random).
The committed artifacts/master_scan.csv is the frozen scan this paper
used; this stage exists so the from-zero chain is runnable and
inspectable. Deterministic given the catalog and the cache: the search
rng is seeded per dataset from its label.
"""
import argparse
import os
import sys
import time
import traceback
import zlib
from pathlib import Path

if "--workers" in " ".join(sys.argv):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latticecp.data.loaders import fetch_dataset
from latticecp.data.views import ViewSearchConfig, scan_dataset_views


def scan_one(job):
    """One dataset -> ('ok', rows) | ('fail', text)."""
    try:
        frame = fetch_dataset(job["reference"], job["cache"])
        columns = {name: frame[name].to_numpy() for name in frame.columns}
        rng = np.random.default_rng(zlib.crc32(job["label"].encode()))
        records = scan_dataset_views(columns, ViewSearchConfig(), rng)
        rows = [dict(
            dataset=job["label"], combo="|".join(r.combo), max_k=r.max_k,
            n=r.n, lattice=r.lattice, n_per_cell=r.n_per_cell,
            delta3_raw=r.delta3_raw, delta3_debiased=r.delta3_debiased,
            g2_pvalue=r.g2_pvalue, source=r.source) for r in records]
        return "ok", rows
    except Exception:
        return "fail", f"{job['label']}\n{traceback.format_exc()}\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--catalog", default=str(ROOT / "artifacts" / "catalog.csv"))
    p.add_argument("--cache", default=str(ROOT / "data_cache"))
    p.add_argument("--out",
                   default=str(ROOT / "results" / "master_scan_regen.csv"))
    args = p.parse_args()

    catalog = pd.read_csv(args.catalog).astype(str)
    if args.limit:
        catalog = catalog.head(args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = set(pd.read_csv(out).dataset.astype(str).unique())
    jobs = [dict(label=r.label, reference=r.reference, cache=args.cache)
            for r in catalog.itertuples() if r.label not in done]
    print(f"{len(catalog)} datasets in catalog, {len(jobs)} to scan "
          f"({len(done)} done), workers={args.workers}")
    failures = open(out.parent / "scan_failures.log", "a")
    started, completed = time.time(), 0

    def handle(outcome):
        nonlocal completed
        status, payload = outcome
        if status == "ok":
            if payload:
                pd.DataFrame(payload).to_csv(
                    out, mode="a", index=False, header=not out.exists())
        else:
            failures.write(payload); failures.flush()
        completed += 1
        if completed % 10 == 0:
            rate = completed / (time.time() - started)
            print(f"  {completed}/{len(jobs)} datasets ({rate:.2f}/s)")

    if args.workers <= 1:
        for job in jobs:
            handle(scan_one(job))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import get_context
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=get_context("spawn")) as pool:
            for fut in as_completed([pool.submit(scan_one, j) for j in jobs]):
                handle(fut.result())
    print(f"done: {completed} datasets in "
          f"{(time.time() - started) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
