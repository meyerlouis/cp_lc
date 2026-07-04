"""Stage 03: the benchmark. One core function (run_view) driven in two
modes:

  SLURM-array task (one forecaster, one seed, one chunk of the shortlist):
      python stages/s03_run_benchmark.py --task SEED FORECASTER CHUNK NCHUNKS
  Local (serial or process-pool over all (view, forecaster, seed) jobs):
      python stages/s03_run_benchmark.py --local [--workers 8] [--quick]

REPRODUCTION CONTRACT (the acceptance test of this repository): in task
mode with the frozen artifacts, this driver reproduces the audited
results_june.csv exactly on every column that run existed to produce.
The guarantees making that true, do not change them casually:
  * rng stream = default_rng([seed, crc32("dataset|combo|max_k")]);
  * view_labeled_data is called before make_splits on the same stream;
  * fold fractions switch at n < small_n exactly as before;
  * every new output (LAC, pooled reference, delta3Z) is an ADDITIVE
    column computed after, and independently of, all existing ones.
Verify with:  make diff-old OLD=/path/to/results_june.csv
"""
import argparse
import os
import sys
import time
import tomllib
import traceback
import zlib
from pathlib import Path

if "--local" in sys.argv and "--workers" in " ".join(sys.argv):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latticecp.data.loaders import fetch_dataset, view_labeled_data
from latticecp.pipeline import TrialConfig, run_trial
from latticecp.scores import FamilyConfig

CONFIG = tomllib.loads((ROOT / "configs" / "benchmark.toml").read_text())
RESULTS_DIR = ROOT / "results_bench"
SHORTLIST = ROOT / "artifacts" / "shortlist.csv"
CATALOG = ROOT / "artifacts" / "catalog.csv"
CACHE = ROOT / "data_cache"
ALPHAS = tuple(CONFIG["run"]["alphas"])
MIN_ROWS = CONFIG["run"]["min_rows"]
SMALL_N = CONFIG["run"]["small_n"]


def slice_config(forecaster: str):
    sc = CONFIG["slices"][forecaster]
    family = (FamilyConfig(max_support=sc["max_support"])
              if "max_support" in sc else FamilyConfig())
    return sc["n_max"], sc["use_features"], family


def view_is_native(frame: pd.DataFrame, combo: tuple, max_k: int) -> bool:
    """True iff code_column bins NO combo column (mirrors its trigger)."""
    for name in combo:
        series = frame[name]
        if series.dtype.kind in "fiu" and series.nunique() > max_k:
            return False
    return True


def trial_rows(results, view, forecaster, seed, is_native) -> list:
    """Flatten run_trial's per-alpha TrialResults into CSV rows.
    Column set = the audited June schema plus the six additive columns."""
    rows = []
    for res in results:
        fold_sizes = res.meta["fold_sizes"]
        rows.append(dict(
            dataset=str(view.dataset), combo=view.combo, max_k=view.max_k,
            forecaster=forecaster, seed=seed, alpha=res.alpha,
            lattice=view.lattice, n_per_cell=view.n_per_cell,
            scan_delta3_debiased=view.delta3_debiased,
            source=getattr(view, "source", ""),
            is_native=bool(is_native),
            delta3_view_raw=res.delta3_view_raw,
            delta3_view_debiased=res.delta3_view_debiased,
            delta3_raw=res.delta3_raw,
            delta3_debiased=res.delta3_debiased,
            delta3B_raw=res.delta3B_raw,
            delta3B_debiased=res.delta3B_debiased,
            forecast_acc_perhead=res.forecast_acc_perhead,
            forecast_acc_joint=res.forecast_acc_joint,
            selected_index=res.selected_index,
            selected_name=res.selected_name,
            threshold=res.threshold,
            size_selected=res.size_selected,
            coverage_selected=res.coverage_selected,
            size_reference=res.size_reference,
            coverage_reference=res.coverage_reference,
            rel_gain=1.0 - res.size_selected / max(res.size_reference, 1e-12),
            stage1_names="|".join(res.stage1_names),
            stage1_sizes="|".join(f"{s:.2f}" for s in res.stage1_sizes),
            all_sizes="|".join(f"{s:.2f}" for s in res.all_sizes),
            all_coverages="|".join(f"{c:.4f}" for c in res.all_coverages),
            n_train=fold_sizes["train"],
            n_sel_threshold=fold_sizes["sel_threshold"],
            n_sel_size=fold_sizes["sel_size"],
            n_cal=fold_sizes["cal"],
            n_test=fold_sizes["test"],
            n_used=sum(fold_sizes.values()),
            # ---- additive columns (release run)
            size_lac=res.size_lac,
            coverage_lac=res.coverage_lac,
            size_reference_pooled=res.size_reference_pooled,
            coverage_reference_pooled=res.coverage_reference_pooled,
            delta3Z_raw=res.delta3Z_raw,
            delta3Z_debiased=res.delta3Z_debiased,
        ))
    return rows


def run_view(view, forecaster, seed, frame) -> list:
    """One (view, forecaster, seed): the reproduction-critical path."""
    n_max, use_features, family = slice_config(forecaster)
    combo = tuple(view.combo.split("|"))
    rng = np.random.default_rng(
        [seed, zlib.crc32(f"{view.dataset}|{view.combo}|{view.max_k}"
                          .encode())])
    data = view_labeled_data(frame, combo, view.max_k,
                             use_features=use_features, n_max=n_max, rng=rng)
    config = TrialConfig(alphas=ALPHAS, family=family)
    if len(data.cell_index) < SMALL_N:
        config.fold_fractions = CONFIG["fractions_small"]["conditional"]
        config.fold_fractions_unconditional = \
            CONFIG["fractions_small"]["unconditional"]
    results = run_trial(data, config, rng)
    return trial_rows(results, view, forecaster, seed,
                      view_is_native(frame, combo, view.max_k))


def load_shortlist():
    shortlist = pd.read_csv(SHORTLIST)
    return shortlist[shortlist.n >= MIN_ROWS].reset_index(drop=True)


def reference_map():
    return dict(pd.read_csv(CATALOG)[["label", "reference"]]
                .astype(str).itertuples(index=False, name=None))


# ============================ task mode ============================

def main_task(seed, forecaster, chunk_id, n_chunks):
    print("=" * 60, flush=True)
    print(f"  benchmark task: forecaster={forecaster} seed={seed} "
          f"chunk={chunk_id}/{n_chunks} alphas={ALPHAS}", flush=True)
    print(f"  node={os.environ.get('SLURMD_NODENAME', 'local')}", flush=True)
    print("=" * 60, flush=True)

    shortlist = load_shortlist()
    mine = shortlist.iloc[chunk_id::n_chunks]
    print(f"  this task: {len(mine)} of {len(shortlist)} views", flush=True)
    reference_of = reference_map()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outfile = (RESULTS_DIR / f"seed_{seed:04d}_{forecaster}"
                             f"_chunk{chunk_id:02d}of{n_chunks:02d}.csv")
    done = set()
    if outfile.exists():
        prev = pd.read_csv(outfile)
        done = set(zip(prev.dataset.astype(str), prev.combo, prev.max_k))

    frame_cache = {}
    t0, n_done = time.time(), 0
    for v in mine.itertuples():
        key = (str(v.dataset), v.combo, v.max_k)
        if key in done:
            continue
        try:
            if v.dataset not in frame_cache:
                frame_cache.clear()
                ref = reference_of.get(str(v.dataset), str(v.dataset))
                frame_cache[v.dataset] = fetch_dataset(ref, str(CACHE))
            rows = run_view(v, forecaster, seed, frame_cache[v.dataset])
            pd.DataFrame(rows).to_csv(
                outfile, mode="a", index=False, header=not outfile.exists())
            n_done += 1
            if n_done % 25 == 0:
                rate = n_done / (time.time() - t0)
                print(f"  {n_done}/{len(mine)} views ({rate:.2f}/s)",
                      flush=True)
        except Exception:
            with open(RESULTS_DIR / f"fail_{seed:04d}_{forecaster}"
                                    f"_chunk{chunk_id:02d}.log", "a") as f:
                f.write(f"{key}\n{traceback.format_exc()}\n")
    print(f"\n  done: {n_done} views in {(time.time() - t0) / 60:.1f} min "
          f"-> {outfile}", flush=True)


# ============================ local mode ===========================

def _local_job(job):
    """(view namedtuple-as-dict, forecaster, seed) -> ('ok', rows)|('fail', txt)."""
    import collections
    View = collections.namedtuple("View", sorted(job["view"]))
    view = View(**job["view"])
    try:
        frame = fetch_dataset(job["reference"], str(CACHE))
        return "ok", run_view(view, job["forecaster"], job["seed"], frame)
    except Exception:
        return "fail", (f"{job['view']['dataset']} {job['view']['combo']} "
                        f"{job['forecaster']} {job['seed']}\n"
                        f"{traceback.format_exc()}\n")


def main_local(workers, quick, seeds, limit):
    shortlist = load_shortlist()
    if quick:
        limit, seeds = limit or 20, min(seeds, 2)
        forecasters = ["marg"]
    else:
        forecasters = ["marg", "logit"]
    if limit:
        shortlist = shortlist.head(limit)
    reference_of = reference_map()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outfile = RESULTS_DIR / "seed_local_all.csv"
    done = set()
    if outfile.exists():
        prev = pd.read_csv(outfile)
        done = set(zip(prev.dataset.astype(str), prev.combo, prev.max_k,
                       prev.forecaster, prev.seed))
    view_fields = ["dataset", "combo", "max_k", "lattice", "n_per_cell",
                   "delta3_debiased", "source"]
    jobs = [dict(view={f: getattr(v, f, "") for f in view_fields},
                 reference=reference_of.get(str(v.dataset), str(v.dataset)),
                 forecaster=fc, seed=s)
            for v in shortlist.itertuples() for fc in forecasters
            for s in range(seeds)
            if (str(v.dataset), v.combo, v.max_k, fc, s) not in done]
    print(f"{len(shortlist)} views -> {len(jobs)} jobs "
          f"(quick={quick}, workers={workers})")
    fail_log = open(RESULTS_DIR / "fail_local.log", "a")

    def handle(outcome):
        status, payload = outcome
        if status == "ok":
            pd.DataFrame(payload).to_csv(
                outfile, mode="a", index=False, header=not outfile.exists())
        else:
            fail_log.write(payload); fail_log.flush()

    if workers <= 1:
        for job in jobs:
            handle(_local_job(job))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import get_context
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=get_context("spawn")) as pool:
            for fut in as_completed([pool.submit(_local_job, j)
                                     for j in jobs]):
                handle(fut.result())
    print(f"done -> {outfile}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", nargs=4,
                   metavar=("SEED", "FORECASTER", "CHUNK", "NCHUNKS"))
    p.add_argument("--local", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=CONFIG["run"]["seeds"])
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    if args.task:
        seed, forecaster, chunk, nchunks = args.task
        main_task(int(seed), forecaster, int(chunk), int(nchunks))
    elif args.local:
        main_local(args.workers, args.quick, args.seeds, args.limit)
    else:
        p.error("choose --task or --local")
