# latticecp

Multi-output conformal classification on product label lattices: a
label-only certificate for higher-order structure, an RBF score family
interpolating covariance and counting, and a two-stage
select-then-recalibrate procedure with exact finite-sample coverage.
This repository is the complete, from-zero pipeline behind the paper's
benchmark (108 datasets, 1,728 label views, 10 seeds).

## Architecture

```
latticecp/                 the package (theory-tested core)
  lattice.py               label lattice, codecs, one-hot embedding
  conformal.py             quantile convention, fold structure
  diagnostics.py           IPF projection, Delta3, debias, G^2
  forecasters.py           marginal and per-head logistic forecasters
  generators.py            synthetic laws with closed-form ground truth
  scores.py                Mahalanobis, lattice-RBF family, count score
  selection.py             stage-1 min-size contest, ties to incumbent
  pipeline.py              run_trial: the canonical experiment flow
  data/                    OpenML catalog, loaders, view search, shortlist
configs/benchmark.toml     the pre-registration artifact (runtime knobs)
artifacts/                 FROZEN inputs: catalog.csv, master_scan.csv,
                           shortlist.csv (see Bootstrap)
stages/s00..s07            the numbered pipeline (see Run everything)
slurm/submit_benchmark.sh  the benchmark as a 130-task SLURM array
scripts/diff_reproduction.py   the acceptance test (see Policies)
tests/                     theorem tests + pipeline behavior tests
figures/                   committed outputs of stage 06 (paper figures)
results/, results_bench/, logs/, data_cache/    generated; gitignored
```

## Policies

**Frozen artifacts.** `artifacts/` holds the catalog, the view scan,
and the shortlist as committed, date-stamped files: the benchmark does
not depend on the live OpenML index. The regeneration code ships
(stages 00-02); `make verify-artifacts` checks the cheap deterministic
link (frozen scan -> shortlist) value-exactly against the frozen
shortlist on every machine. The expensive link (catalog -> scan) is
re-runnable via `make scan`.

**Additive columns and the acceptance test.** Relative to the audited
original run, every new feature of this repository (LAC baseline,
pooled-calibration ablation, value-level certificate delta3Z, the
zero-covariate guard) only APPENDS columns; nothing may perturb an
existing one. The machine check is
`make diff-old OLD=/path/to/results_june.csv`, which requires every
shared column to match value-exactly on every row. If it fails, this
repository is not the audited experiment; stop and investigate.

## Setup

```
python -m venv env && source env/bin/activate      # python >= 3.11
pip install -e .
```

## Bootstrap the frozen artifacts (one time)

Copy the three files from the original run into `artifacts/` and commit
them (from the old repo: `results/catalog.csv`,
`results/master_scan.csv`, `results/shortlist.csv`):

```
cp /path/to/old_repo/results/catalog.csv      artifacts/
cp /path/to/old_repo/results/master_scan.csv  artifacts/
cp /path/to/old_repo/results/shortlist.csv    artifacts/
make verify-artifacts        # frozen scan -> shortlist must match exactly
```

Optionally also copy the OpenML cache (`data_cache/`, gitignored) to
avoid re-downloading ~150 datasets; any node with network access will
otherwise populate it on first use.

## Run everything

```
make test              # theorem + plumbing tests            (~1 min)
make test-full         # + Monte-Carlo validity tests        (~10 min)
make verify-artifacts  # determinism of the frozen chain     (seconds)
make quick             # tiny local end-to-end smoke         (~3 min, needs cache/network)

make bench-slurm       # THE benchmark: 130-task array       (marg minutes/task,
                       #   logit ~5-6 h/task; adjust paths in slurm/submit_benchmark.sh)
make merge             # per-task CSVs -> results/results_bench.csv
make diff-old OLD=/path/to/results_june.csv    # acceptance test
make analyze           # the full printed analysis (sections 1-11)
make figures           # figures/fig1.png + appendix grids
make audit             # anomaly hunt: worst views, margins, oracle, failures
```

Full from-zero regeneration (optional; not needed to reproduce the
paper): `python stages/s00_fetch_catalog.py` (network) then
`make scan` (hours) then stage 02 with `--scan
results/master_scan_regen.csv`, and compare against the frozen
artifacts before adopting anything.

## Expected results (10 seeds, alphas 0.2/0.1/0.05/0.02)

Coverage on target for the selected procedure, the incumbent, and every
family member (per-member watchdog, analysis section 2). Selected-vs-
incumbent gain at alpha 0.1: about +0.2% on certified-null views and
+25% (marg) / +18% (logit) on the certified tail, floor at zero for
marg and above -13% for logit (sections 3-4). One view fails in the
conditional slice by construction (it uses every column as labels; the
loader names the reason).

## Troubleshooting

matplotlib missing on the cluster: `pip install matplotlib` in the env.
OpenML fetches fail on compute nodes without internet: warm
`data_cache/` from a node with access (any stage populates it).
SLURM paths: the two ADJUST lines in `slurm/submit_benchmark.sh`
(module/venv and repo root).

License: TODO (decide before making the repository public).
