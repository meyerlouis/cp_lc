"""Stage 04: merge per-task CSVs into results/results_bench.csv."""
import glob
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
files = sorted(glob.glob(str(ROOT / "results_bench" / "seed_*.csv")))
if not files:
    raise SystemExit("no per-task CSVs found in results_bench/")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.drop_duplicates(
    ["dataset", "combo", "max_k", "forecaster", "seed", "alpha"])
out = ROOT / "results" / "results_bench.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"{len(files)} files -> {len(df)} rows -> {out}")
print(df.groupby(["forecaster", "alpha"]).size())
