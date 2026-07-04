"""Stage 00 (network; run once): fetch the OpenML catalog, apply the
frozen filter, merge the curated list, write artifacts/catalog.csv.
The committed artifact is the frozen truth; this stage exists so anyone
can regenerate it and inspect drift against the frozen copy.
    python stages/s00_fetch_catalog.py [--out artifacts/catalog_regen.csv]
"""
import argparse, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latticecp.data.catalog import CatalogConfig, fetch_catalog

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="artifacts/catalog_regen.csv")
    args = p.parse_args()
    merged = fetch_catalog(CatalogConfig())
    df = pd.DataFrame(merged, columns=["label", "reference"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} datasets -> {args.out}")
    print("compare against the frozen artifacts/catalog.csv before "
          "adopting any change.")
