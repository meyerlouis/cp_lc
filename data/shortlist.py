"""Shortlist construction from a scan table (the select_shortlist logic):
junk/alias drops, a reliability floor on samples-per-cell, and a per-dataset
round-robin across Delta3 bins so no dataset dominates and every dataset
contributes both low- and high-Delta3 views."""

import re
from dataclasses import dataclass, field

import pandas as pd

JUNK_NAME = re.compile(
    r"_seed_\d+|_copy$|_reproduced(?:_\d+)?$|TESTFORDESCRIPTION|autoUniv", re.I
)

# redundant aliases of datasets already present under another name
ALIAS_DROP = {
    "parity5_plus_5",
    "online-shoppers-intention",
    "blastchar",
    "kr-vs-k",
    "thyroid-allrep",
    "thyroid-allbp",
    "thyroid-dis",
}


@dataclass
class ShortlistConfig:
    min_n_per_cell: float = 6.0  # stricter than the scan floor: Delta3
    lattice_max: int = 1100  # is sparsity-inflated below this
    max_per_dataset: int = 16
    bin_column: str = "delta3_debiased"
    bin_edges: tuple = (0, 0.02, 0.05, 0.10, 0.15, 0.25, 0.40, 9.0)


def _round_robin(group: pd.DataFrame, config: ShortlistConfig) -> pd.DataFrame:
    """Interleave the dataset's Delta3 bins (most samples-per-cell first within
    each bin) up to max_per_dataset rows."""
    # deterministic total order: stable sort with explicit tie-breakers
    # (n_per_cell has heavy ties; the historical unstable quicksort made
    # the pick within ties depend on the pandas version)
    bins = [
        rows.sort_values(
            ["n_per_cell", "combo", "max_k"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        for _, rows in group.groupby("d3bin", observed=True)
    ]
    chosen, depth = [], 0
    while len(chosen) < config.max_per_dataset and any(len(b) > depth for b in bins):
        for b in bins:
            if len(b) > depth:
                chosen.append(b.iloc[depth])
                if len(chosen) >= config.max_per_dataset:
                    break
        depth += 1
    return pd.DataFrame(chosen)


def make_shortlist(
    scan: pd.DataFrame, config: ShortlistConfig = ShortlistConfig()
) -> pd.DataFrame:
    """scan columns required: dataset, combo, lattice, n, n_per_cell, and the
    bin column (default delta3_debiased)."""
    d = scan.copy()
    d = d[~d.dataset.astype(str).str.contains(JUNK_NAME, na=False)]
    d = d[~d.dataset.isin(ALIAS_DROP)]
    d = d.drop_duplicates(
        subset=["dataset", "combo", "max_k"] if "max_k" in d else ["dataset", "combo"]
    )
    d = d[
        (d.lattice >= 2)
        & (d.lattice <= config.lattice_max)
        & (d.n_per_cell >= config.min_n_per_cell)
    ]
    d = d.dropna(subset=[config.bin_column])
    d["d3bin"] = pd.cut(d[config.bin_column], list(config.bin_edges), right=False)
    parts = [_round_robin(rows, config) for _, rows in d.groupby("dataset")]
    shortlist = pd.concat(parts, ignore_index=True) if parts else d.iloc[:0].copy()
    return shortlist.drop(columns=["d3bin"], errors="ignore")
