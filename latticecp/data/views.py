"""Multi-head view construction: turn a dataset's columns into label lattices
and score every candidate view with the Delta3 diagnostics.

Design (ported from the megascan analysis):
  * LOW-cardinality-first: every column is coded at granularity max_k in
    {2, 3, 4}; many small heads beat few large ones (higher Delta3 density,
    healthier cells at the same lattice size).
  * BOUNDED search per (dataset, max_k): exhaustive over small orders up to
    exhaust_max combos, greedy growth with one swap-refine pass from the best
    seeds (reaches the high-Delta3 tail), plus random combos (covers the low
    end so the benchmark keeps true negatives). Deterministic given the rng.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import chi2
from ..lattice import Lattice
from ..diagnostics import delta3, dof


# ---------------- column coding ----------------

def code_column(values, max_k: int) -> np.ndarray:
    """Integer-code one column at granularity max_k.
    Numerics with many distinct values: quantile-binned into <= max_k bins
    (missing values get their own bucket). Categoricals: top max_k - 1 levels
    plus 'other'. Returns codes in {0, 1, ...}."""
    series = pd.Series(values)
    if series.dtype.kind in "fiu" and series.nunique() > max_k:
        x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        missing = np.isnan(x)
        codes = np.zeros(len(x), dtype=int)
        if (~missing).any():
            edges = np.unique(np.quantile(x[~missing], np.linspace(0, 1, max_k + 1)))
            if len(edges) >= 3:
                codes[~missing] = np.digitize(x[~missing], edges[1:-1])
        if missing.any():
            codes[missing] = codes.max() + 1
        return codes
    text = series.astype("object").where(pd.notnull(series), "NA").astype(str)
    levels, counts = np.unique(text.to_numpy(), return_counts=True)
    kept = levels if len(levels) <= max_k else levels[np.argsort(-counts)[:max_k - 1]]
    code_of = {level: i for i, level in enumerate(kept)}
    other = len(kept)
    return np.array([code_of.get(v, other) for v in text.to_numpy()], dtype=int)


# ---------------- view scoring ----------------

@dataclass
class ViewRecord:
    combo: tuple              # column names, sorted
    max_k: int
    head_sizes: tuple
    n: int
    lattice: int
    n_per_cell: float
    delta3_raw: float
    delta3_debiased: float
    g2_pvalue: float
    source: str               # "exhaustive" | "greedy" | "random"


@dataclass
class ViewSearchConfig:
    max_k_values: tuple = (2, 3, 4)
    m_min: int = 3
    m_max: int = 10
    lattice_cap: int = 6000
    min_n_per_cell: float = 3.0
    exhaust_orders: tuple = (3, 4)
    exhaust_max: int = 3000          # per order: sample beyond this many combos
    n_random: int = 300
    n_restarts: int = 4              # greedy restarts from the best seed combos
    swap_passes: int = 1
    max_columns: int = 24            # most-informative-first truncation
    ipf_iterations: int = 200        # screening-grade IPF
    ipf_tolerance: float = 1e-9


def _evaluate(coded: dict, combo: tuple, max_k: int, config) -> ViewRecord | None:
    """Diagnostics for one candidate view; None if infeasible."""
    head_sizes = tuple(int(coded[name].max()) + 1 for name in combo)
    if any(k < 2 for k in head_sizes):
        return None
    lattice_size = int(np.prod(head_sizes))
    n = len(coded[combo[0]])
    if lattice_size > config.lattice_cap or n / lattice_size < config.min_n_per_cell:
        return None
    lattice = Lattice(head_sizes)
    labels = np.stack([coded[name] for name in combo], axis=1)
    counts = np.bincount(lattice.encode(labels), minlength=lattice.n_cells)
    raw = delta3(counts / counts.sum(), lattice,
                 max_iterations=config.ipf_iterations,
                 tolerance=config.ipf_tolerance)
    degrees = dof(lattice)
    return ViewRecord(
        combo=tuple(sorted(combo)), max_k=max_k, head_sizes=head_sizes, n=n,
        lattice=lattice_size, n_per_cell=n / lattice_size, delta3_raw=raw,
        delta3_debiased=max(0.0, raw - degrees / (2.0 * n)),
        g2_pvalue=float(chi2.sf(2.0 * n * raw, degrees)), source="")


def _entropy(codes: np.ndarray) -> float:
    p = np.bincount(codes) / len(codes)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def scan_dataset_views(columns: dict, config: ViewSearchConfig,
                       rng: np.random.Generator) -> list:
    """columns: dict column_name -> raw values (equal length). Returns the
    deduplicated list of ViewRecords across all granularities and sources."""
    records = {}

    def add(record, source):
        if record is None:
            return None
        key = (record.combo, record.max_k)
        if key not in records:
            record.source = source
            records[key] = record
        return records[key]

    for max_k in config.max_k_values:
        coded = {name: code_column(values, max_k) for name, values in columns.items()}
        coded = {name: codes for name, codes in coded.items() if codes.max() >= 1}
        names = sorted(coded, key=lambda name: -_entropy(coded[name]))[:config.max_columns]
        if len(names) < config.m_min:
            continue

        # exhaustive over small orders (sampled past exhaust_max)
        for order in config.exhaust_orders:
            if order > len(names):
                continue
            all_combos = list(combinations(names, order))
            if len(all_combos) > config.exhaust_max:
                keep = rng.choice(len(all_combos), config.exhaust_max, replace=False)
                all_combos = [all_combos[i] for i in keep]
            for combo in all_combos:
                add(_evaluate(coded, combo, max_k, config), "exhaustive")

        # greedy growth with swap refinement, from the best small seeds
        seeds = sorted((r for r in records.values()
                        if r.max_k == max_k and len(r.combo) == config.m_min),
                       key=lambda r: -r.delta3_raw)[:config.n_restarts]
        for seed in seeds:
            current = list(seed.combo)
            for _ in range(config.m_min, config.m_max):
                grown = _best_change(coded, current, names, max_k, config,
                                     mode="add")
                if grown is None:
                    break
                current = grown
                for _ in range(config.swap_passes):
                    swapped = _best_change(coded, current, names, max_k, config,
                                           mode="swap")
                    if swapped is None:
                        break
                    current = swapped
                add(_evaluate(coded, tuple(current), max_k, config), "greedy")

        # random combos: low-Delta3 coverage (true negatives matter)
        attempts = 0
        added = 0
        while added < config.n_random and attempts < 5 * config.n_random:
            attempts += 1
            order = int(rng.integers(config.m_min, config.m_max + 1))
            if order > len(names):
                continue
            combo = tuple(rng.choice(names, order, replace=False))
            if (tuple(sorted(combo)), max_k) in records:
                continue
            if add(_evaluate(coded, combo, max_k, config), "random") is not None:
                added += 1
    return list(records.values())


def _best_change(coded, current, names, max_k, config, mode):
    """Best single addition ('add') or member replacement ('swap') by raw
    Delta3; None if nothing feasible improves."""
    if mode == "add":
        candidates = [current + [name] for name in names if name not in current]
        baseline = -np.inf
    else:
        base = _evaluate(coded, tuple(current), max_k, config)
        if base is None:
            return None
        baseline = base.delta3_raw
        candidates = [[name if member == out else member for member in current]
                      for out in current
                      for name in names if name not in current]
    best, best_value = None, baseline
    for candidate in candidates:
        record = _evaluate(coded, tuple(candidate), max_k, config)
        if record is not None and record.delta3_raw > best_value:
            best, best_value = candidate, record.delta3_raw
    return best
