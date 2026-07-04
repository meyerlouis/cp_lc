"""OpenML catalog selection: pure filtering logic (offline-testable) plus the
curated must-keep list of combinatorial / rule datasets that the automatic
filter misses but the high-Delta3 tail lives in."""
from dataclasses import dataclass
import pandas as pd


@dataclass
class CatalogConfig:
    n_min: int = 1000
    n_max: int = 200_000
    min_symbolic_features: int = 4
    max_features: int = 40
    max_missing_fraction: float = 0.2
    n_keep: int = 500


# (label, openml name-or-id) -- force-included regardless of the auto filter.
CURATED = [
    ("kr-vs-kp", "kr-vs-kp"), ("mushroom", "mushroom"), ("tic-tac-toe", "tic-tac-toe"),
    ("monks-1", "monks-problems-1"), ("monks-2", "monks-problems-2"),
    ("monks-3", "monks-problems-3"), ("nursery", "nursery"), ("car", "car"),
    ("splice", "splice"), ("dna", "dna"), ("connect-4", "40668"), ("kropt", "40664"),
    ("balance-scale", "balance-scale"), ("led7", "40496"), ("led24", "40497"),
    ("poker-8", "155"), ("mux6", "mux6"), ("parity5", "parity5"),
    ("parity5p5", "parity5_plus_5"), ("xd6", "xd6"), ("vote", "vote"),
    ("solar-flare", "solar-flare"), ("hayes-roth", "hayes-roth"), ("tae", "tae"),
    ("primary-tumor", "171"), ("audiology", "audiology"), ("soybean", "soybean"),
    ("lymph", "lymph"), ("adult", "adult"), ("credit-g", "31"), ("cmc", "23"),
    ("vehicle", "54"), ("vowel", "307"), ("segment", "36"), ("satimage", "182"),
    ("letter", "6"), ("waveform", "60"), ("pendigits", "32"), ("optdigits", "28"),
    ("page-blocks", "30"), ("yeast", "181"), ("mfeat-fourier", "14"),
    ("mfeat-factors", "12"), ("spambase", "44"), ("wine-quality", "40498"),
    ("electricity", "151"), ("bank", "1461"), ("magic", "1120"), ("nomao", "1486"),
]


def filter_catalog(catalog: pd.DataFrame, config: CatalogConfig = CatalogConfig()):
    """openml.list_datasets() dataframe -> list of (name, did). Drops inactive,
    too small/large, feature-poor, synthetic-generator (BNG*) and benchmark-
    resample names; deduplicates case/punctuation-insensitive aliases."""
    d = catalog.copy()
    numeric_columns = ["NumberOfInstances", "NumberOfFeatures", "NumberOfClasses",
                       "NumberOfSymbolicFeatures", "NumberOfInstancesWithMissingValues"]
    for column in numeric_columns:
        if column in d:
            d[column] = pd.to_numeric(d[column], errors="coerce")
    if "status" in d:
        d = d[d["status"] == "active"]
    d = d[d.NumberOfInstances.between(config.n_min, config.n_max)]
    d = d[d.NumberOfFeatures <= config.max_features]
    d = d[d.NumberOfClasses >= 2]
    d = d[d.NumberOfSymbolicFeatures >= config.min_symbolic_features]
    d = d[d.NumberOfInstancesWithMissingValues.fillna(0)
          < config.max_missing_fraction * d.NumberOfInstances]
    d = d[~d.name.str.upper().str.startswith("BNG")]
    d = d[~d.name.str.contains(
        r"_seed_\d+|nrows_\d+|ncols_\d+|stratify_(?:true|false)",
        case=False, regex=True)]
    d = d.sort_values("NumberOfInstances", ascending=False)
    d["_key"] = d.name.str.lower().str.replace(r"[-_ ]", "", regex=True)
    d = d.drop_duplicates("_key")
    return list(d.head(config.n_keep)[["name", "did"]]
                .itertuples(index=False, name=None))


def fetch_catalog(config: CatalogConfig = CatalogConfig()):
    """Network call (run on a machine with openml access)."""
    import openml
    raw = openml.datasets.list_datasets(output_format="dataframe")
    auto = filter_catalog(raw, config)
    seen = {str(name).lower() for name, _ in auto}
    merged = auto + [(label, ref) for label, ref in CURATED
                     if label.lower() not in seen]
    return merged
