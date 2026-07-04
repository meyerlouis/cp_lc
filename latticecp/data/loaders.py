"""Dataset loading: cached OpenML fetch (network calls happen on the user's
machine; everything else is offline-testable) and conversion of a chosen view
into the pipeline's LabeledData."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from ..lattice import Lattice
from ..pipeline import LabeledData
from .views import code_column


def fetch_dataset(name_or_id, cache_dir: str) -> pd.DataFrame:
    """Fetch one OpenML dataset as a dataframe (target column included),
    pickle-cached so each dataset is downloaded exactly once."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{str(name_or_id).replace('/', '_')}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    import openml
    reference = int(name_or_id) if str(name_or_id).isdigit() else str(name_or_id)
    dataset = openml.datasets.get_dataset(
        reference, download_data=True, download_qualities=False,
        download_features_meta_data=False)
    X, y, _, _ = dataset.get_data(dataset_format="dataframe")
    frame = X if y is None else pd.concat([X, y.rename(y.name or "target")], axis=1)
    pickle.dump(frame, open(path, "wb"))
    return frame


def encode_features(frame: pd.DataFrame, max_onehot_levels: int = 10) -> np.ndarray:
    """Numeric columns: median-imputed and standardized. Categorical columns:
    one-hot over the top max_onehot_levels levels. Returns (n, d) float array."""
    blocks = []
    for name in frame.columns:
        series = frame[name]
        if series.dtype.kind in "fiu":
            x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            x = np.where(np.isnan(x), np.nanmedian(x) if np.isfinite(
                np.nanmedian(x)) else 0.0, x)
            spread = x.std()
            blocks.append(((x - x.mean()) / (spread if spread > 0 else 1.0))[:, None])
        else:
            text = series.astype("object").where(pd.notnull(series), "NA").astype(str)
            levels, counts = np.unique(text.to_numpy(), return_counts=True)
            kept = levels[np.argsort(-counts)[:max_onehot_levels]]
            blocks.append((text.to_numpy()[:, None] == kept[None, :]).astype(float))
    return np.concatenate(blocks, axis=1) if blocks else np.zeros((len(frame), 0))


def view_labeled_data(frame: pd.DataFrame, combo: tuple, max_k: int,
                      use_features: bool, n_max: int,
                      rng: np.random.Generator) -> LabeledData:
    """Build the pipeline task for one view: code the combo columns at
    granularity max_k into a lattice; optionally encode all OTHER columns as
    the covariates X (the conditional / logistic slice). Subsamples to n_max."""
    codes = {name: code_column(frame[name], max_k) for name in combo}
    head_sizes = tuple(int(codes[name].max()) + 1 for name in combo)
    if any(k < 2 for k in head_sizes):
        raise ValueError(f"degenerate head in view {combo}: sizes {head_sizes}")
    lattice = Lattice(head_sizes)
    labels = np.stack([codes[name] for name in combo], axis=1)
    cell_index = lattice.encode(labels)

    keep = np.arange(len(frame))
    if len(frame) > n_max:
        keep = np.sort(rng.choice(len(frame), n_max, replace=False))
    X = None
    if use_features:
        feature_columns = [c for c in frame.columns if c not in combo]
        if not feature_columns:
            raise ValueError(
                f"view {combo} uses every column as a label head, leaving "
                "no covariates: infeasible for the conditional slice "
                "(kept in the label-only slice).")
        X = encode_features(frame[feature_columns])[keep]
    return LabeledData(lattice, X, cell_index[keep])
