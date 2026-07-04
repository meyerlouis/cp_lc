"""Driver smoke test (no network): trial_rows flattens real TrialResults into
a complete, analyzable CSV schema. Run:  python tests/test_driver_rows.py"""
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stages"))
from latticecp.generators import ConditionalLaw
from latticecp.pipeline import LabeledData, TrialConfig, run_trial
from latticecp.scores import FamilyConfig
from s03_run_benchmark import trial_rows, view_is_native

View = namedtuple("View",
                  "dataset combo max_k lattice n_per_cell delta3_debiased")


def main():
    rng = np.random.default_rng(7)
    law = ConditionalLaw(head_size=3, tau=2.0)
    X, cells = law.sample(2500, rng)
    config = TrialConfig(alphas=(0.2, 0.1), family=FamilyConfig(max_support=256))
    view = View("fake", "a|b|c", 3, 27, 92.6, 0.31)

    for X_or_none, forecaster in ((X, "logit"), (None, "marg")):
        results = run_trial(LabeledData(law.lattice, X_or_none, cells),
                            config, rng)
        rows = trial_rows(results, view, forecaster, seed=0, is_native=True)
        df = pd.DataFrame(rows)
        assert len(df) == 2 and set(df.alpha) == {0.2, 0.1}
        # schema completeness: everything the analysis reads must be present
        needed = ["dataset", "combo", "max_k", "forecaster", "seed", "alpha",
                  "lattice", "is_native", "source",
                  "delta3_view_debiased", "delta3_debiased", "delta3B_debiased",
                  "selected_name", "size_selected", "coverage_selected",
                  "size_reference", "coverage_reference", "rel_gain",
                  "stage1_names", "all_sizes", "all_coverages",
                  "size_lac", "coverage_lac",
                  "size_reference_pooled", "coverage_reference_pooled",
                  "delta3Z_debiased",
                  "n_train", "n_sel_threshold", "n_sel_size", "n_cal", "n_test"]
        missing = [c for c in needed if c not in df.columns]
        assert not missing, missing
        names = df.stage1_names.iloc[0].split("|")
        assert names[0] == "mahalanobis"
        assert len(df.all_sizes.iloc[0].split("|")) == len(names)
        assert len(df.all_coverages.iloc[0].split("|")) == len(names)
        assert df.delta3_view_debiased.notna().all()
        if forecaster == "logit":
            assert df.delta3B_debiased.notna().all()
            assert df.delta3Z_debiased.notna().all()
        else:
            assert df.delta3B_debiased.isna().all()
            assert df.delta3Z_debiased.isna().all()
        assert df.size_lac.notna().all()
        assert df.size_reference_pooled.notna().all()
        print(f"{forecaster}: {len(names)} members, "
              f"rel_gain={df.rel_gain.iloc[0]:+.3f}, schema ok")

    # native flag mirrors code_column's binning trigger
    frame = pd.DataFrame({"a": ["x", "y"] * 50,
                          "b": np.arange(100),          # numeric, 100 levels
                          "c": [0, 1] * 50})            # numeric, 2 levels
    assert view_is_native(frame, ("a", "c"), 3)
    assert not view_is_native(frame, ("a", "b"), 3)     # b would be binned
    print("native flag: ok")


if __name__ == "__main__":
    main()
    print("\ndriver rows test passed")
