"""Stage-1 selection: pick the family member with the smallest estimated
deployed set. Pure arrays in, result out; the pipeline does all the wiring,
so validity properties live there and this module stays trivially testable.
"""
from dataclasses import dataclass
import numpy as np
from .conformal import conformal_quantile


@dataclass(frozen=True)
class SelectionResult:
    chosen: int                     # index into the family (0 = reference member)
    stage1_thresholds: np.ndarray   # provisional q_tilde per member
    stage1_sizes: np.ndarray        # estimated deployed size per member


def select_min_size(stage1_scores: np.ndarray, size_at, alpha: float,
                    switch_margin: float = 0.0) -> SelectionResult:
    """stage1_scores: (n_members, n_points) scores at the selection labels.
    size_at(member, threshold) -> estimated deployed size.
    Ties break to the EARLIEST member (np.argmin), i.e. the reference;
    with switch_margin > 0, stay on member 0 unless the improvement exceeds it."""
    stage1_scores = np.atleast_2d(stage1_scores)
    thresholds = np.array([conformal_quantile(member_scores, alpha)
                           for member_scores in stage1_scores])
    sizes = np.array([float(size_at(member, thresholds[member]))
                      for member in range(len(thresholds))])
    chosen = int(np.argmin(sizes))
    if sizes[0] - sizes[chosen] <= switch_margin:
        chosen = 0
    return SelectionResult(chosen, thresholds, sizes)
