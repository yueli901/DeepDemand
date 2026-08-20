"""Random Forest baseline (one results-table row). RF on od_structured features (screened,
directional simple-mean of O / D). max_depth selected on val (min MGEH), 500 trees. Shared
3-way split + scoring via _baseline_utils. Run: python model/baseline_rf.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline_utils as bu  # noqa: E402

from sklearn.ensemble import RandomForestRegressor  # noqa: E402

# Basic-but-honest grid: capacity (max_depth) x regularization (min_samples_leaf), selected on val.
GRID = [{"max_depth": d, "min_samples_leaf": l}
        for d in (None, 16, 24) for l in (1, 5, 20)]


def estimator(p):
    return RandomForestRegressor(n_estimators=500, random_state=bu.SEED, n_jobs=-1, **p)


if __name__ == "__main__":
    bu.run_tabular("RF", "od_structured", estimator, grid=GRID)
