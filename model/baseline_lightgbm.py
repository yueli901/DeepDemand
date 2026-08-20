"""LightGBM baseline (one results-table row). Gradient-boosted trees on od_structured features
(screened, directional simple-mean of O / D). max_depth selected on val (min MGEH); 500 trees,
lr=0.05, num_leaves=31. Shared 3-way split + scoring via _baseline_utils.
Needs `pip install lightgbm` in the env. Run: python model/baseline_lightgbm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline_utils as bu  # noqa: E402

from lightgbm import LGBMRegressor  # noqa: E402

# Basic-but-honest grid: num_leaves x learning_rate x n_estimators, selected on val.
GRID = [{"num_leaves": nl, "learning_rate": lr, "n_estimators": n}
        for nl in (31, 63, 127) for lr in (0.03, 0.1) for n in (400, 1000)]


def estimator(p):
    return LGBMRegressor(random_state=bu.SEED, n_jobs=-1, verbosity=-1, **p)


if __name__ == "__main__":
    bu.run_tabular("LightGBM", "od_structured", estimator, grid=GRID)
