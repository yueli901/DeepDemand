"""XGBoost baseline (one results-table row). Gradient-boosted trees on od_structured features
(screened, directional simple-mean of O / D). max_depth selected on val (min MGEH); 500 trees,
lr=0.05. Shared 3-way split + scoring via _baseline_utils. Run: python model/baseline_xgb.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline_utils as bu  # noqa: E402

from xgboost import XGBRegressor  # noqa: E402

# Basic-but-honest grid: depth x learning_rate x n_estimators, selected on val.
GRID = [{"max_depth": d, "learning_rate": lr, "n_estimators": n}
        for d in (4, 6, 8) for lr in (0.03, 0.1) for n in (400, 1000)]


def estimator(p):
    return XGBRegressor(random_state=bu.SEED, n_jobs=-1, verbosity=0, tree_method="hist", **p)


if __name__ == "__main__":
    bu.run_tabular("XGB", "od_structured", estimator, grid=GRID)
