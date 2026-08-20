"""Ridge baseline (one results-table row). L2 linear regression on od_structured features
(screened, directional simple-mean of O / D), standardized. alpha selected on val (min MGEH).
Shared 3-way split + scoring via _baseline_utils. Run: python model/baseline_ridge.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline_utils as bu  # noqa: E402

from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

ALPHAS = [0.1, 1.0, 10.0, 100.0]


def estimator(alpha):
    return Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=bu.SEED))])


if __name__ == "__main__":
    bu.run_tabular("Ridge", "od_structured", estimator, grid=ALPHAS)
