"""OLS baseline (one results-table row). Ordinary least squares on od_structured features
(screened, directional simple-mean of O / D), standardized. No hyper-parameter to tune.
Shared 3-way split + scoring via _baseline_utils. Run: python model/baseline_ols.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline_utils as bu  # noqa: E402

from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402


def estimator(_hp):
    return Pipeline([("sc", StandardScaler()), ("ols", LinearRegression())])


if __name__ == "__main__":
    bu.run_tabular("OLS", "od_structured", estimator, grid=[None])
