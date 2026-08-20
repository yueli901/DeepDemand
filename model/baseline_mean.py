"""Naive global-mean baseline (the R^2=0 floor), wired to the shared 3-way split.

Predicts the TRAIN mean AADT for every edge (fit on train only -> test stays unbiased),
under the identical random-5 + spatial-9 folds DeepDemand uses. No hyper-parameters, so the
val set is predicted but not tuned on. 
Run:  python model/baseline_mean.py
Out:  eval/baselines/baseline_mean_{all_folds,summary}.csv ; param/baselines/mean_*/pred.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline_utils as bu  # noqa: E402 (sibling import; bu sets cwd=ROOT and path)

OUT = "eval/baselines"
os.makedirs(OUT, exist_ok=True)


def run():
    edge_to_gt, all_edges = bu.load_gt_edges()
    y = {e: float(edge_to_gt[e]) for e in all_edges}
    rows = []
    for cv_type in ("random", "spatial"):
        for f, tr, va, te in bu.folds(all_edges, cv_type):
            mu = float(np.mean([y[e] for e in tr]))
            sets = {name: (list(ids), [y[e] for e in ids], [mu] * len(ids))
                    for name, ids in (("train", tr), ("val", va), ("test", te))}
            bu.save_pred(f"mean_{cv_type}{f}", sets)
            mt = bu.metrics(sets["test"][1], sets["test"][2])
            rows.append({"split": cv_type, "fold": f, "model": "Mean", "regime": "none",
                         **{f"test_{k}": v for k, v in mt.items()}})
            print(f"[{cv_type}] Mean f{f}: R2={mt['R2']:.3f} MGEH={mt['MGEH']:.1f}")
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/baseline_mean_all_folds.csv", index=False)
    df.groupby(["split", "model"]).mean(numeric_only=True).reset_index() \
      .to_csv(f"{OUT}/baseline_mean_summary.csv", index=False)
    print(f"Saved -> {OUT}/baseline_mean_*.csv ; preds -> {bu.PRED_ROOT}/mean_*")


if __name__ == "__main__":
    run()
