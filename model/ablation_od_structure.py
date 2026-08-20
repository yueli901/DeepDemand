"""Data-construction ablation of DeepDemand's OD prior: pooled vs od_structured .

This is NOT a horse-race baseline row -- it is the ablation of the two preprocessing steps that
build DeepDemand's input and live OUTSIDE the model (so they cannot be ablated in the forward
pass): the competitive two-source Dijkstra extraction and the shortest-path OD-pair screening.

  pooled : mean LSOA over the pooled O- and D-side 1-h extraction zones, UNSCREENED, no direction.
  od_structured : [mean(unique O) || mean(unique D)] over the SCREENED od_use set, directional.

A -> B adds back exactly {local OD region extraction + OD screening + direction}. Per fold we
fit each tree model on both regimes (same shared 3-way split, max_depth on val) and report the
paired ΔR²(B - A): ΔR² > 0 across folds = "why we move to OD zoning rather than a pooled zone".
DeepDemand's learned per-pair interaction then tops od_structured's flat aggregate (separate table).

Models RF + XGB + LightGBM (whatever is installed). Run: python model/ablation_od_structure.py
Out: eval/baselines/od_structure/{all_folds,summary,paired}.csv ; param/baselines/odstruct_*/pred.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline_utils as bu  # noqa: E402

from sklearn.ensemble import RandomForestRegressor  # noqa: E402

OUT_DIR = "eval/baselines/od_structure"
os.makedirs(OUT_DIR, exist_ok=True)

# (estimator_factory, depth grid) for each tree model -- only the installed ones are run.
MODELS = {"RF": (lambda d: RandomForestRegressor(n_estimators=500, max_depth=d,
                                                 random_state=bu.SEED, n_jobs=-1), [None, 16, 24])}
try:
    from xgboost import XGBRegressor
    MODELS["XGB"] = (lambda d: XGBRegressor(max_depth=d, n_estimators=500, learning_rate=0.05,
                                            random_state=bu.SEED, n_jobs=-1, verbosity=0,
                                            tree_method="hist"), [4, 6, 8])
except ImportError:
    print("[WARN] xgboost not installed; XGB arm skipped.")
try:
    from lightgbm import LGBMRegressor
    MODELS["LightGBM"] = (lambda d: LGBMRegressor(max_depth=d, n_estimators=500, learning_rate=0.05,
                                                  num_leaves=31, random_state=bu.SEED, n_jobs=-1,
                                                  verbosity=-1), [4, 6, 8])
except ImportError:
    print("[WARN] lightgbm not installed; LightGBM arm skipped.")


def run():
    XA, all_edges, edge_to_gt = bu.features("pooled")
    XB, _, _ = bu.features("od_structured")
    y = np.array([float(edge_to_gt[e]) for e in all_edges], dtype=np.float32)
    pos = {e: i for i, e in enumerate(all_edges)}

    rows = []
    for cv_type in ("random", "spatial"):
        for model_name, (factory, grid) in MODELS.items():
            for regime, X in (("pooled", XA), ("od_structured", XB)):
                for f, tr, va, te in bu.folds(all_edges, cv_type):
                    itr = np.array([pos[e] for e in tr]); iva = np.array([pos[e] for e in va])
                    ite = np.array([pos[e] for e in te])
                    best_hp, best_s, best_m = grid[0], np.inf, None
                    for hp in grid:
                        m = factory(hp); m.fit(X[itr], y[itr])
                        s = bu.val_mgeh(y[iva], m.predict(X[iva]))
                        if s < best_s:
                            best_s, best_hp, best_m = s, hp, m
                    sets = {
                        "train": (list(tr), y[itr].tolist(), best_m.predict(X[itr]).tolist()),
                        "val":   (list(va), y[iva].tolist(), best_m.predict(X[iva]).tolist()),
                        "test":  (list(te), y[ite].tolist(), best_m.predict(X[ite]).tolist()),
                    }
                    bu.save_pred(f"odstruct_{regime}_{model_name.lower()}_{cv_type}{f}", sets)
                    mt = bu.metrics(sets["test"][1], sets["test"][2])
                    rows.append({"regime": regime, "split": cv_type, "fold": f, "model": model_name,
                                 "hp": str(best_hp), "val_MGEH": best_s,
                                 **{f"test_{k}": v for k, v in mt.items()}})
                    print(f"  {regime}/{model_name}/{cv_type} f{f} hp={best_hp}: "
                          f"R2={mt['R2']:.3f} MGEH={mt['MGEH']:.1f}")
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/all_folds.csv", index=False)
    df.groupby(["regime", "split", "model"]).mean(numeric_only=True).reset_index() \
      .to_csv(f"{OUT_DIR}/summary.csv", index=False)

    # paired A-vs-B per (split, model): does OD structure (B) beat the pooled zone (A) per fold?
    paired = []
    for (split, mdl), sub in df.groupby(["split", "model"]):
        a = sub[sub.regime == "pooled"].set_index("fold")["test_R2"]
        b = sub[sub.regime == "od_structured"].set_index("fold")["test_R2"]
        d = (b - a).dropna()
        paired.append({"split": split, "model": mdl, "mean_dR2_B_minus_A": float(d.mean()),
                       "folds_B_gt_A": int((d > 0).sum()), "n_folds": int(len(d))})
    pd.DataFrame(paired).to_csv(f"{OUT_DIR}/paired.csv", index=False)
    print(f"Saved -> {OUT_DIR}/ (all_folds.csv, summary.csv, paired.csv) ; preds -> {bu.PRED_ROOT}/odstruct_*")


if __name__ == "__main__":
    run()
