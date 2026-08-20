"""Gravity-interaction baseline (distance-decay), wired to the shared 3-way split.

Per edge, over the screened od_use set:
    S_e(gamma) = sum_{OD} mass_O * mass_D * exp(-gamma * t_OD),   mass = pop + emp (raw LSOA).
Log-linear fit  log(y) ~ a0 + a1 * log(S + eps)  on TRAIN; gamma chosen by minimum VAL MGEH
(DeepDemand's early-stop criterion), then the model is refit on TRAIN and scored on TEST.
Edges with no OD / zero mass (S==0) predict 0, matching DeepDemand's empty-extraction handling.
Random-5 + spatial-9 under the identical DeepDemand folds. Script form of
model/baseline_gravity.ipynb (the submitted version tuned gamma on an ad-hoc 80/20 inner split).

Run:  python model/baseline_gravity.py
Out:  eval/baselines/baseline_gravity_{all_folds,summary}.csv ; param/baselines/gravity_*/pred.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline_utils as bu  # noqa: E402
from config import DATA  # noqa: E402
from model.dataloader import load_json  # noqa: E402

OUT = "eval/baselines"
os.makedirs(OUT, exist_ok=True)

EPS_S = 1e-12
EPS_Y = 1e-12
GAMMA_GRID = np.logspace(-7, -3, 17)          # 1/seconds (t_OD is in seconds)
POP_LEVEL = DATA.get("population_level", "lv3")
EMP_LEVEL = DATA.get("employment_level", "lv3")
RAW_LSOA_JSON = "data/node_features/lsoa21_features_raw.json"


def lsoa_mass(rec):
    pop = emp = 0.0
    p = rec.get("population")
    if isinstance(p, dict) and p.get(POP_LEVEL):
        pop = float(p[POP_LEVEL][0])
    e = rec.get("employment")
    if isinstance(e, dict) and e.get(EMP_LEVEL):
        emp = float(e[EMP_LEVEL][0])
    return pop + emp


def build_edge_cache(all_edges, node_to_lsoa, lsoa_to_mass):
    """edge_id -> (mO, mD, t) raw arrays over screened od_use, or None if empty/missing."""
    def node_mass(n):
        return float(lsoa_to_mass.get(node_to_lsoa[str(n)][0], 0.0))

    cache = {}
    for eid in all_edges:
        fp = f"{bu.SUBGRAPH_ROOT}/{eid}/od_use.feather"
        if not os.path.exists(fp):
            cache[eid] = None
            continue
        try:
            df = pd.read_feather(fp, columns=["O", "D", "t_OD"])
        except Exception:
            df = None
        if df is None or len(df) == 0:
            cache[eid] = None
            continue
        mO = np.array([node_mass(o) for o in df["O"].astype(str)], dtype=np.float64)
        mD = np.array([node_mass(d) for d in df["D"].astype(str)], dtype=np.float64)
        # store only w = mO*mD (float32) + t (float32): ~3x less RAM than 3 float64 arrays
        # (the full float64 cache OOM-killed a 16 GB box).
        cache[eid] = ((mO * mD).astype(np.float32), df["t_OD"].to_numpy(dtype=np.float32))
    return cache


def score(ids, cache, gamma):
    S = np.zeros(len(ids), dtype=np.float64)
    for i, e in enumerate(ids):
        rec = cache.get(e)
        if rec is None:
            continue
        w, t = rec
        S[i] = float(np.sum(w * np.exp(-gamma * t), dtype=np.float64))   # accumulate in float64
    return S


def fit_loglinear(S, y):
    mask = S > 0
    if mask.sum() == 0:
        return None
    m = LinearRegression()
    m.fit(np.log(S[mask] + EPS_S).reshape(-1, 1), np.log(y[mask] + EPS_Y))
    return m


def predict(ids, S, model):
    pred = np.zeros(len(ids), dtype=np.float64)
    if model is None:
        return pred
    mask = S > 0
    if mask.any():
        pred[mask] = np.exp(model.predict(np.log(S[mask] + EPS_S).reshape(-1, 1)))
    return np.clip(pred, 0.0, None)


def run():
    edge_to_gt, all_edges = bu.load_gt_edges()
    node_to_lsoa = load_json("data/node_features/node_to_lsoa.json")
    raw = load_json(RAW_LSOA_JSON)
    lsoa_to_mass = {c: lsoa_mass(r) for c, r in raw.items()}
    print("Building gravity OD cache (raw masses + t_OD)...")
    cache = build_edge_cache(all_edges, node_to_lsoa, lsoa_to_mass)
    y = {e: float(edge_to_gt[e]) for e in all_edges}

    rows = []
    for cv_type in ("random", "spatial"):
        for f, tr, va, te in bu.folds(all_edges, cv_type):
            ytr = np.array([y[e] for e in tr], dtype=np.float64)
            yva = np.array([y[e] for e in va], dtype=np.float64)
            best_g, best_s = float(GAMMA_GRID[0]), np.inf
            for g in GAMMA_GRID:
                g = float(g)
                m = fit_loglinear(score(tr, cache, g), ytr)
                pv = predict(va, score(va, cache, g), m)
                s = bu.val_mgeh(yva, pv)
                if s < best_s:
                    best_s, best_g = s, g
            model = fit_loglinear(score(tr, cache, best_g), ytr)
            sets = {}
            for name, ids in (("train", tr), ("val", va), ("test", te)):
                sets[name] = (list(ids), [y[e] for e in ids],
                              predict(ids, score(ids, cache, best_g), model).tolist())
            bu.save_pred(f"gravity_{cv_type}{f}", sets)
            mt = bu.metrics(sets["test"][1], sets["test"][2])
            rows.append({"split": cv_type, "fold": f, "model": "Gravity", "regime": "interaction",
                         "gamma": best_g, "val_MGEH": best_s,
                         **{f"test_{k}": v for k, v in mt.items()}})
            print(f"[{cv_type}] Gravity f{f}: gamma={best_g:.1e} R2={mt['R2']:.3f} MGEH={mt['MGEH']:.1f}")
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/baseline_gravity_all_folds.csv", index=False)
    df.groupby(["split", "model"]).mean(numeric_only=True).reset_index() \
      .to_csv(f"{OUT}/baseline_gravity_summary.csv", index=False)
    print(f"Saved -> {OUT}/baseline_gravity_*.csv ; preds -> {bu.PRED_ROOT}/gravity_*")


if __name__ == "__main__":
    run()
