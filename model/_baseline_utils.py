"""Shared engine for the baselines -- this is NOT a baseline model. Every
baseline_*.py imports it so all baselines train/validate/test on the EXACT same edge sets as
DeepDemand and are scored through one code path. (Underscore-prefixed so it sorts apart from the
runnable baseline_<model>.py scripts and is never mistaken for a results-table row.)

Provides:
  * folds(all_edges, cv_type)   -> shared 3-way split == DeepDemand (random-5 / spatial-9)
  * features(regime)            -> cached feature matrix, regime in {'pooled','od_structured'}
  * run_tabular(name, regime, estimator_factory, grid)
        -> full per-fold loop for ONE tabular model on ONE regime: fit on train, select the
           hyper-param on val by min MGEH (DeepDemand's early-stop criterion), report on test,
           save per-edge pred.csv, write eval/baselines/baseline_<name>_{all_folds,summary}.csv
  * metrics / val_mgeh / save_pred / build_bank / load_gt_edges  -> scoring + IO helpers

Discipline (all baselines): fit on TRAIN, select on VAL (min MGEH), report on TEST; val never
folded back, so train == DeepDemand's train. GT is AADT space (load_gt scaler=None), so metrics
are called with scaler=None. The submitted 2-way notebooks are left untouched.
"""
import os
import sys
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from config import DATA, TRAINING
from model.dataloader import load_gt, load_json, get_lsoa_vector
import model.utils as utils

SEED = TRAINING["seed"]          # 1
CV_K = TRAINING["cv_k"]          # 5
VAL_FRAC = 0.125                 # 0.8 * 0.125 = 10% val ; 70/10/20 overall
SUBGRAPH_ROOT = "data/subgraphs/subgraphs"
PRED_ROOT = "param/baselines"            # param/baselines/<run>/pred.csv
OUT_DIR = "eval/baselines"      # per-model all_folds + summary csv
FEAT_DIR = "eval/baselines/features"  # cached feature matrices per regime

np.random.seed(SEED)
torch.manual_seed(SEED)

# Same metric set posthoc_metrics.py computes, so inline numbers == post-hoc numbers.
_METRICS = [("RMSE", utils.RMSE), ("MAE", utils.MAE), ("MGEH", utils.MGEH),
            ("R2", utils.R_square), ("wMAPE", utils.wMAPE), ("SMAPE", utils.SMAPE),
            ("MGEH_hourly", utils.MGEH_hourly), ("MAPE_true", utils.MAPE_true)]


def metrics(gt, pred):
    g = torch.as_tensor(np.asarray(gt, dtype=np.float32))
    p = torch.as_tensor(np.asarray(pred, dtype=np.float32))
    return {n: fn(g, p, None).item() for n, fn in _METRICS}


def val_mgeh(gt, pred):
    g = torch.as_tensor(np.asarray(gt, dtype=np.float32))
    p = torch.as_tensor(np.asarray(pred, dtype=np.float32))
    return utils.MGEH(g, p, None).item()


def load_gt_edges():
    edge_to_gt, scaler = load_gt()
    assert scaler is None, "baselines expect AADT-space GT (load_gt scaler=None)"
    return edge_to_gt, list(edge_to_gt.keys())


def folds(all_edges, cv_type):
    """(fold, train_ids, val_ids, test_ids) under the shared 3-way split == DeepDemand.
    cv_type 'random' -> folds 0..4 ; 'spatial' -> folds 1..9 (regions E12000001..009)."""
    fold_range = range(CV_K) if cv_type == "random" else range(1, 10)
    for f in fold_range:
        tr, va, te = utils.make_split(all_edges, cv_type=cv_type, cv_fold=f, three_way=True,
                                      val_frac=VAL_FRAC, seed=SEED, cv_k=CV_K)
        yield f, tr, va, te


def save_pred(run_name, sets):
    """sets: {set: (edge_ids, gt, pred)} -> param/baselines/<run>/pred.csv (DeepDemand schema)."""
    d = os.path.join(PRED_ROOT, run_name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "pred.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["edge_id", "set", "gt", "pred"])
        for s, (eids, gt, pred) in sets.items():
            for e, g, p in zip(eids, gt, pred):
                w.writerow([e, s, float(g), float(p)])


def build_bank():
    """LSOA feature bank {code: float32 vec} + node->lsoa + vec(node) lookup."""
    lsoa = load_json(DATA["lsoa_json"])
    node_to_lsoa = load_json("data/node_features/node_to_lsoa.json")
    codes = sorted(lsoa.keys())
    bank = {c: get_lsoa_vector(lsoa[c]).cpu().numpy().astype(np.float32) for c in codes}
    feat_dim = next(iter(bank.values())).shape[0]

    def vec(node):
        return bank[node_to_lsoa[str(node)][0]]

    return bank, node_to_lsoa, feat_dim, vec


# ---------------------------------------------------------------------------
# Feature regimes (shared so every model uses the identical feature matrix)
#   pooled : mean LSOA over the pooled O- and D-side 1-h extraction zones
#              (meta.json), UNSCREENED, no direction            -> 121-dim
#   od_structured : [mean(unique O) || mean(unique D)] over the SCREENED od_use set,
#              directional (competitive 2-source Dijkstra + OD screening) -> 242-dim
# ---------------------------------------------------------------------------
def _build_pooled(all_edges, feat_dim, vec):
    X = np.zeros((len(all_edges), feat_dim), np.float32)
    for i, eid in enumerate(all_edges):
        mp = f"{SUBGRAPH_ROOT}/{eid}/meta.json"
        if not os.path.exists(mp):
            continue
        feats = []
        for nid in json.load(open(mp)).keys():
            try:
                feats.append(vec(nid))
            except KeyError:
                pass
        if feats:
            X[i] = np.mean(feats, axis=0)
    return X


def _build_od_structured(all_edges, feat_dim, vec):
    X = np.zeros((len(all_edges), 2 * feat_dim), np.float32)
    for i, eid in enumerate(all_edges):
        fp = f"{SUBGRAPH_ROOT}/{eid}/od_use.feather"
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_feather(fp, columns=["O", "D"])
        except Exception:
            df = None
        if df is None or len(df) == 0:
            continue
        uo = list(dict.fromkeys(df["O"].astype(str)))
        ud = list(dict.fromkeys(df["D"].astype(str)))
        if uo:
            X[i, :feat_dim] = np.mean([vec(n) for n in uo], axis=0)
        if ud:
            X[i, feat_dim:] = np.mean([vec(n) for n in ud], axis=0)
    return X


_BUILDERS = {"pooled": _build_pooled, "od_structured": _build_od_structured}


def features(regime):
    """Cached feature matrix for a regime, aligned to load_gt() edge order.
    Returns (X, all_edges, edge_to_gt). Delete eval/baselines/features/<regime>.npz to rebuild."""
    edge_to_gt, all_edges = load_gt_edges()
    os.makedirs(FEAT_DIR, exist_ok=True)
    path = f"{FEAT_DIR}/{regime}.npz"
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        if d["edges"].tolist() == all_edges:
            return d["X"], all_edges, edge_to_gt
    print(f"Building '{regime}' features (one-off; cached to {path})...")
    _, _, feat_dim, vec = build_bank()
    X = _BUILDERS[regime](all_edges, feat_dim, vec)
    np.savez(path, X=X, edges=np.array(all_edges))
    return X, all_edges, edge_to_gt


def run_tabular(name, regime, estimator_factory, grid):
    """One tabular model (name) on one regime: fit train / select hp on val (min MGEH) / report test;
    save per-edge pred.csv per fold; write baseline_<name>_{all_folds,summary}.csv. Returns the df."""
    X, all_edges, edge_to_gt = features(regime)
    y = np.array([float(edge_to_gt[e]) for e in all_edges], dtype=np.float32)
    pos = {e: i for i, e in enumerate(all_edges)}
    tag = name.lower()
    rows = []
    for cv_type in ("random", "spatial"):
        for f, tr, va, te in folds(all_edges, cv_type):
            itr = np.array([pos[e] for e in tr]); iva = np.array([pos[e] for e in va])
            ite = np.array([pos[e] for e in te])
            best_hp, best_s, best_m = grid[0], np.inf, None
            for hp in grid:
                m = estimator_factory(hp)
                m.fit(X[itr], y[itr])
                s = val_mgeh(y[iva], m.predict(X[iva]))
                if s < best_s:
                    best_s, best_hp, best_m = s, hp, m
            sets = {
                "train": (list(tr), y[itr].tolist(), best_m.predict(X[itr]).tolist()),
                "val":   (list(va), y[iva].tolist(), best_m.predict(X[iva]).tolist()),
                "test":  (list(te), y[ite].tolist(), best_m.predict(X[ite]).tolist()),
            }
            save_pred(f"{tag}_{cv_type}{f}", sets)
            mt = metrics(sets["test"][1], sets["test"][2])
            rows.append({"split": cv_type, "fold": f, "model": name, "regime": regime,
                         "hp": str(best_hp), "val_MGEH": best_s,
                         **{f"test_{k}": v for k, v in mt.items()}})
            print(f"[{cv_type}] {name} f{f} hp={best_hp}: R2={mt['R2']:.3f} MGEH={mt['MGEH']:.1f}")
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(f"{OUT_DIR}/baseline_{tag}_all_folds.csv", index=False)
    df.groupby(["split", "model"]).mean(numeric_only=True).reset_index() \
      .to_csv(f"{OUT_DIR}/baseline_{tag}_summary.csv", index=False)
    print(f"Saved -> {OUT_DIR}/baseline_{tag}_*.csv ; preds -> {PRED_ROOT}/{tag}_*")
    return df
