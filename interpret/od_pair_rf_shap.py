#!/usr/bin/env python3
"""
OD-pair scoring (DL half pass) -> save CSV -> (optional subsample) -> RF -> SHAP

Key idea
- DL scoring uses FULL LSOA feature vector used in training:
    get_lsoa_vector(rec) -> PCA -> enc_O/enc_D -> pair_head -> softplus
- RF/SHAP uses ONLY a user-chosen extractor (domain/strata):
    extract_features(rec) -> small feature vector
  Then X_rf = [O_feats, D_feats].

What you change between experiments
- OUT_TAG
- extract_features(rec)  (most important)
- optionally: SUBSAMPLE_N, SAMPLE_N_OD_PAIRS, CKPT_PATH, etc.

Outputs (under interpret/OD_pair_shap/{OUT_TAG}/)
- od_pair_sample_with_xy.csv   (contains O,D, y_pair_score, log_y, O_*, D_*)
- shap_logOD_pair_score_bar.pdf
- shap_logOD_pair_score_beeswarm.pdf
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

# ------------------------- CONFIG -------------------------
OD_POOL_PARQUET = "interpret/OD_pairs_pool/od_pairs_pool_unique.parquet"   # columns: O,D
LSOA_JSON       = "data/node_features/lsoa21_features_normalized.json"
NODE2LSOA_JSON  = "data/node_features/node_to_lsoa.json"
PCA_MODEL_NPZ   = "data/node_features/pca_model_lsoa21.npz"               # optional
CKPT_PATH       = "param/cv_0/best_stage_1_lr1e-03.pt"

OUT_TAG = "AADT_pairscore_RF_hhtype"
OUT_DIR = Path("interpret/OD_pair_shap") / OUT_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

# pool sampling (before mapping)
SAMPLE_N_OD_PAIRS = 50_000
SEED_SAMPLE = 42

# DL scoring batch
SCORE_BATCH = 50_000

# CSV subsample before RF/SHAP (set None to disable)
SUBSAMPLE_N = 50_000
SUBSAMPLE_SEED = 42

# RF/SHAP params
RF_N_ESTIMATORS = 600
RF_MIN_LEAF     = 3
SHAP_BG         = 1000
SHAP_EXPLAIN    = 3000
SHAP_BATCH      = 500

EPS = 1e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIZE = (6, 6)
MAX_DISPLAY = 20
X_LABEL = ""
OUT_SAMPLE_CSV = OUT_DIR / "od_pair_sample_with_xy.csv"
OUT_BAR_PDF    = OUT_DIR / "shap_logOD_pair_score_bar.svg"
OUT_BEE_PDF    = OUT_DIR / "shap_logOD_pair_score_beeswarm.svg"

# ------------------ import training helpers ------------------
from config import MODEL
from model.dataloader import get_lsoa_vector  # must match training feature construction


# ------------------------- CHANGE THIS -------------------------
def extract_features(rec):
    """
    Return (feats, names) for ONE LSOA record.
    You will redefine/replace this in notebook experiments.

    Example: household types (your current).
    """
    hh = rec["households"]["lv3"]
    idxs = [3, 6, 12]  # 0-based
    feats = [float(hh[i]) for i in idxs]
    names = ["One-person household", "Single family household", "Other household type"]
    return feats, names


# ------------------------- MODEL -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.1, act=nn.ReLU):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), act(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DeepDemandLitePair(nn.Module):
    """enc_O, enc_D, pair_head + softplus -> OD pair score"""
    def __init__(self, lsoa_dim: int):
        super().__init__()
        self.enc_O     = MLP(lsoa_dim, MODEL["node_hidden"], MODEL["node_out"])
        self.enc_D     = MLP(lsoa_dim, MODEL["node_hidden"], MODEL["node_out"])
        self.pair_head = MLP(2 * MODEL["node_out"], MODEL["pair_hidden"], 1)
        self.softplus  = nn.Softplus()

    @torch.no_grad()
    def score_pairs(self, XO: torch.Tensor, XD: torch.Tensor) -> torch.Tensor:
        EO = self.enc_O(XO)
        ED = self.enc_D(XD)
        z = self.pair_head(torch.cat([EO, ED], dim=1)).squeeze(-1)
        return self.softplus(z)


# ------------------------- HELPERS -------------------------
def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def maybe_pca_project(X: np.ndarray, pca_npz_path: str) -> np.ndarray:
    if not pca_npz_path or not os.path.isfile(pca_npz_path):
        print(f"[PCA] No PCA model at {pca_npz_path} -> using raw features.")
        return X.astype(np.float32, copy=False)
    npz = np.load(pca_npz_path, allow_pickle=True)
    mean = npz["mean"].astype(np.float32)
    comps = npz["components"].astype(np.float32)
    Xp = (X - mean) @ comps.T
    print(f"[PCA] Projected to {Xp.shape[1]} dims.")
    return Xp.astype(np.float32)

def build_full_feature_matrix(lsoa_dict, lsoa_codes_sorted):
    rows = []
    for code in lsoa_codes_sorted:
        v = get_lsoa_vector(lsoa_dict[code]).cpu().numpy()
        rows.append(v)
    return np.vstack(rows).astype(np.float32)

def fit_rf_and_shap(X, y, feature_names, out_bar_pdf, out_bee_pdf, title,
                    n_bg=1000, n_explain=3000, random_state=42):
    """
    If cache exists, skip RF+SHAP and just plot from cached (shap_values, X_ex, feature_names).
    Cache is saved under the same OUT_DIR as the plots.
    """
    cache_path = Path(out_bar_pdf).with_suffix(".shap_cache.npz")

    # ------------------ LOAD CACHE IF EXISTS ------------------
    if cache_path.is_file():
        print(f"\n[CACHE] Found: {cache_path} -> skipping RF+SHAP, plotting directly.")
        npz = np.load(cache_path, allow_pickle=True)
        shap_values = npz["shap_values"]
        X_ex = npz["X_ex"]
        cached_names = feature_names

        # sanity
        if len(cached_names) != X_ex.shape[1] or shap_values.shape != X_ex.shape:
            raise RuntimeError(
                f"[CACHE ERROR] Shape mismatch in cache.\n"
                f"shap_values={shap_values.shape}, X_ex={X_ex.shape}, n_names={len(cached_names)}"
            )

        # ---- BAR ----
        plt.figure(figsize=(6, 8))
        shap.summary_plot(shap_values, X_ex, feature_names=cached_names, plot_type="bar", show=False)
        plt.title(f"SHAP – {title} (bar)")
        plt.tight_layout()
        plt.savefig(out_bar_pdf)
        plt.close()

        # ---- BEESWARM ----
        plt.figure(figsize=(6, 8))
        shap.summary_plot(shap_values, X_ex, feature_names=cached_names, plot_type="dot",
                          show=False, plot_size=SIZE, color_bar=False, max_display=MAX_DISPLAY)
        fig = plt.gcf()
        ax = fig.axes[0]          # main plot axis (axes[1] is often the colorbar)
        ax.set_xlabel(X_LABEL)
        plt.tight_layout()
        plt.savefig(out_bee_pdf)
        plt.close()

        print(f"Saved: {out_bar_pdf}")
        print(f"Saved: {out_bee_pdf}")
        return

    # ------------------ OTHERWISE RUN RF + SHAP ------------------
    print(f"\n[RF] Training: {title}")
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=None,
        min_samples_leaf=RF_MIN_LEAF,
        random_state=random_state,
        n_jobs=-1,
        warm_start=True,
    )
    for n in tqdm(range(50, RF_N_ESTIMATORS + 1, 50), desc=f"RF training ({title})"):
        rf.n_estimators = n
        rf.fit(X, y)

    N = X.shape[0]
    rng = np.random.default_rng(random_state)
    bg_idx = rng.choice(N, size=min(n_bg, N), replace=False)
    ex_idx = rng.choice(N, size=min(n_explain, N), replace=False)
    X_bg = X[bg_idx]
    X_ex = X[ex_idx]
    print(f"[SHAP] background={X_bg.shape[0]} explain={X_ex.shape[0]}")

    explainer = shap.TreeExplainer(rf, data=X_bg, feature_perturbation="interventional")

    shap_list = []
    for i in tqdm(range(0, X_ex.shape[0], SHAP_BATCH), desc=f"SHAP ({title})"):
        part = X_ex[i:i + SHAP_BATCH]
        sv = explainer.shap_values(part, check_additivity=False)
        shap_list.append(sv)
    shap_values = np.vstack(shap_list)

    # ------------------ SAVE CACHE ------------------
    np.savez_compressed(
        cache_path,
        shap_values=shap_values.astype(np.float32, copy=False),
        X_ex=X_ex.astype(np.float32, copy=False),
        feature_names=np.array(feature_names, dtype=object),
        title=title,
        random_state=np.int64(random_state),
        n_bg=np.int64(min(n_bg, N)),
        n_explain=np.int64(min(n_explain, N)),
        RF_N_ESTIMATORS=np.int64(RF_N_ESTIMATORS),
        RF_MIN_LEAF=np.int64(RF_MIN_LEAF),
    )
    print(f"[CACHE] Saved: {cache_path}")

    # ------------------ PLOTS ------------------
    plt.figure(figsize=(6, 8))
    shap.summary_plot(shap_values, X_ex, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"SHAP – {title} (bar)")
    plt.tight_layout()
    plt.savefig(out_bar_pdf)
    plt.close()

    plt.figure(figsize=(6, 8))
    shap.summary_plot(shap_values, X_ex, feature_names=feature_names, plot_type="dot",
                      show=False, plot_size=SIZE, color_bar=False, max_display=MAX_DISPLAY)
    fig = plt.gcf()
    ax = fig.axes[0]          # main plot axis (axes[1] is often the colorbar)
    ax.set_xlabel(X_LABEL)
    plt.tight_layout()
    plt.savefig(out_bee_pdf)
    plt.close()

    print(f"Saved: {out_bar_pdf}")
    print(f"Saved: {out_bee_pdf}")


# ------------------------- PIPELINE -------------------------
def main():
    print("[Load] OD pool:", OD_POOL_PARQUET)
    od_pool = pd.read_parquet(OD_POOL_PARQUET, columns=["O", "D"])
    print("[Load] OD pool shape:", od_pool.shape)

    if SAMPLE_N_OD_PAIRS and SAMPLE_N_OD_PAIRS < len(od_pool):
        od_pool = od_pool.sample(n=SAMPLE_N_OD_PAIRS, random_state=SEED_SAMPLE).reset_index(drop=True)
    print(f"[OD] Using {len(od_pool):,} pairs (sampled)")

    print("[Load] LSOA JSON:", LSOA_JSON)
    lsoa = load_json(LSOA_JSON)

    print("[Load] node_to_lsoa:", NODE2LSOA_JSON)
    node_to_lsoa = load_json(NODE2LSOA_JSON)

    lsoa_codes = sorted(lsoa.keys())
    lsoa_index = {c: i for i, c in enumerate(lsoa_codes)}

    # (1) FULL features for DL scoring
    print("[Prep] FULL LSOA matrix via get_lsoa_vector ...")
    X_full_raw = build_full_feature_matrix(lsoa, lsoa_codes)
    print("[Prep] FULL matrix:", X_full_raw.shape)

    X_full_in = maybe_pca_project(X_full_raw, PCA_MODEL_NPZ)
    print("[Prep] FULL->PCA:", X_full_in.shape)

    # (2) RF feature cache (per LSOA)
    print("[Prep] RF feature cache (per LSOA) ...")
    rf_cache = []
    rf_names = None
    for code in tqdm(lsoa_codes, desc="RF cache (LSOA)"):
        feats, names = extract_features(lsoa[code])
        rf_cache.append(feats)
        if rf_names is None:
            rf_names = names
    rf_cache = np.asarray(rf_cache, dtype=np.float64)

    # (3) Map node -> LSOA index
    print("[Prep] Mapping node ids -> LSOA indices ...")
    O_nodes = od_pool["O"].astype(str).to_numpy()
    D_nodes = od_pool["D"].astype(str).to_numpy()

    O_idx = np.full(len(od_pool), -1, dtype=np.int32)
    D_idx = np.full(len(od_pool), -1, dtype=np.int32)

    bad = 0
    for i in tqdm(range(len(od_pool)), desc="node->lsoa"):
        try:
            o_code = node_to_lsoa[O_nodes[i]][0]
            d_code = node_to_lsoa[D_nodes[i]][0]
            O_idx[i] = lsoa_index[o_code]
            D_idx[i] = lsoa_index[d_code]
        except Exception:
            bad += 1

    keep = (O_idx >= 0) & (D_idx >= 0)
    if bad:
        print(f"[WARN] Dropping {bad:,} pairs with missing node->LSOA mapping")
    od_pool = od_pool.loc[keep].reset_index(drop=True)
    O_idx = O_idx[keep]
    D_idx = D_idx[keep]
    print(f"[OD] Remaining after mapping: {len(od_pool):,}")

    # (4) Build X_rf = [O_feats, D_feats]
    XO_rf = rf_cache[O_idx]
    XD_rf = rf_cache[D_idx]
    X_rf = np.concatenate([XO_rf, XD_rf], axis=1).astype(np.float64, copy=False)
    feature_names = [f"O: {n}" for n in rf_names] + [f"D: {n}" for n in rf_names]
    print("[RF] X shape:", X_rf.shape)

    # (5) DL scoring inputs
    XO_model = torch.from_numpy(X_full_in[O_idx]).to(DEVICE)
    XD_model = torch.from_numpy(X_full_in[D_idx]).to(DEVICE)

    # (6) Load model
    in_dim = X_full_in.shape[1]
    model = DeepDemandLitePair(lsoa_dim=in_dim).to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    sd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in sd and sd[k].shape == v.shape}
    sd.update(filtered)
    model.load_state_dict(sd)
    model.eval()
    print(f"[Load] Loaded {len(filtered)} keys into DeepDemandLitePair")

    # (7) Score pairs
    print("[Score] DL half-pass scoring ...")
    batch = SCORE_BATCH if DEVICE == "cuda" else min(50_000, SCORE_BATCH)
    y_parts = []
    with torch.no_grad():
        for i in tqdm(range(0, XO_model.shape[0], batch), desc="pair_score"):
            yb = model.score_pairs(XO_model[i:i+batch], XD_model[i:i+batch])
            y_parts.append(yb.detach().cpu().numpy().astype(np.float64))
    y = np.concatenate(y_parts, axis=0)
    y_log = np.log(y + EPS)

    print("[Score] y stats:",
          f"min={np.nanmin(y):.6g}",
          f"max={np.nanmax(y):.6g}",
          f"mean={np.nanmean(y):.6g}",
          f"nan={np.isnan(y).sum():,}")

    # (8) Save CSV (contains everything RF needs)
    out_df = od_pool.copy()
    out_df["y_pair_score"] = y
    out_df["log_y"] = y_log
    for j, fn in enumerate(feature_names):
        out_df[fn] = X_rf[:, j]
    out_df.to_csv(OUT_SAMPLE_CSV, index=False)
    print("Saved:", OUT_SAMPLE_CSV)

    # (9) Reload CSV + subsample + run RF/SHAP (more stable, resumable)
    df = pd.read_csv(OUT_SAMPLE_CSV)

    if SUBSAMPLE_N is not None and SUBSAMPLE_N < len(df):
        df = df.sample(n=SUBSAMPLE_N, random_state=SUBSAMPLE_SEED).reset_index(drop=True)
    print(f"[RF] Using {len(df):,} rows after SUBSAMPLE_N={SUBSAMPLE_N}")

    y_log2 = df["log_y"].to_numpy(dtype=float)
    feat_cols = [c for c in df.columns if c.startswith("O: ") or c.startswith("D: ")]
    X_rf2 = df[feat_cols].to_numpy(dtype=float)

    mask = np.isfinite(y_log2) & np.isfinite(X_rf2).all(axis=1)
    X_rf2 = X_rf2[mask]
    y_log2 = y_log2[mask]
    print("[RF] Prepared:", X_rf2.shape, y_log2.shape)

    fit_rf_and_shap(
        X_rf2, y_log2, feat_cols,
        OUT_BAR_PDF, OUT_BEE_PDF,
        title="log(OD pair score) ~ [O,D] selected strata (RF)",
        n_bg=SHAP_BG,
        n_explain=SHAP_EXPLAIN,
        random_state=42,
    )

    print("Done.")

if __name__ == "__main__":
    main()