import torch
import random
import numpy as np
from pyproj import Transformer
from config import TRAINING

import json
from typing import Dict, List, Sequence, Tuple

# Set random seeds for reproducibility
torch.manual_seed(TRAINING['seed'])
random.seed(TRAINING['seed'])

# ---------- simple PCA via SVD (no sklearn dependency) ----------
def pca_project(X: np.ndarray, k: int):
    """
    X: (n_samples, n_features) float32
    returns X_proj (n_samples, k), mean (n_features,), components (k, n_features)
    """
    X = X.astype(np.float32, copy=False)
    mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    Xc = X - mean

    # economy SVD on (n_samples x n_features): Xc = U S Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:k]  # (k, n_features)
    X_proj = Xc @ comps.T

    # ---- Added print message ----
    var_explained = (S**2) / np.sum(S**2)
    cum_var = np.cumsum(var_explained)
    print(f"[PCA] Reduced from {X.shape[1]} → {k} dims; "
          f"explained variance = {cum_var[k-1]*100:.2f}%")

    return X_proj.astype(np.float32), mean, comps.astype(np.float32)

class ScalerZscore:
    def __init__(self, data):
        """
        Standard scaler that normalizes data while ignoring NaN values.
        """
        valid_data = torch.where(torch.isnan(data), torch.zeros_like(data), data)
        count = torch.sum(~torch.isnan(data)).float()

        self.mean = torch.sum(valid_data) / count
        self.std = torch.sqrt(torch.sum((valid_data - self.mean) ** 2) / count)

        print("Mean:", self.mean.item(), "Standard Deviation:", self.std.item())

    def transform(self, data):
        return torch.where(torch.isnan(data), data, (data - self.mean) / (self.std + 1e-8))

    def inverse_transform(self, data):
        return torch.where(torch.isnan(data), data, data * self.std + self.mean)
    

class ScalerMinMax:
    def __init__(self, data, feature_range=(0.0, 1.0)):
        """
        Min-Max scaler that normalizes data while ignoring NaN values.
        Scales input into the given feature_range (default [0,1]).
        """
        # Replace NaNs with +inf/-inf to safely compute min/max over valid entries
        valid_mask = ~torch.isnan(data)
        valid_data = data[valid_mask]

        self.min_val = valid_data.min()
        self.max_val = valid_data.max()
        self.scale = feature_range[1] - feature_range[0]
        self.min_range = feature_range[0]

        print("Min:", self.min_val.item(), "Max:", self.max_val.item())

    def transform(self, data):
        return torch.where(
            torch.isnan(data),
            data,
            (data - self.min_val) / (self.max_val - self.min_val + 1e-8) * self.scale + self.min_range
        )

    def inverse_transform(self, data):
        return torch.where(
            torch.isnan(data),
            data,
            (data - self.min_range) / self.scale * (self.max_val - self.min_val) + self.min_val
        )
    

def MAE(label_z, pred_z, scaler):
    pred = scaler.inverse_transform(pred_z) if scaler else pred_z
    label = scaler.inverse_transform(label_z) if scaler else label_z
    mae = torch.mean(torch.abs(label - pred))
    # print(f"GT: {label}, Pred: {pred}, MAE: {mae}")
    return mae

def MSE(label_z, pred_z, scaler):
    pred = scaler.inverse_transform(pred_z) if scaler else pred_z
    label = scaler.inverse_transform(label_z) if scaler else label_z
    mse = torch.mean((label - pred) ** 2)
    # print(f"GT: {label}, Pred: {pred}, MSE: {mse}")
    return mse

def RMSE(label_z, pred_z, scaler):
    pred = scaler.inverse_transform(pred_z) if scaler else pred_z
    label = scaler.inverse_transform(label_z) if scaler else label_z
    rmse = torch.sqrt(torch.mean((label - pred) ** 2))
    # print(f"GT: {label}, Pred: {pred}, RMSE: {rmse}")
    return rmse

def MAPE(label_z, pred_z, scaler):
    pred = scaler.inverse_transform(pred_z) if scaler else pred_z
    label = scaler.inverse_transform(label_z) if scaler else label_z
    
    mape = torch.mean(torch.abs((label - pred) / (label/2 + pred/2 + 1e-8))) * 100
    # print(f"GT: {label}, Pred: {pred}, MAPE: {mape}")
    return mape

def MGEH(label_z, pred_z, scaler):
    pred = scaler.inverse_transform(pred_z) if scaler else pred_z
    pred = torch.clamp(pred, min=0.0)
    label = scaler.inverse_transform(label_z) if scaler else label_z
    gehs = torch.sqrt(2 * (pred - label) ** 2 / (pred + label + 1e-8))
    mgeh = torch.mean(gehs)
    # print(f"GT: {label}, Pred: {pred}, GEH: {mgeh}")
    return mgeh

def R_square(label_z, pred_z, scaler):
    pred = scaler.inverse_transform(pred_z) if scaler else pred_z
    label = scaler.inverse_transform(label_z) if scaler else label_z

    ss_res = torch.sum((label - pred) ** 2)
    ss_tot = torch.sum((label - torch.mean(label)) ** 2)

    r2 = 1 - ss_res / (ss_tot + 1e-8)
    # print(f"GT: {label}, Pred: {pred}, R^2: {r2}")
    return r2

def kfold_split(edge_ids: Sequence[str], k: int = 5, seed: int = 2) -> List[List[str]]:
    """
    Deterministic K-fold partition of edge_ids.
    - Does NOT mutate the input.
    - Uses a local RNG seeded for reproducibility.
    - Distributes the remainder to the first folds (sizes differ by at most 1).
    """
    # Make a copy so the caller's list isn't mutated
    items = list(edge_ids)

    # Optional: sort for stability w.r.t. upstream ordering
    # (uncomment if you want identical folds even if edge_ids ordering changes)
    # items.sort()

    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    folds: List[List[str]] = []
    start = 0
    for i in range(k):
        fold_size = n // k + (1 if i < (n % k) else 0)
        folds.append(items[start:start+fold_size])
        start += fold_size
    return folds

def get_cv_split(edge_ids: Sequence[str], k: int = 5, fold_idx: int = 0, seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Returns (train_ids, val_ids) for the given fold index [0..k-1].
    Folds are disjoint and deterministic for a fixed seed.
    """
    assert 0 <= fold_idx < k, "fold_idx must be in [0, k-1]"
    folds = kfold_split(edge_ids, k=k, seed=seed)
    val_ids = folds[fold_idx]
    train_ids = [e for i, f in enumerate(folds) if i != fold_idx for e in f]
    return train_ids, val_ids


def get_spatial_cv_split(
    edge_ids: Sequence[str],
    fold_idx: int = 1,
) -> Tuple[List[str], List[str]]:
    """
    Spatial CV split:

      - edge_to_region.json : {edge_id: region_code or null}
        region_code is one of:
          E12000001, ..., E12000009
        plus some edges with None (7 edges) which are NEVER used as a
        standalone region, but are always included in training.

      - fold_idx is 1-based (1..9):
          fold_idx = 1  -> E12000001 as validation region
          ...
          fold_idx = 9  -> E12000009 as validation region

      - Validation edges: all edges whose region == chosen region.
      - Training edges: all remaining edges, including those with region None.

    Arguments
    ---------
    edge_ids : list of edge_ids that are eligible for CV (e.g. from load_gt()).
    fold_idx : which region (1..9) to use as validation.

    Returns
    -------
    train_ids, val_ids : lists of edge_ids.
    """
    # ---- read mapping json ----
    with open("data/traffic_volume/edge_to_region.json", "r") as f:
        edge_to_region: Dict[str, str] = json.load(f)

    # explicit ordered list of the 9 regions (ignoring None)
    regions: List[str] = [
        "E12000001",
        "E12000002",
        "E12000003",
        "E12000004",
        "E12000005",
        "E12000006",
        "E12000007",
        "E12000008",
        "E12000009",
    ]

    if not (1 <= fold_idx <= len(regions)):
        raise ValueError(
            f"fold_idx must be in [1, {len(regions)}] for spatial CV; got {fold_idx}."
        )

    # pick validation region (1-based index)
    val_region = regions[fold_idx - 1]
    print(f"[Spatial CV] Validation region: {val_region}")

    train_ids: List[str] = []
    val_ids: List[str] = []

    for e in edge_ids:
        r = edge_to_region.get(e, None)

        # validation: region exactly matches val_region
        if r == val_region:
            val_ids.append(e)
        else:
            # everything else (including None) goes into training
            train_ids.append(e)

    print(f"[Spatial CV] #val_edges = {len(val_ids)}, #train_edges = {len(train_ids)}")
    return train_ids, val_ids