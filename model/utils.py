import torch
import random
import numpy as np
from pyproj import Transformer
from config import TRAINING

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

def train_test_sampler(edge_ids, true_probability=TRAINING['train_prop']):
    """
    Splits edge IDs into training and test sets based on probability.
    """
    random.shuffle(edge_ids)
    train_ids = [edge_id for edge_id in edge_ids if random.random() < true_probability]
    test_ids = [edge_id for edge_id in edge_ids if edge_id not in train_ids]
    return train_ids, test_ids

# def train_test_split_spatial(edge_ids):
#     """
#     Split edge_ids into train and test sets based on spatial bounding box.
#     Edges inside the bbox go to test set, others to train.
#     """
#     # Convert bbox to shapely box
#     lat1, lon1, lat2, lon2 = TRAINING['eval_bbox']
#     transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
#     x1, y1 = transformer.transform(lon1, lat1)
#     x2, y2 = transformer.transform(lon2, lat2)
#     test_box = box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

#     # Load geometries
#     with open('edges', 'r') as f:
#         edge_geoms = json.load(f)

#     train_ids, test_ids = [], []

#     for edge_id in edge_ids:
#         geom = shape(edge_geoms[str(edge_id)]['geometry'])
#         if test_box.intersects(geom):
#             test_ids.append(edge_id)
#         else:
#             train_ids.append(edge_id)

#     print(f"Train edges: {len(train_ids)} | Test edges: {len(test_ids)}")
#     return train_ids, test_ids
