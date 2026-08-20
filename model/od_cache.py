"""Per-edge OD index cache.

Each target edge has its own cache_dir/edges/{edge}.npz holding dense node indices O and D
(uint16) and travel times t (float32). Two global files accompany them: dense_to_row.npy,
mapping each dense index to its feature row, and F.npy, the post-PCA feature matrix.

    cache = OdIndexCache()
    o_dense, d_dense, t = cache.get(edge_id)
    cache.dense_to_row, cache.feature_matrix()
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUBGRAPH_ROOT = "data/subgraphs/subgraphs"
CACHE_DIR = "data/subgraphs/od_index_cache"


class OdIndexCache:
    """Read access to the prebuilt per-edge OD-index cache."""
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        self.edges_dir = f"{cache_dir}/edges"
        self.lsoa_codes = json.load(open(f"{cache_dir}/lsoa_codes.json"))
        self.dense_to_row = np.load(f"{cache_dir}/dense_to_row.npy")   # [N] dense -> LSOA row
        print(f"[OdIndexCache] per-edge, dense_nodes={len(self.dense_to_row)}")

    def feature_matrix(self) -> np.ndarray:
        """The prebuilt [n_lsoa, pca_dim] post-PCA feature matrix F (rows in lsoa_codes order)."""
        return np.load(f"{self.cache_dir}/F.npy")

    def _path(self, edge_id) -> str:
        return f"{self.edges_dir}/{edge_id}.npz"

    def has(self, edge_id) -> bool:
        return os.path.exists(self._path(edge_id))

    def get(self, edge_id):
        # NpzFile[key] reads the full array into memory, so it stays valid after the
        # context manager closes the file handle (no per-step handle leak).
        with np.load(self._path(edge_id)) as z:
            return z["O"], z["D"], z["t"]


def build_feature_matrix(lsoa, lsoa_codes, out_dir=CACHE_DIR, pca_components=64,
                         pca_model_path="data/node_features/pca_model_lsoa21.npz"):
    """Prebuild the post-PCA feature matrix F once (features are fixed; no node-feature
    ablation). Projects with the EXISTING pca model if present, so F is identical to the
    trained run; otherwise computes + saves the PCA model. Writes F.npy to out_dir."""
    from model.dataloader import get_lsoa_vector
    X = np.vstack([get_lsoa_vector(lsoa[c]).cpu().numpy() for c in lsoa_codes]).astype(np.float32)
    if os.path.exists(pca_model_path):
        npz = np.load(pca_model_path, allow_pickle=True)
        mean = npz["mean"].astype(np.float32)
        comps = npz["components"].astype(np.float32)
        F = (X - mean) @ comps.T
        print(f"[F] projected with existing PCA model: {comps.shape[0]} comps")
    else:
        from model.utils import pca_project
        F, mean, comps = pca_project(X, pca_components)
        os.makedirs(os.path.dirname(pca_model_path), exist_ok=True)
        np.savez(pca_model_path, mean=mean, components=comps, codes=np.array(lsoa_codes, dtype=object))
        print(f"[F] computed PCA {pca_components} comps; saved {pca_model_path}")
    np.save(f"{out_dir}/F.npy", F.astype(np.float32))
    print(f"[F] saved {out_dir}/F.npy  shape={F.shape}")
