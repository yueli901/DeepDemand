from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from model.dataloader import load_json, get_lsoa_vector
from config import DATA, MODEL, TRAINING

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

class DistanceProb(nn.Module):
    """
    Learn p(use ego | distance) from a scalar distance (e.g., t_OD seconds).
    """
    def __init__(self, quadratic: bool = True):
        super().__init__()
        self.quadratic = quadratic

        self.register_buffer("mu", torch.tensor(MODEL["t_mean"], dtype=torch.float32))
        self.register_buffer("s",  torch.tensor(MODEL["t_std"],  dtype=torch.float32))

        # logistic weights
        self.alpha = nn.Parameter(torch.zeros(()))
        self.beta  = nn.Parameter(torch.ones(()))
        if quadratic:
            self.beta2 = nn.Parameter(torch.zeros(()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        # d: [N] (float, seconds)
        z = (d - self.mu) / (self.s.abs() + 1e-6)
        if self.quadratic:
            logit = self.alpha + self.beta * z + self.beta2 * (z * z)
        else:
            logit = self.alpha + self.beta * z
        return self.sigmoid(logit)  # [N]

class Mukara(nn.Module):
    """
    Forward(edge_id: str) -> predicted volume (scalar tensor)
    File layout:
      data/
        subgraphs/{edge_id}/meta.json
        subgraphs/{edge_id}/O/{node_id}.npy
        subgraphs/{edge_id}/D/{node_id}.npy
        node_features/LSOA_features_normalized.json
        node_features/node_to_lsoa.json            # mapping node_id(str) -> LSOA code(str)
    """

    def __init__(self):
        super().__init__()
        self.device = TRAINING['device']

        # Load LSOA features once to know dims
        self.lsoa_json = load_json("data/node_features/LSOA_features_normalized.json")
        self.node_to_lsoa = load_json("data/node_features/node_to_lsoa.json")
        self._feat_cache: Dict[str, torch.Tensor] = {}

        # Build one example vector to infer input dim
        any_lsoa = next(iter(self.lsoa_json.values()))
        lsoa_dim = len(get_lsoa_vector(any_lsoa))

        # Node encoders (two towers; independent parameters)
        self.enc_O = MLP(lsoa_dim, MODEL['node_hidden'], MODEL['node_out'])
        self.enc_D = MLP(lsoa_dim, MODEL['node_hidden'], MODEL['node_out'])

        # Pair scorer (maps concat of two node embeddings to a non-negative scalar)
        pair_in = MODEL['node_out'] * 2 + 1
        self.pair_head = MLP(pair_in, MODEL['pair_hidden'], 1)
        self.pair_activation = nn.Softplus()  # non-negative scalar

    def _get_feature_tensor(self, node_id_str: str) -> torch.Tensor:
        """Return node feature tensor on device, cached."""
        t = self._feat_cache.get(node_id_str)
        if t is not None:
            return t
        # build once
        lsoa_code = self.node_to_lsoa[str(node_id_str)][0]
        rec = self.lsoa_json[lsoa_code]
        t = get_lsoa_vector(rec).to(self.device)   # 1D tensor on device
        self._feat_cache[node_id_str] = t
        return t

    def _stack_features(self, node_ids: List[str]) -> torch.Tensor:
        """Stack cached feature tensors for a list of node ids (keeps order)."""
        return torch.stack([self._get_feature_tensor(nid) for nid in node_ids], dim=0)

    def forward(self, edge_id: str) -> torch.Tensor:
        """
        Returns: predicted traffic volume (scalar tensor) for the given ego edge.
        Uses OD pairs from od_use.csv. Processes rows in chunks to control memory.
        """
        edge_dir = f"data/subgraphs/subgraphs/{edge_id}"

        # === Load OD pairs (only the selected ones) ===
        df = pd.read_feather(f"{edge_dir}/od_use.feather", columns=["O","D","t_OD"])
        # print(f"Number of od pairs for {edge_id}: {df.shape[0]}")

        O_all = df["O"].tolist()
        D_all = df["D"].tolist()
        tOD_all = torch.tensor(df["t_OD"].values, dtype=torch.float32, device=self.device)  # [N]

        # === Chunked processing ===
        chunk_size = MODEL["chunk_size"]
        N = len(df)
        total_volume = torch.zeros((), device=self.device)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)

            # slice this chunk
            O_nodes = O_all[start:end]
            D_nodes = D_all[start:end]
            t_ODs   = tOD_all[start:end]                             # [B]

            # unique + inverse indices
            uniq_O, inv_O = np.unique(O_nodes, return_inverse=True)
            uniq_D, inv_D = np.unique(D_nodes, return_inverse=True)

            # ---- Node features -> embeddings (chunk) ----
            O_feats = self._stack_features(uniq_O)               # [B, lsoa_dim]
            D_feats = self._stack_features(uniq_D)               # [B, lsoa_dim]
            O_emb_u   = self.enc_O(O_feats)                            # [B, d]
            D_emb_u   = self.enc_D(D_feats)                            # [B, d]
            # gather back to original order
            O_emb = O_emb_u[torch.from_numpy(inv_O).to(self.device)]
            D_emb = D_emb_u[torch.from_numpy(inv_D).to(self.device)]

            # ---- Pair scorer (chunk) ----
            t_ODs = t_ODs.unsqueeze(1) 
            pair_in = torch.cat([O_emb, D_emb, t_ODs], dim=1)               # [B, 2d+1]
            z_pair  = self.pair_head(pair_in).squeeze(-1)            # [B]
            s_pair  = self.pair_activation(z_pair)                   # [B] ≥ 0

            # ---- Accumulate (keeps gradient graph across chunks) ----
            total_volume = total_volume + torch.sum(s_pair) # scalar

        return torch.sqrt(total_volume) #* torch.tensor(0.00001, device=self.device) #self.global_scale.abs()