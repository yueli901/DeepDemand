from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

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

class DistanceProbLogit(nn.Module):
    """
    Learn p(use ego | distance) from a scalar distance (e.g., t_OD seconds).
    """
    def __init__(self):
        super().__init__()

        self.register_buffer("mu", torch.tensor(MODEL["t_mean"], dtype=torch.float32))
        self.register_buffer("s",  torch.tensor(MODEL["t_std"],  dtype=torch.float32))

        # logistic weights
        self.alpha = nn.Parameter(torch.zeros(()))
        self.beta  = nn.Parameter(torch.ones(()))

        self.sigmoid = nn.Sigmoid()

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        # d: [N] (float, seconds)
        if MODEL['t_normalize']:
            d = (d - self.mu) / (self.s.abs() + 1e-6)
        logit = self.alpha + self.beta * d
        return self.sigmoid(logit)  # [N]
    
class DistanceProbMLP(nn.Module):
    """
    Learn p(use ego | distance) from a scalar distance (t, e.g., seconds)
    using a general MLP followed by a sigmoid.
    """
    def __init__(self):
        super().__init__()

        # normalization buffers (same idea as DistanceProbLinear)
        self.register_buffer("mu", torch.tensor(MODEL["t_mean"], dtype=torch.float32))
        self.register_buffer("s",  torch.tensor(MODEL["t_std"],  dtype=torch.float32))

        # use your existing MLP class
        self.mlp = MLP(1, MODEL["t_hidden"], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        # d: [N] (float, seconds)
        if MODEL["t_normalize"]:
            d = (d - self.mu) / (self.s.abs() + 1e-6)
        d = d.view(-1, 1)  # ensure shape [N, 1] for the MLP
        logits = self.mlp(d).squeeze(-1)
        return self.sigmoid(logits)  # [N]


class DeepDemand(nn.Module):
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

    def __init__(self, feature_bank: Dict[str, torch.Tensor] = None,
                 node_to_lsoa: Dict[str, list] = None):
        super().__init__()
        self.device = TRAINING['device']

        # ---------------- feature store ----------------
        # precomputed (raw or PCA) features are supplied by trainer
        self._feature_bank_cpu: Dict[str, torch.Tensor] = feature_bank  # CPU tensors
        self.node_to_lsoa = node_to_lsoa
        # infer dim from first entry
        any_code = next(iter(self._feature_bank_cpu.keys()))
        lsoa_dim = int(self._feature_bank_cpu[any_code].numel())
        print(f"LSOA feature dim (preloaded): {lsoa_dim}")
        self._feat_cache: Dict[str, torch.Tensor] = {}

        # ---------------- encoders & heads ----------------
        # Node encoders (two towers; independent parameters)
        self.enc_O = MLP(lsoa_dim, MODEL['node_hidden'], MODEL['node_out'])
        self.enc_D = MLP(lsoa_dim, MODEL['node_hidden'], MODEL['node_out'])

        # Pair scorer (maps concat of two node embeddings to a non-negative scalar)
        pair_in = MODEL['node_out'] * 2
        self.pair_head = MLP(pair_in, MODEL['pair_hidden'], 1)
        self.pair_activation = nn.Softplus()

        # time deterrence
        if MODEL['t_function'] == 'mlp':
            self.dist_head = DistanceProbMLP()
        if MODEL['t_function'] == 'logit':
            self.dist_head = DistanceProbLogit()

    def _get_feature_tensor(self, node_id_str: str) -> torch.Tensor:
        """Return node feature tensor on device, cached."""
        t = self._feat_cache.get(node_id_str)
        if t is not None:
            return t
        # build once
        lsoa_code = self.node_to_lsoa[str(node_id_str)][0]
        vec = self._feature_bank_cpu[lsoa_code]                # CPU tensor (1D)
        t = vec.to(self.device)  
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
        sum_contrib = torch.zeros((), device=self.device, requires_grad=True)

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
            pair_in = torch.cat([O_emb, D_emb], dim=1)               # [B, 2d]
            z_pair  = self.pair_head(pair_in).squeeze(-1)            # [B]
            s_pair  = self.pair_activation(z_pair)                   # [B] ≥ 0

            # ---- Distance-based probability (chunk) ----
            p_dist  = self.dist_head(t_ODs)                          # [B] in (0,1)

            # ---- Accumulate (keeps gradient graph across chunks) ----
            sum_contrib = sum_contrib + torch.sum(s_pair * p_dist) # scalar

        pred = 100 * torch.sqrt(sum_contrib)

        return pred