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
                 node_to_lsoa: Dict[str, list] = None,
                 od_cache=None, feature_matrix: torch.Tensor = None,
                 shared_encoder: bool = False):
        super().__init__()
        self.device = TRAINING['device']

        # P0 speedup (optional). When both are provided, forward() uses the cache
        # fast path (no per-step feather read); otherwise it falls back to reading
        # od_use.feather. self.F is the [n_lsoa, dim] feature matrix on device,
        # rows ordered by sorted(lsoa_codes) -- the same order od_cache indexes into.
        self.od_cache = od_cache
        self.F = feature_matrix.to(self.device) if feature_matrix is not None else None
        # dense node index -> LSOA feature row (on device); used by the cache fast path
        self.dense_to_row = (
            torch.as_tensor(od_cache.dense_to_row, dtype=torch.long, device=self.device)
            if od_cache is not None else None
        )

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
        # Node encoder. DEFAULT = two INDEPENDENT towers for origins and destinations (enc_O and
        # enc_D) -- the reported model, unchanged from the first submission. Tying them into a single
        # shared tower does not affect accuracy and is the AblationSharedEncoder variant
        # (shared_encoder=True).
        self.enc_O = MLP(lsoa_dim, MODEL['node_hidden'], MODEL['node_out'])
        self.enc_D = self.enc_O if shared_encoder else MLP(lsoa_dim, MODEL['node_hidden'], MODEL['node_out'])

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

    def _aggregate(self, S, eps=0.0):
        """Final monotonic transform from the summed OD score S to a flow. Overridden by the
        linear-aggregation ablation. Elementwise, so it serves both the scalar (single-edge)
        and the [B] (batched) paths."""
        return 100 * torch.sqrt(S + eps)

    def forward(self, edge_id: str) -> torch.Tensor:
        """Predicted traffic volume for one target edge.

        An edge with no entry in the OD index cache has no screened OD pair, so its
        contributions sum to zero.
        """
        if self.od_cache.has(edge_id):
            return self._forward_cached(edge_id)
        return self._aggregate(torch.zeros((), device=self.device, requires_grad=True))

    def _forward_cached(self, edge_id: str) -> torch.Tensor:
        """Fast path: OD pairs read from the preloaded index cache (no disk I/O).
        The cache stores a DENSE NODE INDEX per pair in ascending node-id order, so
        torch.unique(dense) reproduces the feather path's np.unique(node-id) exactly
        dense_to_row maps each dense index to its LSOA row for the gather from F."""
        o_dense, d_dense, t_np = self.od_cache.get(edge_id)
        O_all = torch.as_tensor(np.asarray(o_dense).astype(np.int64), device=self.device)
        D_all = torch.as_tensor(np.asarray(d_dense).astype(np.int64), device=self.device)
        tOD_all = torch.as_tensor(np.asarray(t_np), dtype=torch.float32, device=self.device)

        chunk_size = MODEL["chunk_size"]
        N = O_all.shape[0]
        sum_contrib = torch.zeros((), device=self.device, requires_grad=True)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            uniq_O, inv_O = torch.unique(O_all[start:end], return_inverse=True)
            uniq_D, inv_D = torch.unique(D_all[start:end], return_inverse=True)
            t_ODs = tOD_all[start:end]

            O_emb = self.enc_O(self.F[self.dense_to_row[uniq_O]])[inv_O]
            D_emb = self.enc_D(self.F[self.dense_to_row[uniq_D]])[inv_D]

            z_pair = self.pair_head(torch.cat([O_emb, D_emb], dim=1)).squeeze(-1)
            s_pair = self.pair_activation(z_pair)
            p_dist = self.dist_head(t_ODs)

            sum_contrib = sum_contrib + torch.sum(s_pair * p_dist)

        return self._aggregate(sum_contrib)

    def forward_batch(self, edge_ids):
        """Predict a batch of edges in one segment-summed forward.

        Concatenate all edges' OD pairs, push them through the encoders/MLPs together
        (one big GPU op instead of one-per-edge), then a segment-sum (index_add by each
        pair's edge id) recovers per-edge flow sums. Keeps the GPU saturated and
        amortises per-edge Python/launch overhead. Requires the OD cache.

        Batching changes the dropout grouping and the
        optimizer sees a B-edge mini-batch.
        Returns predictions as a [len(edge_ids)] tensor, in edge_ids order.
        """
        # gather each edge's dense O/D indices + travel times, tagged by batch position.
        # an edge with no OD pairs has no cache entry -> empty arrays -> its seg_sum stays
        # 0 -> pred ~ 0 (matches the single-edge path's feather fallback for empty edges).
        O_parts, D_parts, t_parts, seg_parts = [], [], [], []
        for i, eid in enumerate(edge_ids):
            if self.od_cache.has(eid):
                o, d, t = self.od_cache.get(eid)
            else:
                o = d = np.empty(0, np.uint16); t = np.empty(0, np.float32)
            O_parts.append(np.asarray(o)); D_parts.append(np.asarray(d))
            t_parts.append(np.asarray(t)); seg_parts.append(np.full(len(o), i, dtype=np.int64))
        O_all = torch.as_tensor(np.concatenate(O_parts).astype(np.int64), device=self.device)
        D_all = torch.as_tensor(np.concatenate(D_parts).astype(np.int64), device=self.device)
        t_all = torch.as_tensor(np.concatenate(t_parts), dtype=torch.float32, device=self.device)
        seg = torch.as_tensor(np.concatenate(seg_parts), device=self.device)   # pair -> edge index

        B = len(edge_ids)
        P = O_all.shape[0]
        chunk_size = MODEL["chunk_size"]
        seg_sum = torch.zeros(B, device=self.device, requires_grad=True)         # per-edge flow sum

        for start in range(0, P, chunk_size):
            end = min(start + chunk_size, P)
            uniq_O, inv_O = torch.unique(O_all[start:end], return_inverse=True)
            uniq_D, inv_D = torch.unique(D_all[start:end], return_inverse=True)

            O_emb = self.enc_O(self.F[self.dense_to_row[uniq_O]])[inv_O]
            D_emb = self.enc_D(self.F[self.dense_to_row[uniq_D]])[inv_D]
            s_pair = self.pair_activation(self.pair_head(torch.cat([O_emb, D_emb], dim=1)).squeeze(-1))
            p_dist = self.dist_head(t_all[start:end])

            # scatter each pair's contribution into its edge's running total
            seg_sum = seg_sum.index_add(0, seg[start:end], s_pair * p_dist)

        # + eps under sqrt: an edge with no pairs in the batch sums to 0 (sqrt'(0)=inf)
        return self._aggregate(seg_sum, eps=1e-12)