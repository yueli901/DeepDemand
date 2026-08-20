"""GAT baseline — the single GNN row (one results-table row). Self-contained.

Per target edge: STAR graph from the stored subgraph (no topology rebuild) — nodes = O/D pool nodes
(meta.json) + the edge endpoints u,v; BIDIRECTIONAL edges {each O<->u, each D<->v, u<->v}. Node
features = LSOA projected through DeepDemand's PCA-64 (identical feature pipeline; u,v zero-init);
edge features = travel time (t_O on O<->u, t_D on D<->v, t_e=t_OD-t_O-t_D on u<->v), normalized /3600.
SHARED node-embedder + SHARED edge-embedder feed GAT layers (Velickovic 2018); readout = concat
final [h_u || h_v] -> MLP -> AADT.

Mirrors DeepDemand's training recipe where applicable so the comparison isolates *architecture*, not
tricks: same PCA-64 inputs, same 3-way split, MSE loss, early-stop on val MGEH, **staged LR
(3e-3->1e-3->3e-4) with drop-on-plateau + restore-best**, patience 15. **Only depth (1/2/3) is tuned**
(hidden/heads fixed). Full per-depth logging -> baseline_gat_depth_sweep.csv.

NOT a 'generic' GNN: it uses our OD extraction + direction + travel time (an OD-catchment GAT, a learned
attention aggregator) but has no O x D cross-terms vs DeepDemand's pair interaction, so it sits below
DeepDemand. Needs torch_geometric. Run: python model/baseline_gat.py
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline_utils as bu  # noqa: E402
from config import DATA  # noqa: E402
from model.dataloader import load_json, get_lsoa_vector  # noqa: E402

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATConv
    HAS_PYG = True
except Exception:
    HAS_PYG = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRAPH_CACHE = "eval/baselines/gat_graphs_pca.pt"   # PCA-64 node feats -> new cache
PCA_MODEL = "data/node_features/pca_model_lsoa21.npz"
TIME_SCALE = 3600.0          # normalize edge travel times (raw seconds) -> O(1), like the node feats
LR_STAGES = (3e-3, 1e-3, 3e-4)   # DeepDemand's staged schedule


def build_bank_pca():
    """LSOA node features through DeepDemand's EXACT PCA-64 pipeline: F = (X - mean) @ comps.T,
    X = get_lsoa_vector per LSOA (same order). Returns (bank, node_to_lsoa, feat_dim=64, vec)."""
    lsoa = load_json(DATA["lsoa_json"])
    node_to_lsoa = load_json("data/node_features/node_to_lsoa.json")
    codes = sorted(lsoa.keys())
    X = np.vstack([get_lsoa_vector(lsoa[c]).cpu().numpy() for c in codes]).astype(np.float32)
    npz = np.load(PCA_MODEL, allow_pickle=True)
    mean = npz["mean"].astype(np.float32); comps = npz["components"].astype(np.float32)
    F_pca = ((X - mean) @ comps.T).astype(np.float32)   # identical to DeepDemand's F.npy
    bank = {c: F_pca[i] for i, c in enumerate(codes)}
    feat_dim = F_pca.shape[1]

    def vec(node):
        return bank[node_to_lsoa[str(node)][0]]

    print(f"[GAT] PCA-64 node features: {F_pca.shape} (same pipeline as DeepDemand)")
    return bank, node_to_lsoa, feat_dim, vec


def _edge_time(eid):
    """t_e = t_OD - t_O - t_D (constant per edge); median over od_use rows; 0 if no OD."""
    fp = f"{bu.SUBGRAPH_ROOT}/{eid}/od_use.feather"
    if not os.path.exists(fp):
        return 0.0
    try:
        df = pd.read_feather(fp, columns=["t_O", "t_D", "t_OD"])
    except Exception:
        return 0.0
    if len(df) == 0:
        return 0.0
    return float((df["t_OD"] - df["t_O"] - df["t_D"]).median())


def build_graph(eid, y, vec, lsoa_dim):
    O, D = [], []
    mp = f"{bu.SUBGRAPH_ROOT}/{eid}/meta.json"
    if os.path.exists(mp):
        for n, m in json.load(open(mp)).items():
            try:
                v = vec(n)
            except KeyError:
                continue
            (O if m["direction"] == "O" else D).append((v, float(m["travel_time_sec"])))
    nO, nD = len(O), len(D)
    feats = [v for v, _ in O] + [v for v, _ in D]
    feats.append(np.zeros(lsoa_dim, np.float32))   # u (zero-init)
    feats.append(np.zeros(lsoa_dim, np.float32))   # v (zero-init)
    x = torch.tensor(np.array(feats, dtype=np.float32))
    u, w = nO + nD, nO + nD + 1
    te = _edge_time(eid)
    src, dst, ea = [], [], []
    for i, (_, t) in enumerate(O):
        src += [i, u]; dst += [u, i]; ea += [t, t]
    for j, (_, t) in enumerate(D):
        k = nO + j
        src += [k, w]; dst += [w, k]; ea += [t, t]
    src += [u, w]; dst += [w, u]; ea += [te, te]
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
    edge_attr = (torch.tensor(ea, dtype=torch.float32).unsqueeze(1) / TIME_SCALE) if ea else torch.empty((0, 1), dtype=torch.float32)
    is_u = torch.zeros(nO + nD + 2, dtype=torch.bool); is_u[u] = True
    is_v = torch.zeros(nO + nD + 2, dtype=torch.bool); is_v[w] = True
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([float(y)], dtype=torch.float32), is_u=is_u, is_v=is_v)


def load_or_build_graphs(all_edges, edge_to_gt, vec, lsoa_dim):
    if os.path.exists(GRAPH_CACHE):
        blob = torch.load(GRAPH_CACHE, weights_only=False)   # our own trusted cache; PyG Data needs full unpickle (torch>=2.6)
        if set(blob.keys()) == set(all_edges):
            return blob
    print(f"Building {len(all_edges)} star graphs (one-off; cached to {GRAPH_CACHE})...")
    g = {}
    for i, e in enumerate(all_edges):
        g[e] = build_graph(e, edge_to_gt[e], vec, lsoa_dim)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(all_edges)}")
    os.makedirs(os.path.dirname(GRAPH_CACHE), exist_ok=True)
    torch.save(g, GRAPH_CACHE)
    return g


class ODGAT(nn.Module):
    """Shared node-embedder + shared edge-embedder -> GAT layers (edge-feature attention) ->
    concat endpoint embeddings -> MLP. Depth is the tuned hyper-parameter; hidden/heads fixed."""
    def __init__(self, lsoa_dim, hidden=64, heads=4, layers=2, edge_hidden=16):
        super().__init__()
        assert hidden % heads == 0
        self.node_emb = nn.Linear(lsoa_dim, hidden)
        self.edge_emb = nn.Linear(1, edge_hidden)
        self.convs = nn.ModuleList([
            GATConv(hidden, hidden // heads, heads=heads, concat=True,
                    edge_dim=edge_hidden, add_self_loops=False)
            for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, data):
        x = self.node_emb(data.x)
        ea = self.edge_emb(data.edge_attr)
        for conv in self.convs:
            x = F.elu(conv(x, data.edge_index, edge_attr=ea))
        h = torch.cat([x[data.is_u], x[data.is_v]], dim=-1)
        return self.head(h).squeeze(-1)


def _predict(model, ids, graphs, ymean, ystd):
    model.eval()
    loader = DataLoader([graphs[e] for e in ids], batch_size=256, shuffle=False)
    out = []
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            out.append(model(b).cpu().numpy() * ystd + ymean)
    return np.concatenate(out) if out else np.array([])


def _train(model, tr_ids, va_ids, graphs, edge_to_gt, ymean, ystd,
           lr_stages=LR_STAGES, patience=15, max_epochs_per_stage=200):
    """DeepDemand-style: MSE on standardized AADT, early-stop on val MGEH, STAGED LR with
    drop-on-plateau + restore-best between stages. Returns (best_val_mgeh, total_epochs)."""
    loader = DataLoader([graphs[e] for e in tr_ids], batch_size=64, shuffle=True)
    yva = np.array([edge_to_gt[e] for e in va_ids], np.float32)
    best, best_state, total_ep = np.inf, None, 0
    for lr in lr_stages:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        bad = 0
        for _ in range(max_epochs_per_stage):
            total_ep += 1
            model.train()
            for b in loader:
                b = b.to(DEVICE); opt.zero_grad()
                loss = F.mse_loss(model(b), (b.y.view(-1) - ymean) / ystd)
                loss.backward(); opt.step()
            vm = bu.val_mgeh(yva, _predict(model, va_ids, graphs, ymean, ystd))
            if vm < best - 1e-4:
                best, best_state, bad = vm, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)   # restore best before dropping LR
    return best, total_ep


def run_gat(depths=(1, 2, 3), hidden=64, heads=4, edge_hidden=16, patience=15, max_epochs_per_stage=200):
    """Random-5 + spatial-9 on the shared split; per fold tune depth on val MGEH, report on test.
    Logs EVERY depth (val MGEH + test R²) to baseline_gat_depth_sweep.csv; best -> pred.csv + all_folds."""
    if not HAS_PYG:
        raise ImportError("torch_geometric not installed; run `pip install torch_geometric` in the env.")
    edge_to_gt, all_edges = bu.load_gt_edges()
    _, _, lsoa_dim, vec = build_bank_pca()
    graphs = load_or_build_graphs(all_edges, edge_to_gt, vec, lsoa_dim)
    torch.manual_seed(bu.SEED); np.random.seed(bu.SEED)
    rows, sweep = [], []
    for cv_type in ("random", "spatial"):
        for f, tr, va, te in bu.folds(all_edges, cv_type):
            ytr = np.array([edge_to_gt[e] for e in tr], np.float32)
            ymean, ystd = float(ytr.mean()), float(ytr.std() + 1e-8)
            best_val, best_depth, best_sets = np.inf, depths[0], None
            for L in depths:
                model = ODGAT(lsoa_dim, hidden, heads, L, edge_hidden).to(DEVICE)
                vm, nep = _train(model, tr, va, graphs, edge_to_gt, ymean, ystd,
                                 patience=patience, max_epochs_per_stage=max_epochs_per_stage)
                preds = {s: (list(ids), [float(edge_to_gt[e]) for e in ids],
                             _predict(model, ids, graphs, ymean, ystd).tolist())
                         for s, ids in (("train", tr), ("val", va), ("test", te))}
                mt = bu.metrics(preds["test"][1], preds["test"][2])
                sweep.append({"split": cv_type, "fold": f, "depth": L, "epochs": nep, "val_MGEH": vm,
                              "test_R2": mt["R2"], "test_MGEH": mt["MGEH"], "test_wMAPE": mt["wMAPE"]})
                print(f"[{cv_type}] GAT f{f} depth={L}: val_MGEH={vm:.2f} test_R2={mt['R2']:.3f} (ep={nep})")
                if vm < best_val:
                    best_val, best_depth, best_sets = vm, L, preds
            bu.save_pred(f"gat_{cv_type}{f}", best_sets)
            mt = bu.metrics(best_sets["test"][1], best_sets["test"][2])
            rows.append({"split": cv_type, "fold": f, "model": "GAT", "depth": best_depth,
                         "val_MGEH": best_val, **{f"test_{k}": v for k, v in mt.items()}})
            print(f"[{cv_type}] GAT f{f} BEST depth={best_depth}: R2={mt['R2']:.3f} MGEH={mt['MGEH']:.1f}")
    df = pd.DataFrame(rows)
    os.makedirs(bu.OUT_DIR, exist_ok=True)
    df.to_csv(f"{bu.OUT_DIR}/baseline_gat_all_folds.csv", index=False)
    df.groupby(["split", "model"]).mean(numeric_only=True).reset_index() \
      .to_csv(f"{bu.OUT_DIR}/baseline_gat_summary.csv", index=False)
    sw = pd.DataFrame(sweep)
    sw.to_csv(f"{bu.OUT_DIR}/baseline_gat_depth_sweep.csv", index=False)
    print("\n=== per-depth mean TEST R2 over folds ===")
    print(sw.groupby(["split", "depth"]).test_R2.mean().reset_index().to_string(index=False))
    print(f"\nSaved -> {bu.OUT_DIR}/baseline_gat_{{all_folds,summary,depth_sweep}}.csv ; preds -> {bu.PRED_ROOT}/gat_*")
    return df


if __name__ == "__main__":
    run_gat(depths=(1, 2, 3, 4, 5, 6))
