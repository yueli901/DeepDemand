from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from model.dataloader import load_json, get_lsoa_vector, load_seq_npy, make_pair_sequence
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
        # Build one example vector to infer input dim
        any_lsoa = next(iter(self.lsoa_json.values()))
        lsoa_dim = len(get_lsoa_vector(any_lsoa))

        # Node encoders (two towers; independent parameters)
        self.enc_O = MLP(lsoa_dim, MODEL['node_hidden'], MODEL['node_out'])
        self.enc_D = MLP(lsoa_dim, MODEL['node_hidden'], MODEL['node_out'])

        # Pair scorer (maps concat of two node embeddings to a non-negative scalar)
        pair_in = MODEL['node_out'] * 2
        self.pair_head = MLP(pair_in, MODEL['pair_hidden'], 1)
        self.pair_activation = nn.Softplus()  # non-negative scalar

        # GRU on path sequences (input_dim = feature dim in npy)
        self.gru = nn.GRU(
            input_size=MODEL['edge_feature_dim'],
            hidden_size=MODEL['gru_hidden'],
            num_layers=1,
            batch_first=True, # [b, l, d]
            dropout=0.0,
            bidirectional=False,
        )
        self.seq_head = nn.Linear(MODEL['gru_hidden'], 1)  # probability via sigmoid
        self.sigmoid = nn.Sigmoid()

        # Buffers for features to avoid reloading json every call
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)  # just to carry device
        self.node_to_lsoa = load_json("data/node_features/node_to_lsoa.json")

    def get_lsoa_vec_for_node(self, node_id_str):
        lsoa_code = self.node_to_lsoa.get(str(node_id_str), None)[0]
        rec = self.lsoa_json.get(lsoa_code, None)
        v = get_lsoa_vector(rec).to(self.device)
        return v

    def load_node_features(self, node_ids: List[str]) -> torch.Tensor:
        # Build a batch of LSOA vectors and encode with enc_O/enc_D outside (caller decides which)
        vecs = [self.get_lsoa_vec_for_node(n) for n in node_ids]
        return torch.stack(vecs, dim=0)

    def forward(self, edge_id: str) -> torch.Tensor:
        """
        Returns: predicted traffic volume (scalar tensor) for the given ego edge.
        """
        edge_dir = f"data/subgraphs/subgraphs/{edge_id}"
        meta = load_json(f"{edge_dir}/meta.json")

        # Separate O and D node sets; also gather their npy arrays
        O_nodes, D_nodes = [], []
        O_paths, D_paths = {}, {}

        for nid_str, info in meta.items():
            d = info["direction"]
            p = f"{edge_dir}/{d}/{nid_str}.npy"
            arr = load_seq_npy(p)
            if d == "O":
                O_nodes.append(nid_str)
                O_paths[nid_str] = arr
            else:
                D_nodes.append(nid_str)
                D_paths[nid_str] = arr

        # Precompute node embeddings
        O_node_features = self.load_node_features(O_nodes)
        D_node_features = self.load_node_features(D_nodes)

        O_node_embeddings = self.enc_O(O_node_features)  # [nO, d]
        D_node_embeddings = self.enc_D(D_node_features)  # [nD, d]

        # Build pairwise scores in blocks to avoid O(n^2) memory blowup
        total_volume = torch.zeros((), device=self.device)

        # Build dict index to retrieve embeddings by node id
        O_index = {nid: i for i, nid in enumerate(O_nodes)}
        D_index = {nid: i for i, nid in enumerate(D_nodes)}

        # Prepare lists for batching pairs for GRU
        def pairwise_iterator(batch_size: int):
            # simple nested loop chunked on D side
            for o_id in O_nodes:
                o_idx = O_index[o_id]
                # embed O once
                eo = O_node_embeddings[o_idx].unsqueeze(0)  # [1,d]
                # build all D embeddings in chunks
                for start in range(0, len(D_nodes), batch_size):
                    end = min(start + batch_size, len(D_nodes))
                    Ds = D_nodes[start:end]
                    eDs = D_node_embeddings[start:end]  # [B, d]
                    yield o_id, eo, Ds, eDs

        # Iterate and compute contributions
        for o_id, eo, Ds, eDs in pairwise_iterator(MODEL['pair_batch_size']):
            # Pair scalar head (total possible volume regardless of route)
            eo_rep = eo.expand(eDs.size(0), -1)                      # [B, d]
            pair_in = torch.cat([eo_rep, eDs], dim=1)               # [B, 2d]
            z_pair = self.pair_head(pair_in).squeeze(-1)            # [B]
            s_pair = self.pair_activation(z_pair)                   # non-negative scalar

            # Build sequences for each (o, d) pair
            seq_list: List[torch.Tensor] = []
            lengths: List[int] = []
            for d_id in Ds:
                s = make_pair_sequence(O_paths[o_id], D_paths[d_id])
                seq_list.append(s)
                lengths.append(s.size(0))

            # Pad & pack
            seq_list = [t.to(self.device) for t in seq_list]
            padded = pad_sequence(seq_list, batch_first=True)       # [B, Tmax, F]
            packed = pack_padded_sequence(padded, lengths=lengths, batch_first=True, enforce_sorted=False)

            # GRU -> probs
            _, h_n = self.gru(packed)                               # h_n: [1, B, H]
            h_last = h_n[-1]                                        # [B, H]
            logits = self.seq_head(h_last).squeeze(-1)              # [B]
            p_seq = self.sigmoid(logits)                            # probability

            # Accumulate volume
            total_volume = total_volume + torch.sum(s_pair * p_seq)

        return total_volume