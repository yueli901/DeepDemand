#!/usr/bin/env python3
# od_screening_nx_single_source_csv.py
# Process ONE ego edge directory: keep OD pairs with ratio < 1.2 (time-only).
# Writes results directly to CSV to save RAM.

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
import pickle

import numpy as np
import networkx as nx
from tqdm import tqdm

# ===== Fixed settings =====
GPKL_FILE       = "../highway_network/uk_driving_graph_simplified_travel_time_added.gpickle"
TIME_WEIGHT_KEY = "travel_time"
# RATIO_MAX       = 1.2
CUTOFF_SEC      = 2 * 3600  # 2 hours


# ---------------------------
# Helpers
# ---------------------------
def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def min_edge_travel_time(G: nx.MultiDiGraph, u: int, v: int, weight_key: str = TIME_WEIGHT_KEY) -> float:
    if not G.has_edge(u, v):
        raise KeyError(f"No edge {u}->{v} in G")
    best = None
    for _, data in G[u][v].items():  # iterate over parallel edges (keys) between u and v
        w = data.get(weight_key)
        if w is None:
            continue
        best = w if best is None else min(best, w)
    if best is None:
        raise KeyError(f"No '{weight_key}' on edges {u}->{v}")
    return float(best)


def single_source_times(G: nx.MultiDiGraph, source: int, cutoff: float = CUTOFF_SEC):
    """Dijkstra distances only; returns dict node->time."""
    return nx.single_source_dijkstra_path_length(G, source=source, cutoff=cutoff, weight=TIME_WEIGHT_KEY)


def safe_int(s: str) -> int:
    try:
        return int(s)
    except Exception:
        return int(float(s))


def process_one_ego_dir(G: nx.MultiDiGraph, ego_dir: Path) -> None:
    meta_path = ego_dir / "meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Parse ego edge id u_v_k
    u_str, v_str, k_str = ego_dir.name.split("_")
    u_e, v_e, _ = int(u_str), int(v_str), int(k_str)

    # Time on ego edge (u->v). If needed, try reverse as fallback.
    try:
        t_edge = min_edge_travel_time(G, u_e, v_e, TIME_WEIGHT_KEY)
    except KeyError:
        t_edge = min_edge_travel_time(G, v_e, u_e, TIME_WEIGHT_KEY)

    # Split O and D nodes and collect their meta times
    O_nodes, D_nodes = [], []
    tO, tD = {}, {}
    for nid_str, info in meta.items():
        side = info.get("direction")
        tval = float(info.get("travel_time_sec", np.inf))
        nid = safe_int(nid_str)
        if side == "O":
            O_nodes.append(nid); tO[nid] = tval
        elif side == "D":
            D_nodes.append(nid); tD[nid] = tval

    D_set = set(D_nodes)

    # CSV output
    # out_path_candidate = ego_dir / "od_candidates.csv"
    # fw_candidate = open(out_path_candidate, "w", newline="")
    # writer_candidate = csv.writer(fw_candidate)
    # writer_candidate.writerow(["O", "D", "t_O", "t_D", "t_edge", "t_OD", "ratio"])

    out_path_use = ego_dir / "od_use.csv"
    fw_use = open(out_path_use, "w", newline="")
    writer_use = csv.writer(fw_use)
    writer_use.writerow(["O", "D", "t_O", "t_D", "t_edge", "t_OD", "ratio"])

    for o in tqdm(
        O_nodes,
        desc=f"  O-sources ({ego_dir.name})",
        leave=False,
        mininterval=1.0,
        disable=not sys.stderr.isatty(),
    ):
        times_from_o = single_source_times(G, o, cutoff=CUTOFF_SEC)

        for d in D_set:
            t_od = times_from_o.get(d)
            if t_od is None or not np.isfinite(t_od):
                continue

            t_sum = float(tO[o]) + float(t_edge) + float(tD[d])
            ratio = t_sum / t_od

            # Decide which CSV to write to
            if ratio < 1.0 + 1e-6:   # "use ego" file, threshold checked
                writer_use.writerow([o, d, tO[o], tD[d], t_edge, t_od, ratio])
            # elif ratio < RATIO_MAX:  # candidate file
            #     writer_candidate.writerow([o, d, tO[o], tD[d], t_edge, t_od, ratio])

        del times_from_o

    # Close files at the end
    # fw_candidate.close()
    fw_use.close()

    # print(f"[{ts()}] Wrote CSVs → {out_path_candidate}, {out_path_use}")


def parse_args():
    ap = argparse.ArgumentParser(description="OD screening for ONE ego edge directory (time-only).")
    ap.add_argument("--ego-dir", required=True, help="Path to a single ego dir")
    return ap.parse_args()


def main():
    args = parse_args()
    ego_dir = Path(args.ego_dir)
    assert ego_dir.is_dir(), f"Not a directory: {ego_dir}"
    assert (ego_dir / "meta.json").exists(), f"Missing meta.json in {ego_dir}"

    out_path = ego_dir / "od_use.csv"
    if out_path.exists():
        print(f"[{ts()}] {out_path} already exists → skipping.")
        return

    # Load graph (assumes 'travel_time' already exists on edges)
    print(f"[{ts()}] Loading graph: {GPKL_FILE}")
    with open(GPKL_FILE, "rb") as f:
        G: nx.MultiDiGraph = pickle.load(f)

    process_one_ego_dir(G, ego_dir)

    # === Count rows in saved CSV ===
    out_path = ego_dir / "od_use.csv"
    if out_path.exists():
        with open(out_path, "r") as f:
            row_count = sum(1 for _ in f) - 1  # subtract header
        print(f"[{ts()}] {out_path} has {row_count} rows.")
    else:
        print(f"[{ts()}] No CSV written for {ego_dir}.")


if __name__ == "__main__":
    main()