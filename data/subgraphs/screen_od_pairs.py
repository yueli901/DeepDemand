#!/usr/bin/env python3
"""Shortest-path screening of candidate OD pairs, and assembly of the OD index cache.

For a target edge (u, v), a candidate pair (o, d) is kept when its fastest route uses that edge,
that is when t_O(o) + t_e + t_D(d) equals the network shortest time t_OD(o, d). Origin and
destination pools and their travel times come from the competitive partition; t_OD comes from
the prebuilt reachability tree, so no shortest-path search is performed here.

The kept pairs are written directly as the per-edge OD index cache the model reads. Dense
indices are centroid indices, and the centroid array is sorted by node id.

  python data/subgraphs/screen_od_pairs.py --run --meta data/subgraphs/subgraphs --tau 3600 \
      --out data/subgraphs/od_index_cache
"""
import argparse
import json
import os
import random
import time
from multiprocessing import Pool

import numpy as np

CSR = "data/highway_network/uk_driving_graph_csr.npz"
TREE = "data/subgraphs/reach_4h"
NODE_LSOA = "data/node_features/node_to_lsoa.json"
CANON = "data/subgraphs/od_index_cache"
EPS_RATIO = 1.0 + 1e-6

_S = {}


def _load(meta_dir=None, tau=None, tod_cut=None):
    if _S:
        return
    z = np.load(CSR)
    _S["nodes"] = z["nodes"]
    _S["iptr"] = z["indptr"]; _S["ind"] = z["indices"]; _S["dat"] = z["data"]

    ix = np.load(f"{TREE}/index.npz")
    cent = ix["cent"]; counts = ix["counts"]; chunk = int(ix["chunk"])
    indptr = np.zeros(len(cent) + 1, np.int64)
    np.cumsum(counts, out=indptr[1:])
    total = int(indptr[-1])
    dests = np.empty(total, np.int32)
    times = np.empty(total, np.float64)
    for lo in range(0, len(cent), chunk):
        c = np.load(f"{TREE}/chunk_{lo:06d}.npz")
        s = int(indptr[lo]); e = s + int(c["counts"].sum())
        dests[s:e] = c["dests"]; times[s:e] = c["times"]
    _S["cent"] = cent
    _S["t_indptr"] = indptr; _S["t_dests"] = dests; _S["t_times"] = times
    _S["cpos"] = {int(n): i for i, n in enumerate(cent)}

    node_to_lsoa = json.load(open(NODE_LSOA))
    codes = json.load(open(f"{CANON}/lsoa_codes.json"))
    c2r = {c: i for i, c in enumerate(codes)}
    row = np.full(len(cent), -1, np.int32)
    for i, n in enumerate(cent):
        lst = node_to_lsoa.get(str(int(n)))
        if lst and lst[0] in c2r:
            row[i] = c2r[lst[0]]
    _S["dense_to_row"] = row
    _S["lsoa_codes"] = codes
    _S["meta_dir"] = meta_dir
    _S["tau"] = tau
    # The old screening ran its own Dijkstra at cutoff 2*tau and dropped anything beyond, so
    # pairs with 2*tau < t_OD <= 2*tau + t_e were lost even though they satisfy the equality.
    # The 4 h tree does see them, so the same truncation has to be reimposed or the output is a
    # superset of the published data (measured: 4,653 extra pairs in 14.27M, all in (7200, 7380]).
    # tod_cut is the cutoff of the TREE the pipeline is built on, not the filter radius. The
    # published pipeline used 7200 s, so reproducing it -- or anything derived from its od_use,
    # such as the tau = 45 / 30 caches -- requires 7200 even when tau is smaller. A pipeline
    # built fresh at radius tau uses 2*tau.
    _S["tod_cut"] = float(tod_cut) if tod_cut else (2.0 * tau if tau else None)


def _t_edge(u, v):
    nodes, iptr, ind, dat = _S["nodes"], _S["iptr"], _S["ind"], _S["dat"]
    for a, b in ((u, v), (v, u)):
        ai = int(np.searchsorted(nodes, a)); bi = int(np.searchsorted(nodes, b))
        if ai >= len(nodes) or nodes[ai] != a or bi >= len(nodes) or nodes[bi] != b:
            continue
        s, e = iptr[ai], iptr[ai + 1]
        j = np.searchsorted(ind[s:e], bi)
        if j < e - s and ind[s + j] == bi:
            return float(dat[s + j])
    return None


def screen_edge(eid, with_times=False):
    """Screened pairs of one target edge as (o_cent_idx, d_cent_idx, t_OD).

    With with_times, also returns the per-pair t_O and t_D, which the feature-based baselines
    use when the OD pair table is exported."""
    meta_p = f"{_S['meta_dir']}/{eid}/meta.json"
    if not os.path.exists(meta_p):
        return None
    a, b, _k = eid.split("_")
    u, v = int(a), int(b)
    te = _t_edge(u, v)
    if te is None:
        return None
    tau = _S["tau"]
    cpos = _S["cpos"]
    meta = json.load(open(meta_p))
    Oi = []; tO = []; Di = []; tD = []
    for nid, info in meta.items():
        t = float(info["travel_time_sec"])
        if t > tau:
            continue
        i = cpos.get(int(nid))
        if i is None:
            continue
        if info["direction"] == "O":
            Oi.append(i); tO.append(t)
        else:
            Di.append(i); tD.append(t)
    if not Oi or not Di:
        e16 = np.empty(0, np.uint16); e32 = np.empty(0, np.float32)
        return (e16, e16, e32, e32, e32) if with_times else (e16, e16, e32)
    Di = np.asarray(Di, np.int32); tD = np.asarray(tD, np.float64)
    o = np.argsort(Di); Di = Di[o]; tD = tD[o]

    tp, td, tt = _S["t_indptr"], _S["t_dests"], _S["t_times"]
    Op = []; Dp = []; Tp = []; TOp = []; TDp = []
    for oi, to in zip(Oi, tO):
        s, e = tp[oi], tp[oi + 1]
        if e <= s:
            continue
        row_d = td[s:e]; row_t = tt[s:e]
        j = np.searchsorted(row_d, Di)
        jc = np.clip(j, 0, len(row_d) - 1)
        found = row_d[jc] == Di
        tod = row_t[jc]
        keep = (found & (tod > 0) & (tod <= _S["tod_cut"])
                & ((to + te + tD) / np.where(tod > 0, tod, 1.0) < EPS_RATIO))
        if keep.any():
            n = int(keep.sum())
            Op.append(np.full(n, oi, np.uint16))
            Dp.append(Di[keep].astype(np.uint16))
            Tp.append(tod[keep].astype(np.float32))
            if with_times:
                TOp.append(np.full(n, to, np.float32))
                TDp.append(tD[keep].astype(np.float32))
    if not Op:
        e16 = np.empty(0, np.uint16); e32 = np.empty(0, np.float32)
        return (e16, e16, e32, e32, e32) if with_times else (e16, e16, e32)
    O = np.concatenate(Op); D = np.concatenate(Dp); T = np.concatenate(Tp)
    if with_times:
        return O, D, T, np.concatenate(TOp), np.concatenate(TDp)
    return O, D, T


def _job(args):
    eid, out, emit = args
    r = screen_edge(eid, with_times=emit)
    if r is None:
        return eid, 0, "skip"
    if emit:
        O, D, T, TO, TD = r
    else:
        O, D, T = r
    if len(O):
        np.savez(f"{out}/edges/{eid}.npz", O=O, D=D, t=T)
    if emit:
        import pandas as pd
        cent = _S["cent"]
        d = f"{_S['meta_dir']}/{eid}"
        os.makedirs(d, exist_ok=True)
        pd.DataFrame({"O": cent[O.astype(np.int64)], "D": cent[D.astype(np.int64)],
                      "t_O": TO, "t_D": TD, "t_OD": T}).to_feather(f"{d}/od_use.feather")
    return eid, len(O), "ok"


def validate(n, meta_dir, tau, tod_cut=None):
    import pandas as pd
    _load(meta_dir, tau, tod_cut)
    eids = sorted(d for d in os.listdir(meta_dir)
                  if os.path.exists(f"data/subgraphs/subgraphs/{d}/od_use.feather"))
    random.seed(23)
    sample = random.sample(eids, min(n, len(eids)))
    cent = _S["cent"]
    nchk = same = 0; miss = extra = 0; maxdt = 0.0; totref = 0
    t0 = time.time()
    for eid in sample:
        r = screen_edge(eid)
        if r is None:
            continue
        O, D, T = r
        got = {(int(cent[a]), int(cent[b])): float(t) for a, b, t in zip(O, D, T)}
        df = pd.read_feather(f"data/subgraphs/subgraphs/{eid}/od_use.feather")
        ref = {(int(a), int(b)): float(t)
               for a, b, t in zip(df.O.to_numpy(), df.D.to_numpy(), df.t_OD.to_numpy())}
        nchk += 1; totref += len(ref)
        kr, kg = set(ref), set(got)
        miss += len(kr - kg); extra += len(kg - kr)
        bad = bool(kr - kg or kg - kr)
        for k in kr & kg:
            dt = abs(ref[k] - got[k])
            maxdt = max(maxdt, dt)
            if dt > 1e-2:
                bad = True
        if not bad:
            same += 1
    print("edges checked   : %d  (%.2f s/edge)" % (nchk, (time.time() - t0) / max(nchk, 1)))
    print("edges identical : %d/%d   ref pairs %d" % (same, nchk, totref))
    print("missing / extra : %d / %d" % (miss, extra))
    print("max |t_OD diff| : %.3e s  (float32 storage)" % maxdt)
    print("VERDICT: %s" % ("EXACT MATCH" if same == nchk and nchk else "MISMATCH"))


def run(meta_dir, tau, out, workers, tod_cut=None, emit_od_use=False):
    _load(meta_dir, tau, tod_cut)
    os.makedirs(f"{out}/edges", exist_ok=True)
    np.save(f"{out}/dense_to_row.npy", _S["dense_to_row"])
    json.dump(_S["lsoa_codes"], open(f"{out}/lsoa_codes.json", "w"))
    if os.path.exists(f"{CANON}/F.npy"):
        import shutil
        shutil.copyfile(f"{CANON}/F.npy", f"{out}/F.npy")
    eids = sorted(d for d in os.listdir(meta_dir) if os.path.isdir(f"{meta_dir}/{d}"))
    todo = [(e, out, emit_od_use) for e in eids
            if not os.path.exists(f"{out}/edges/{e}.npz")]
    print("edges=%d todo=%d tau=%.0f workers=%d" % (len(eids), len(todo), tau, workers), flush=True)
    t0 = time.time(); tot = 0; ok = 0
    with Pool(workers, initializer=_load, initargs=(meta_dir, tau, tod_cut)) as p:
        for i, (eid, n, st) in enumerate(p.imap_unordered(_job, todo, chunksize=4)):
            tot += n; ok += (st == "ok")
            if (i + 1) % 250 == 0:
                el = time.time() - t0
                print("  %d/%d  pairs=%d  %.0fs  eta %.0fs" % (
                    i + 1, len(todo), tot, el, el / (i + 1) * (len(todo) - i - 1)), flush=True)
    print("DONE edges=%d pairs=%d in %.0fs -> %s" % (ok, tot, time.time() - t0, out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, default=0)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--meta", default="data/subgraphs/subgraphs")
    ap.add_argument("--tau", type=float, default=3600.0)
    ap.add_argument("--out", default="data/subgraphs/od_index_cache_new")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--emit-od-use", action="store_true",
                    help="also write the per-edge OD pair table, which the feature-based "
                         "baselines read")
    ap.add_argument("--tod-cutoff", type=float, default=0.0,
                    help="tree cutoff in seconds; default 2*tau. Use 7200 to reproduce the "
                         "published pipeline or anything derived from its od_use.")
    a = ap.parse_args()
    if a.validate:
        validate(a.validate, a.meta, a.tau, a.tod_cutoff)
    elif a.run:
        run(a.meta, a.tau, a.out, a.workers, a.tod_cutoff, a.emit_od_use)
    else:
        ap.error("nothing to do")


if __name__ == "__main__":
    main()
