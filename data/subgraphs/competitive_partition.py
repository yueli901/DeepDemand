#!/usr/bin/env python3
"""Competitive origin/destination partition of the local region around each target edge.

For a target edge (u, v), the origin side expands backwards from u over in-edges and the
destination side expands forwards from v over out-edges. Both draw from a single priority queue,
each node is claimed by whichever side reaches it first, and neither side expands through a node
the other has claimed. The target edge itself is excluded from both searches. A node is recorded
with its side and its travel time to or from the target edge.

Runs on the CSR graph, one process per group of edges.

  build the transpose : python data/subgraphs/competitive_partition.py --build-transpose
  run                 : python data/subgraphs/competitive_partition.py --run --cutoff 3600 \
                            --out data/subgraphs/subgraphs --workers 22
"""
import argparse
import json
import os
import random
import time
from multiprocessing import Pool

import numpy as np
from numba import njit

CSR = "data/highway_network/uk_driving_graph_csr.npz"
CSR_T = "data/highway_network/uk_driving_graph_csr_T.npz"
SG = "data/subgraphs/subgraphs"
NODE_LSOA = "data/node_features/node_to_lsoa.json"
HEAP_CAP = 1_000_000        # frontier only; peak measured at 2,918 for a 1 h ball

_S = {}


@njit(cache=True, inline="always")
def _less(t1, s1, n1, t2, s2, n2):
    if t1 != t2:
        return t1 < t2
    if s1 != s2:
        return s1 < s2
    return n1 < n2


@njit(cache=True)
def _partition(iptr_f, ind_f, dat_f, iptr_r, ind_r, dat_r, u, v, cutoff,
               dist_O, dist_D, winner, touched, hp_t, hp_s, hp_n):
    hn = 0
    # heapify([(0.0,"O",u),(0.0,"D",v)]) pops the D entry first
    hp_t[0] = 0.0; hp_s[0] = 0; hp_n[0] = v
    hp_t[1] = 0.0; hp_s[1] = 1; hp_n[1] = u
    hn = 2
    dist_O[u] = 0.0
    dist_D[v] = 0.0
    ntouch = 0
    maxheap = 2

    while hn > 0:
        t = hp_t[0]; s = hp_s[0]; x = hp_n[0]
        hn -= 1
        if hn > 0:
            hp_t[0] = hp_t[hn]; hp_s[0] = hp_s[hn]; hp_n[0] = hp_n[hn]
            i = 0
            while True:
                l = 2 * i + 1
                r = l + 1
                m = i
                if l < hn and _less(hp_t[l], hp_s[l], hp_n[l], hp_t[m], hp_s[m], hp_n[m]):
                    m = l
                if r < hn and _less(hp_t[r], hp_s[r], hp_n[r], hp_t[m], hp_s[m], hp_n[m]):
                    m = r
                if m == i:
                    break
                tt = hp_t[i]; hp_t[i] = hp_t[m]; hp_t[m] = tt
                ss = hp_s[i]; hp_s[i] = hp_s[m]; hp_s[m] = ss
                nn = hp_n[i]; hp_n[i] = hp_n[m]; hp_n[m] = nn
                i = m

        d = dist_O[x] if s == 1 else dist_D[x]
        if t != d:
            continue
        if winner[x] != -1:
            continue
        winner[x] = s
        touched[ntouch] = x
        ntouch += 1
        if t > cutoff:
            continue

        if s == 1:
            lo = iptr_r[x]; hi = iptr_r[x + 1]
            for i in range(lo, hi):
                y = ind_r[i]
                if (y == u and x == v) or (y == v and x == u):
                    continue
                nt = t + dat_r[i]
                if nt < dist_O[y] and nt <= cutoff:
                    dist_O[y] = nt
                    if winner[y] == -1:
                        if hn >= hp_t.shape[0]:
                            return -1, maxheap
                        j = hn
                        hp_t[j] = nt; hp_s[j] = 1; hp_n[j] = y
                        hn += 1
                        if hn > maxheap:
                            maxheap = hn
                        while j > 0:
                            p = (j - 1) >> 1
                            if _less(hp_t[j], hp_s[j], hp_n[j], hp_t[p], hp_s[p], hp_n[p]):
                                tt = hp_t[j]; hp_t[j] = hp_t[p]; hp_t[p] = tt
                                ss = hp_s[j]; hp_s[j] = hp_s[p]; hp_s[p] = ss
                                nn = hp_n[j]; hp_n[j] = hp_n[p]; hp_n[p] = nn
                                j = p
                            else:
                                break
        else:
            lo = iptr_f[x]; hi = iptr_f[x + 1]
            for i in range(lo, hi):
                y = ind_f[i]
                if (x == u and y == v) or (x == v and y == u):
                    continue
                nt = t + dat_f[i]
                if nt < dist_D[y] and nt <= cutoff:
                    dist_D[y] = nt
                    if winner[y] == -1:
                        if hn >= hp_t.shape[0]:
                            return -1, maxheap
                        j = hn
                        hp_t[j] = nt; hp_s[j] = 0; hp_n[j] = y
                        hn += 1
                        if hn > maxheap:
                            maxheap = hn
                        while j > 0:
                            p = (j - 1) >> 1
                            if _less(hp_t[j], hp_s[j], hp_n[j], hp_t[p], hp_s[p], hp_n[p]):
                                tt = hp_t[j]; hp_t[j] = hp_t[p]; hp_t[p] = tt
                                ss = hp_s[j]; hp_s[j] = hp_s[p]; hp_s[p] = ss
                                nn = hp_n[j]; hp_n[j] = hp_n[p]; hp_n[p] = nn
                                j = p
                            else:
                                break
    return ntouch, maxheap


def _load():
    if _S:
        return
    z = np.load(CSR)
    zt = np.load(CSR_T)
    _S["nodes"] = z["nodes"]
    _S["f"] = (z["indptr"], z["indices"], z["data"])
    _S["r"] = (zt["indptr"], zt["indices"], zt["data"])
    n = len(_S["nodes"])
    _S["dist_O"] = np.full(n, np.inf)
    _S["dist_D"] = np.full(n, np.inf)
    _S["winner"] = np.full(n, -1, np.int8)
    _S["touched"] = np.empty(n, np.int32)
    _S["hp_t"] = np.empty(HEAP_CAP, np.float64)
    _S["hp_s"] = np.empty(HEAP_CAP, np.int8)
    _S["hp_n"] = np.empty(HEAP_CAP, np.int32)
    _S["node_lsoa"] = set(json.load(open(NODE_LSOA)).keys())


def partition_edge(u_osm, v_osm, cutoff):
    """Returns [(node_osm, 'O'|'D', t)] in claim order, or None if the edge is not in the graph."""
    _load()
    nodes = _S["nodes"]
    ui = int(np.searchsorted(nodes, u_osm)); vi = int(np.searchsorted(nodes, v_osm))
    if ui >= len(nodes) or nodes[ui] != u_osm or vi >= len(nodes) or nodes[vi] != v_osm:
        return None
    dO, dD, win, tou = _S["dist_O"], _S["dist_D"], _S["winner"], _S["touched"]
    nt, mh = _partition(*_S["f"], *_S["r"], ui, vi, float(cutoff),
                        dO, dD, win, tou, _S["hp_t"], _S["hp_s"], _S["hp_n"])
    if nt < 0:
        raise RuntimeError("heap overflow: raise HEAP_CAP")
    out = []
    for i in range(nt):
        x = tou[i]
        s = win[x]
        out.append((int(nodes[x]), "O" if s == 1 else "D", dO[x] if s == 1 else dD[x]))
    for i in range(nt):                      # reset only what was touched
        x = tou[i]
        dO[x] = np.inf; dD[x] = np.inf; win[x] = -1
    return out, mh


def _meta_from(claims, u_osm, v_osm):
    """Same filter as the notebook: LSOA-mapped nodes only, excluding the ego endpoints."""
    meta = {}
    for nid, side, t in claims:
        if str(nid) not in _S["node_lsoa"] or nid in (u_osm, v_osm):
            continue
        if not np.isfinite(t):
            continue
        meta[str(nid)] = {"direction": side, "travel_time_sec": float(t)}
    return meta


def build_transpose():
    from scipy.sparse import csr_matrix
    z = np.load(CSR)
    n = len(z["nodes"])
    M = csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n))
    T = M.T.tocsr()
    np.savez(CSR_T, indptr=T.indptr.astype(np.int64), indices=T.indices.astype(np.int32),
             data=T.data)
    print("transpose saved: nnz=%d -> %s" % (T.nnz, CSR_T))


def _job(args):
    eid, cutoff, out_dir = args
    try:
        a, b, _k = eid.split("_")
        u, v = int(a), int(b)
    except ValueError:
        return eid, "badname", 0
    r = partition_edge(u, v, cutoff)
    if r is None:
        return eid, "notingraph", 0
    claims, _mh = r
    meta = _meta_from(claims, u, v)
    d = os.path.join(out_dir, eid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return eid, "ok", len(meta)


def validate(n_sample, cutoff=3600.0):
    _load()
    eids = sorted(d for d in os.listdir(SG) if os.path.exists(f"{SG}/{d}/meta.json"))
    random.seed(4)
    sample = random.sample(eids, min(n_sample, len(eids)))
    nchk = same = 0
    diff_dir = diff_t = only_ref = only_new = 0
    maxdt = 0.0
    maxheap = 0
    t0 = time.time()
    for eid in sample:
        a, b, _k = eid.split("_")
        r = partition_edge(int(a), int(b), cutoff)
        if r is None:
            continue
        claims, mh = r
        maxheap = max(maxheap, mh)
        new = _meta_from(claims, int(a), int(b))
        ref = json.load(open(f"{SG}/{eid}/meta.json"))
        nchk += 1
        ks_r, ks_n = set(ref), set(new)
        only_ref += len(ks_r - ks_n); only_new += len(ks_n - ks_r)
        bad = False
        for k in ks_r & ks_n:
            if ref[k]["direction"] != new[k]["direction"]:
                diff_dir += 1; bad = True
            dt = abs(float(ref[k]["travel_time_sec"]) - new[k]["travel_time_sec"])
            if dt > maxdt:
                maxdt = dt
            if dt > 1e-9:
                diff_t += 1; bad = True
        if not bad and not (ks_r - ks_n) and not (ks_n - ks_r):
            same += 1
    el = time.time() - t0
    print("edges checked      : %d  (%.2f s/edge, includes JIT warmup)" % (nchk, el / max(nchk, 1)))
    print("edges identical    : %d/%d" % (same, nchk))
    print("nodes only in ref  : %d" % only_ref)
    print("nodes only in new  : %d" % only_new)
    print("direction mismatch : %d" % diff_dir)
    print("time mismatch      : %d  (max |diff| %.3e s)" % (diff_t, maxdt))
    print("peak heap entries  : %d / %d" % (maxheap, HEAP_CAP))
    print("VERDICT: %s" % ("EXACT MATCH" if same == nchk and nchk else "MISMATCH"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-transpose", action="store_true")
    ap.add_argument("--validate", type=int, default=0)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--cutoff", type=float, default=7200.0)
    ap.add_argument("--out", default="data/subgraphs/subgraphs_2h")
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.build_transpose:
        build_transpose(); return
    if args.validate:
        validate(args.validate); return
    if not args.run:
        ap.error("nothing to do")

    eids = sorted(d for d in os.listdir(SG) if os.path.isdir(f"{SG}/{d}"))
    if args.limit:
        eids = eids[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    todo = [(e, args.cutoff, args.out) for e in eids
            if not os.path.exists(os.path.join(args.out, e, "meta.json"))]
    print("edges=%d todo=%d cutoff=%.0fs workers=%d" % (len(eids), len(todo), args.cutoff,
                                                        args.workers), flush=True)
    t0 = time.time()
    ok = skip = 0
    with Pool(args.workers, initializer=_load) as p:
        for i, (eid, status, n) in enumerate(p.imap_unordered(_job, todo, chunksize=4)):
            if status == "ok":
                ok += 1
            else:
                skip += 1
            if (i + 1) % 250 == 0:
                el = time.time() - t0
                print("  %d/%d  ok=%d skip=%d  %.0fs  eta %.0fs" % (
                    i + 1, len(todo), ok, skip, el, el / (i + 1) * (len(todo) - i - 1)), flush=True)
    print("DONE ok=%d skip=%d in %.0fs -> %s" % (ok, skip, time.time() - t0, args.out), flush=True)


if __name__ == "__main__":
    main()
