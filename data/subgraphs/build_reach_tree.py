#!/usr/bin/env python3
"""Reachability tree for every centroid node at a given cutoff.

For each origin, records the shortest travel time to every other centroid node reachable within
the cutoff. A catchment radius of tau requires a cutoff of 2*tau, since a screened OD pair
satisfies t_O + t_e + t_D = t_OD with both sides bounded by tau.

Output is packed: origins[i] owns dests[indptr[i]:indptr[i+1]] as indices into the sorted
centroid array, with float64 times. Workers are forked after the graph is loaded so that it is
shared between them.

  python data/subgraphs/build_reach_tree.py --cutoff 7200 --out data/subgraphs/reach_2h
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

CSR = "data/highway_network/uk_driving_graph_csr.npz"
_G = {}


def _init():
    z = np.load(CSR)
    nodes = z["nodes"]
    _G["nodes"] = nodes
    _G["M"] = csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(len(nodes), len(nodes)))
    cent = np.array(sorted(int(k) for k in json.load(open("data/node_features/node_to_lsoa.json"))),
                    dtype=np.int64)
    ci = np.searchsorted(nodes, cent)
    _G["cent"] = cent
    _G["cent_idx"] = ci


def _chunk(job):
    lo, hi, cutoff = job
    M, nodes, cent, ci = _G["M"], _G["nodes"], _G["cent"], _G["cent_idx"]
    counts, dparts, tparts = [], [], []
    for k in range(lo, hi):
        oi = ci[k]
        d = dijkstra(M, directed=True, indices=oi, limit=cutoff + 1e-6)
        t = d[ci]
        keep = np.isfinite(t) & (t <= cutoff)
        keep[k] = False                      # exclude self, matching the original
        idx = np.flatnonzero(keep)
        counts.append(len(idx))
        dparts.append(idx.astype(np.int32))
        tparts.append(t[idx])
    return (lo, hi, np.array(counts, np.int64),
            np.concatenate(dparts) if dparts else np.empty(0, np.int32),
            np.concatenate(tparts) if tparts else np.empty(0, np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, required=True, help="seconds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N origins")
    ap.add_argument("--chunk", type=int, default=200)
    args = ap.parse_args()

    _init()
    n_cent = len(_G["cent"]) if not args.limit else min(args.limit, len(_G["cent"]))
    print("origins=%d cutoff=%.0fs workers=%d" % (n_cent, args.cutoff, args.workers), flush=True)

    jobs = [(lo, min(lo + args.chunk, n_cent), args.cutoff)
            for lo in range(0, n_cent, args.chunk)]
    os.makedirs(args.out, exist_ok=True)
    counts = np.zeros(n_cent, np.int64)

    # Chunks are written straight to disk; accumulating every pair in the parent and then
    # concatenating would need several times the peak memory.
    t0 = time.time()
    done = 0
    with Pool(args.workers, initializer=_init) as p:
        for j, (lo, hi, c, dd, tt) in enumerate(p.imap(_chunk, jobs)):
            counts[lo:hi] = c
            np.savez(f"{args.out}/chunk_{lo:06d}.npz", lo=lo, hi=hi, counts=c, dests=dd, times=tt)
            done += hi - lo
            if (j + 1) % 20 == 0:
                el = time.time() - t0
                print("  %d/%d origins  %.0fs elapsed  eta %.0fs  pairs=%d" % (
                    done, n_cent, el, el / done * (n_cent - done), counts.sum()), flush=True)

    indptr = np.zeros(n_cent + 1, np.int64)
    np.cumsum(counts, out=indptr[1:])
    np.savez(f"{args.out}/index.npz", origins=_G["cent"][:n_cent], cent=_G["cent"],
             indptr=indptr, counts=counts, cutoff=np.float64(args.cutoff),
             chunk=np.int64(args.chunk))
    print("DONE %d origins, %d pairs, %.0fs -> %s/" % (
        n_cent, int(counts.sum()), time.time() - t0, args.out), flush=True)


if __name__ == "__main__":
    main()
