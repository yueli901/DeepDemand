#!/usr/bin/env python3
"""Compact CSR representation of the UK driving graph.

Converts the networkx MultiDiGraph into flat index and weight arrays so that shortest-path work
can run in parallel processes. Edges without a positive `travel_time` are dropped, parallel edges
between the same node pair collapse to their minimum, and weights are kept in float64.

Writes data/highway_network/uk_driving_graph_csr.npz (nodes, indptr, indices, data).
"""
import gc
import pickle
import time

import numpy as np

GP = "data/highway_network/uk_driving_graph_simplified_travel_time_added.gpickle"
OUT = "data/highway_network/uk_driving_graph_csr.npz"


def main():
    t0 = time.time()
    G = pickle.load(open(GP, "rb"))
    n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
    print("loaded %.0fs |V|=%d |E|=%d" % (time.time() - t0, n_nodes, n_edges), flush=True)

    nodes = np.fromiter(G.nodes(), dtype=np.int64, count=n_nodes)
    nodes.sort()

    src = np.empty(n_edges, np.int64)
    dst = np.empty(n_edges, np.int64)
    w = np.empty(n_edges, np.float64)
    i = 0
    for a, b, data in G.edges(data=True):
        tt = data.get("travel_time")
        if tt is None:
            continue
        tt = float(tt)
        if tt <= 0.0:
            continue
        src[i] = a; dst[i] = b; w[i] = tt; i += 1
    src = src[:i]; dst = dst[:i]; w = w[:i]
    print("kept %d/%d edges with positive travel_time" % (i, n_edges), flush=True)

    del G
    gc.collect()

    si = np.searchsorted(nodes, src).astype(np.int32)
    di = np.searchsorted(nodes, dst).astype(np.int32)
    del src, dst
    gc.collect()

    order = np.lexsort((di, si))
    si = si[order]; di = di[order]; w = w[order]
    del order
    gc.collect()

    new = np.empty(len(si), bool)
    new[0] = True
    np.logical_or(si[1:] != si[:-1], di[1:] != di[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    wmin = np.minimum.reduceat(w, starts)
    si = si[starts]; di = di[starts]
    print("collapsed parallel edges: %d -> %d unique (u,v)" % (len(w), len(si)), flush=True)

    indptr = np.zeros(len(nodes) + 1, np.int64)
    np.cumsum(np.bincount(si, minlength=len(nodes)), out=indptr[1:])

    np.savez(OUT, nodes=nodes, indptr=indptr, indices=di, data=wmin)
    print("saved %s  nnz=%d  arrays=%.0f MB  %.0fs total" % (
        OUT, len(di), (nodes.nbytes + indptr.nbytes + di.nbytes + wmin.nbytes) / 1e6,
        time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
