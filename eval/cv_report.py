import re, os, sys
import numpy as np

def parse(log):
    ev=[]; last_t=None
    for l in open(log):
        mt=re.search(r"Train Eval: RMSE: ([\d.]+), MAE: ([\d.]+), MGEH: ([\d.]+), R2: ([\d.eE+-]+)", l)
        mv=re.search(r"Validation Eval: RMSE: ([\d.]+), MAE: ([\d.]+), MGEH: ([\d.]+), R2: ([\d.eE+-]+)", l)
        if mt:
            last_t=dict(rmse=float(mt.group(1)),mae=float(mt.group(2)),mgeh=float(mt.group(3)),r2=float(mt.group(4)))
        elif mv:
            ev.append(dict(train=last_t, val=dict(rmse=float(mv.group(1)),mae=float(mv.group(2)),
                                                  mgeh=float(mv.group(3)),r2=float(mv.group(4)))))
    return ev

def ma(a, w=5):
    return np.convolve(a, np.ones(w)/w, mode="valid") if len(a) >= w else a

names = sys.argv[1:] if len(sys.argv) > 1 else ["cv_0", "cv_1"]
for name in names:
    log = "eval/logs/deepdemand/%s.log" % name
    if not os.path.exists(log):
        print(name, "-> no log"); continue
    ev = parse(log); n = len(ev)
    if n == 0:
        print(name, "-> no evals yet"); continue
    bi = min(range(n), key=lambda i: ev[i]["val"]["mgeh"])
    stopped = any(("EARLY STOP" in l or "Stopping training" in l) for l in open(log))
    t = ev[bi]["train"]; v = ev[bi]["val"]
    print()
    print("===== %s: %d evals (~%dk edges) | best @ eval %d | early-stopped=%s =====" % (name, n, n, bi+1, stopped))
    if t:
        print("  TRAIN @best : R2=%.4f  MGEH=%.3f  MAE=%.1f  RMSE=%.1f" % (t["r2"], t["mgeh"], t["mae"], t["rmse"]))
    print("  VAL   @best : R2=%.4f  MGEH=%.3f  MAE=%.1f  RMSE=%.1f" % (v["r2"], v["mgeh"], v["mae"], v["rmse"]))
    vt = np.array([e["val"]["mgeh"] for e in ev])
    tt = np.array([e["train"]["mgeh"] for e in ev if e["train"]])
    print("  val   MGEH (every 8th): %s  last5=%s" % ([round(x,1) for x in vt[::8]], [round(x,2) for x in vt[-5:]]))
    print("  train MGEH (every 8th): %s  last5=%s" % ([round(x,1) for x in tt[::8]], [round(x,2) for x in tt[-5:]]))
    w = min(20, n)
    vsl = np.polyfit(np.arange(len(ma(vt[-w:]))), ma(vt[-w:]), 1)[0]
    tsl = np.polyfit(np.arange(w), tt[-w:], 1)[0] if len(tt) >= w else float("nan")
    print("  slope over last %d evals: VAL(smoothed)=%+.4f  TRAIN=%+.4f  (neg = still going down)" % (w, vsl, tsl))
    print("  best is %d evals before the end (patience window = 20)" % (n - 1 - bi))
