import glob, csv, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model.utils as utils
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
M = [("R2", utils.R_square), ("MGEH", utils.MGEH), ("MGEHhr", utils.MGEH_hourly),
     ("wMAPE", utils.wMAPE), ("MAE", utils.MAE)]
def tm(pc):
    g, p = [], []
    for r in csv.DictReader(open(pc)):
        if r["set"] == "test":
            g.append(float(r["gt"])); p.append(float(r["pred"]))
    G = torch.tensor(g); P = torch.tensor(p)
    return {n: fn(G, P, None).item() for n, fn in M}
cvt = sys.argv[1] if len(sys.argv) > 1 else "random"
folds = range(5) if cvt == "random" else range(1, 10)
print("Ablation (%s CV) -- mean TEST over folds." % cvt)
print("  full (= main DeepDemand): random R2=0.706 MGEH=56.1 | spatial R2=0.647 MGEH=58.0")
for v in ["shared_encoder", "constant_deterrence", "linear_aggregation"]:
    rows = [tm("param/deepdemand/abl_%s_%s%d/pred.csv" % (v, cvt, i))
            for i in folds if os.path.exists("param/deepdemand/abl_%s_%s%d/pred.csv" % (v, cvt, i))]
    if not rows:
        print("  %-20s: no folds yet" % v); continue
    agg = {k: (np.mean([r[k] for r in rows]), np.std([r[k] for r in rows])) for k in rows[0]}
    print("  %-20s (%d): " % (v, len(rows)) + "  ".join("%s=%.4f(%.3f)" % (k, m, s) for k, (m, s) in agg.items()))
