import re, glob, os
import numpy as np
REGION = {1:"NorthEast",2:"NorthWest",3:"Yorkshire",4:"EastMidlands",5:"WestMidlands",
          6:"EastEngland",7:"London",8:"SouthEast",9:"SouthWest"}
def finals(log):
    m = {}
    for l in open(log):
        mm = re.search(r"FINAL (TRAIN|TEST) Eval: RMSE: ([\d.]+), MAE: ([\d.]+), MGEH: ([\d.]+), R2: ([\d.eE+-]+)", l)
        if mm: m[mm.group(1)] = dict(rmse=float(mm.group(2)),mae=float(mm.group(3)),mgeh=float(mm.group(4)),r2=float(mm.group(5)))
    return m
print("Spatial nine-fold cross-validation (test = held-out region)")
print("submitted (2-way): TRAIN R2=0.736 MGEH=51.09 | TEST R2=0.665 MGEH=56.43 MAE=7669")
print("-"*82)
TE={k:[] for k in ('r2','mgeh','mae')}; TR={k:[] for k in ('r2','mgeh','mae')}
for f in range(1,10):
    log="eval/logs/deepdemand/spatial%d_3way_staged.log"%f
    if not os.path.exists(log): print("region %d %-12s (no log yet)"%(f,REGION[f])); continue
    m=finals(log); tr=m.get('TRAIN'); te=m.get('TEST')
    if tr and te:
        print("region %d %-12s  TEST R2=%.4f MGEH=%.2f MAE=%.0f  | TRAIN R2=%.4f"
              %(f,REGION[f],te['r2'],te['mgeh'],te['mae'],tr['r2']))
        for k in TE: TE[k].append(te[k]); TR[k].append(tr[k])
    else:
        print("region %d %-12s  (in progress)"%(f,REGION[f]))
if TE['r2']:
    n=len(TE['r2'])
    print("-"*82)
    print("MEAN (%d regions)  TEST R2=%.4f(%.3f) MGEH=%.2f MAE=%.0f | TRAIN R2=%.4f"
          %(n,np.mean(TE['r2']),np.std(TE['r2']),np.mean(TE['mgeh']),np.mean(TE['mae']),np.mean(TR['r2'])))
