"""Sensitivity of the model to the multi-year aggregation of the ground truth.

Generates alternative long-term AADT targets from the per-year sensor aggregates, each keyed to
exactly the same edge set so that the cross-validation folds are identical across targets, and
reports how far each target moves relative to the default.
"""
import json

import numpy as np

D = "data/traffic_volume"
META = f"{D}/GT_AADT_8years_meta.json"
COVID = {2020, 2021}

med = json.load(open(f"{D}/GT_AADT_8years.json"))   # reference edge set + fallback values
meta = json.load(open(META))

mean_t, excl_t = {}, {}
for eid, mval in med.items():
    mval = float(mval)
    gpy = {int(y): float(v) for y, v in (meta.get(eid, {}).get("gt_per_year") or {}).items()}
    if gpy:
        vals = list(gpy.values())
        nc = [v for y, v in gpy.items() if y not in COVID]
        mean_t[eid] = float(np.mean(vals))
        excl_t[eid] = float(np.median(nc)) if nc else mval     # fallback: covid-only edge
    else:
        mean_t[eid] = excl_t[eid] = mval                        # fallback: edge absent in meta

for name, obj in [("GT_AADT_mean.json", mean_t),
                  ("GT_AADT_excl_covid.json", excl_t)]:
    assert set(obj.keys()) == set(med.keys()), f"{name} edge set != median"
    json.dump(obj, open(f"{D}/{name}", "w"))
    print(f"wrote {name}: {len(obj)} edges (== median edge set)")

print("\nMedian |relative shift| of each target vs the paper's median target:")
for name, obj in [("mean", mean_t), ("excl_covid", excl_t)]:
    diffs = [abs(obj[e] - float(med[e])) / (float(med[e]) + 1e-9) for e in obj]
    print(f"  {name:10s}: {100 * np.median(diffs):.1f} %")
