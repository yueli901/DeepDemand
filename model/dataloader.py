import torch
import os
import numpy as np
import random
import pandas as pd
import json

from model.utils import ScalerMinMax
from config import DATA, TRAINING

np.random.seed(TRAINING['seed'])
torch.manual_seed(TRAINING['seed'])
random.seed(TRAINING['seed'])

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

import torch

def get_lsoa_vector(rec):
    """
    Build one LSOA node's feature vector (all already min–max normalized upstream).

    Feature order (concatenate in this exact order):
      1) population[DATA['population_level']]           -> list
      2) employment[DATA['employment_level']]           -> list
      3) households[DATA['use_household']]              -> list (if a level string, e.g., 'lv3')
      4) area_popdensity                                -> [area_norm, popdens_norm] (if True)
      5) land_use                                       -> [commercial, industrial, residential, retail] (if True)
      6) poi                                            -> [education, food, health, retail, transport] (if True)
      7) imd                                            -> [income_norm, employment_norm] (if True)

    Parameters
    ----------
    rec : dict
        One LSOA record from the merged JSON, e.g.
        {
          "population": {"lv1":[...], "lv2":[...], "lv3":[...]},
          "employment": {"lv1":[...], "lv2":[...], "lv3":[...]},
          "households": {"lv1":[...], "lv2":[...], "lv3":[...]},
          "area_popdensity": [...],
          "land_use": [...],
          "poi": [...],
          "imd": [...]
        }

    DATA : dict
        Config like:
        {
          'population_level': 'population_lv3',
          'employment_level': 'employment_lv3',
          'use_household': 'lv3',         # or 'lv1'/'lv2' or False/None to disable
          'use_population_density': True,
          'use_land_use': True,
          'use_poi': True,
          'use_imd': True,
        }

    Returns
    -------
    torch.FloatTensor shape (D,)
    """

    parts = []

    # 1) population
    pop_level_key = DATA.get('population_level', 'lv3')
    if isinstance(rec.get('population'), dict):
        parts.extend(float(x) for x in rec['population'].get(pop_level_key, []))

    # 2) employment
    emp_level_key = DATA.get('employment_level', 'lv3')
    if isinstance(rec.get('employment'), dict):
        parts.extend(float(x) for x in rec['employment'].get(emp_level_key, []))

    # 3) households (optional level string)
    hh_level_key = DATA.get('households_level', 'lv3')
    if hh_level_key and isinstance(rec.get('households'), dict):
        parts.extend(float(x) for x in rec['households'].get(hh_level_key, []))

    # 4) area + pop density (optional)
    if DATA.get('use_population_density', False):
        parts.extend(float(x) for x in rec.get('area_popdensity', []) or [])

    # 5) land use (optional)
    if DATA.get('use_land_use', False):
        parts.extend(float(x) for x in rec.get('land_use', []) or [])

    # 6) POI (optional)
    if DATA.get('use_poi', False):
        parts.extend(float(x) for x in rec.get('poi', []) or [])

    # 7) IMD (optional)
    if DATA.get('use_imd', False):
        parts.extend(float(x) for x in rec.get('imd', []) or [])

    return torch.tensor(parts, dtype=torch.float32)

def load_seq_npy(npy_path):
    """
    Load a path feature npy for a node. Returns 2D array [steps, feat_dim].
    Assumes features are already normalized (one-hot + derived), as produced earlier.
    """
    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Bad array shape in {npy_path}: {arr.shape}")
    return arr

def make_pair_sequence(arr_O, arr_D):
    """
    Build the OD sequence for GRU:
      - Reverse O sequence to get forward travel
      - D sequence is forward; drop its first row to avoid duplicating the ego edge
      - Concatenate [O_rev ; D_fwd[1:]]
    Return float32 tensor [T, F]
    """
    O_rev = arr_O[::-1, :]
    D_fwd = arr_D
    if len(D_fwd) > 0:
        D_fwd = D_fwd[1:, :]  # drop first step to avoid ego-edge duplication
    seq = np.concatenate([O_rev, D_fwd], axis=0)
    return torch.tensor(seq, dtype=torch.float32)


def load_gt():
    """
    Load GT traffic volume from a simple JSON mapping {edge_id: volume}.
    TODO: Intersect with edges that have >0 valid OD pairs per od_pairs_stats.json (current)
    Normalize using Scaler and return:
      - edge_to_gt: {edge_id: normalized_value}
      - scaler: fitted Scaler on raw volumes
    """
    # Load JSON
    with open("data/traffic_volume/GT_2022_car.json", "r") as f:
        gt_data = json.load(f)
    with open("data/subgraphs/od_pairs_stats.json", "r") as f:
        stats_raw = json.load(f)

    valid_ids = {eid for eid, cnt in stats_raw.items() if int(cnt) > 0}
    filtered_items = [(eid, float(vol)) for eid, vol in gt_data.items() if eid in valid_ids]

    print(f"Number of valid edges: {len(filtered_items)}")
    edge_ids, volumes = zip(*filtered_items)

    # Torch tensor + normalization
    traffic_volume_tensor = torch.tensor(volumes, dtype=torch.float32, device=TRAINING['device'])
    if TRAINING['normalize']:
        scaler = ScalerMinMax(traffic_volume_tensor)
        traffic_volume_tensor = scaler.transform(traffic_volume_tensor)
    else:
        scaler = None

    # Build mapping {edge_id: normalized_value}
    edge_to_gt = {
        eid: float(val) for eid, val in zip(edge_ids, traffic_volume_tensor.cpu().numpy())
    }

    return edge_to_gt, scaler
