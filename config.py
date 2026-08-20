DATA = {
    'lsoa_json': "data/node_features/lsoa21_features_normalized.json",
    'population_level': 'lv3', # 18
    'employment_level': 'lv3', # 57
    'households_level': 'lv3', # 33
    'use_population_density': True, # 2
    'use_land_use': True, # 4
    'use_poi': True, # 5
    'use_imd': True, # 2
    'remove_link': False,
    'remove_zero_OD': False,
    'gt': "GT_AADT_8years.json",
}

# ----------------------------------------------------------------------
# Model selection
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
MODEL_NAME = 'deepdemand'

MODEL = {
    'node_hidden': [16, 16],
    'node_out': 16,
    'pair_hidden': [16, 8],
    'chunk_size': 300_000,
    # travel-time deterrence
    't_function': 'mlp',   # 'mlp' or 'logit'
    't_normalize': True,
    't_hidden': [16, 16],
    't_mean': 3600.0,
    't_std': 1000.0,
}

TRAINING = {
    # 'checkpoint': "param/stage1/best_stage_1_lr1e-03.pt",
    'name': 'cv_4',
    "pca": True,
    "pca_components": 64,
    'device': 'cuda',
    'seed': 1,
    'lr': [1e-3],
    'patience': 20,
    'normalize': False,
    'epoch': 100,
    'clip_gradient': 5,

    # normal train-test split
    'train_prop': 0.8,
    # cross validation
    'cv_type': 'random',   # 'random' (k-fold by seed) or 'spatial' (hold out 1 of 9 regions); env CV_TYPE overrides
    'cv_k': 5,
    'cv_fold': 4,          # random: 0..k-1; spatial: 1..9 (region). env CV_FOLD overrides

    'loss_function': 'MSE',
    'eval_metrics': ['MGEH', 'MAE', 'RMSE', 'R_square'],
    'eval_interval': 1000,
    'eval_sample_train': 800,
    'eval_sample_eval': 800,

    # P0 speedup: when True (or env USE_OD_CACHE=1), use the prebuilt OD-index cache
    # (data/subgraphs/od_index_cache) instead of reading od_use.feather every step.
    'use_od_cache': False,
    # Gradient accumulation over N steps per optimizer update (env GRAD_ACCUM overrides).
    # >1 changes the optimization trajectory, so reported numbers will differ.
    'grad_accum': 1,
    # Edges per batched segment-sum forward (env BATCH_EDGES overrides). 1 = off
    # (single-edge). >1 saturates the GPU; tune to fit VRAM. Changes numbers (mini-batch).
    'batch_edges': 1,
    }
