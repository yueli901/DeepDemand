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

MODEL = {
    'node_hidden': [16, 16],
    'node_out': 16,
    'pair_hidden': [16, 8],
    'chunk_size': 300_000,
    # time function
    't_function': 'mlp', # 'mlp','exp', or 'logit'
    't_normalize': True,
    't_hidden': [16, 16],
    't_mean': 3600.0,
    't_std': 1000.0,
    }

TRAINING = {
    # 'checkpoint': "param/stage1/best_stage_1_lr1e-03.pt",
    'name': 'test',
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
    'cv_k': 5,
    'cv_fold': 0,

    'loss_function': 'MSE',
    'eval_metrics': ['MGEH', 'MAE', 'RMSE', 'R_square'],
    'eval_interval': 1000,
    'eval_sample_train': 800,
    'eval_sample_eval': 800,
    }
