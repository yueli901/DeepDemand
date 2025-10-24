DATA = {
    'population_level': 'lv3', # 18
    'employment_level': 'lv3', # 57
    'households_level': 'lv3', # 33 
    'use_population_density': True, # 2
    'use_land_use': True, # 4
    'use_poi': True, # 5
    'use_imd': True, # 2
}

MODEL = {
    'node_hidden': [16, 16],
    'node_out': 16,
    'pair_hidden': [16, 8],
    'chunk_size': 100_000,
    # time function
    't_normalize': True,
    't_hidden': [32, 32],
    't_mean': 3600.0,
    't_std': 1000.0,
    }

TRAINING = {
    # 'checkpoint': "param/baseline1/epoch49_20251013-160503.pt",
    "pca": True,
    "pca_components": 64,
    'device': 'cuda',
    'seed': 5,
    'lr': [1e-3, 1e-4, 1e-5],
    'normalize': False,
    'epoch': 100,
    'clip_gradient': 5,
    'train_prop': 0.8,
    'loss_function': 'MSE',
    'eval_metrics': ['MGEH', 'MAE', 'RMSE', 'R_square'],
    'eval_interval': 1000,
    'eval_sample_train': 800,
    'eval_sample_eval': 800,
    }
