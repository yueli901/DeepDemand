DATA = {
    'population_level': 'population_lv1',
    'employment_level': 'employment_lv1',
}

MODEL = {
    'node_hidden': [64, 64],
    'node_out': 64,
    'pair_hidden': [64, 32],
    'pair_batch_size': 16,
    'edge_feature_dim': 22,
    'gru_hidden': 128,
    'gru_dropout': 0.1
    }

TRAINING = {
    'device': 'cuda',
    'seed': 1,
    'lr': 1e-3,
    'epoch': 100,
    'clip_gradient': 5,
    'train_prop': 0.8,
    'loss_function': 'MSE_Z',
    'eval_metrics': ['MGEH', 'MAE'],
    'eval_interval': 100,
    'eval_sample_train': 100,
    'eval_sample_eval': 100,
    }
