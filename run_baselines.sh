#!/bin/bash
# Benchmark models, on the same folds and the same train/validation/test split as DeepDemand.
# Hyperparameters are tuned on the validation set only.
set -u
mkdir -p eval/logs/baselines

for s in baseline_mean baseline_ols baseline_ridge baseline_gravity \
         baseline_rf baseline_xgb baseline_lightgbm ablation_od_structure baseline_gat; do
  echo ">>> [$(date +%F\ %T)] $s"
  python -u "model/${s}.py" > "eval/logs/baselines/${s}.stdout" 2>&1
done
