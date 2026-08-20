#!/bin/bash
# Spatial nine-fold cross-validation, holding out one region at a time, for geographic transfer.
# Protocol is identical to the random folds except that the test set is a whole region.
set -u
mkdir -p eval/logs/deepdemand
COMMON="USE_OD_CACHE=1 BATCH_EDGES=8 GRAD_ACCUM=4 THREE_WAY=1 VAL_FRAC=0.125 LR=3e-3,1e-3,3e-4 PATIENCE=15"

for f in 1 2 3 4 5 6 7 8 9; do
  name="spatial_$f"
  echo ">>> [$(date +%F\ %T)] $name"
  env $COMMON CV_TYPE=spatial CV_FOLD="$f" RUN_NAME="$name" \
    python -u train.py > "eval/logs/deepdemand/${name}.stdout" 2>&1
done
