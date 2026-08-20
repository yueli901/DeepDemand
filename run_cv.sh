#!/bin/bash
# Random five-fold cross-validation.
# Each fold: 70/10/20 train/validation/test, staged learning rate, early stopping on the
# validation set. Metrics and per-edge predictions land under eval/logs/deepdemand/ and
# param/deepdemand/.
set -u
mkdir -p eval/logs/deepdemand
COMMON="USE_OD_CACHE=1 BATCH_EDGES=8 GRAD_ACCUM=4 THREE_WAY=1 VAL_FRAC=0.125 LR=3e-3,1e-3,3e-4 PATIENCE=15"

for f in 0 1 2 3 4; do
  name="cv_$f"
  echo ">>> [$(date +%F\ %T)] $name"
  env $COMMON CV_TYPE=random CV_FOLD="$f" RUN_NAME="$name" \
    python -u train.py > "eval/logs/deepdemand/${name}.stdout" 2>&1
done
