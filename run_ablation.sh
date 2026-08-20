#!/bin/bash
# Component ablations: three variants over the random and spatial folds, under the same
# protocol as the reported model, which is not re-run here.
set -u
mkdir -p eval/logs/deepdemand
COMMON="USE_OD_CACHE=1 BATCH_EDGES=8 GRAD_ACCUM=4 THREE_WAY=1 VAL_FRAC=0.125 LR=3e-3,1e-3,3e-4 PATIENCE=15"

run () {  # variant cv_type fold
  local v=$1 ct=$2 f=$3
  local name="abl_${v}_${ct}${f}"
  echo ">>> [$(date +%F\ %T)] $name"
  env $COMMON MODEL_VARIANT="$v" CV_TYPE="$ct" CV_FOLD="$f" RUN_NAME="$name" \
    python -u train.py > "eval/logs/deepdemand/${name}.stdout" 2>&1
}

for v in shared_encoder constant_deterrence linear_aggregation; do
  for f in 0 1 2 3 4;         do run "$v" random  "$f"; done
  for f in 1 2 3 4 5 6 7 8 9; do run "$v" spatial "$f"; done
done
