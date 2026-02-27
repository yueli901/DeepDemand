#!/bin/bash
# ====================== CSD3 Slurm job script (dummy array) ======================

# ---------- Slurm account & queueing ----------
#SBATCH -A YJIN-SL3-CPU
#SBATCH -p cclake-himem
#SBATCH -t 06:00:00
#SBATCH -J final1

# Working dir + absolute log paths (logs dir must exist before sbatch)
#SBATCH -D /rds/user/yl901/hpc-work/deepdemand/data/subgraphs
#SBATCH -o /rds/user/yl901/hpc-work/deepdemand/logs/%x_%A_%a.out
#SBATCH -e /rds/user/yl901/hpc-work/deepdemand/logs/%x_%A_%a.err

# One core per task
#SBATCH -c 1
#SBATCH --mem=19G

# Exactly 5 tasks: IDs 0..4; run up to 5 at once
#SBATCH --array=0-5087%500

set -euo pipefail

# ---------- Python environment ----------
source /rds/user/yl901/hpc-work/deepdemand/myenv/bin/activate
PY="$(which python)"
export PYTHONUNBUFFERED=1

# Threaded libs respect -c
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---------- User variables ----------
ROOT="/rds/user/yl901/hpc-work/deepdemand/data/subgraphs"
LIST="$ROOT/ego_dirs.txt"                 # must contain exactly 5 lines for this run

# Guards
[[ -s "$LIST" ]] || { echo "[ERROR] Missing or empty $LIST"; exit 4; }
[[ -x "$PY"   ]] || { echo "[ERROR] Python not found at: $PY"; exit 5; }

# ---------- Pick the work item (array) ----------
EGO_DIR="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST" || true)"
[[ -n "$EGO_DIR"    ]] || { echo "[ERROR] No entry for index $SLURM_ARRAY_TASK_ID"; exit 2; }
[[ -d "$EGO_DIR"    ]] || { echo "[ERROR] Not a directory: $EGO_DIR"; exit 3; }

# ---------- Run payload ----------
echo "[$(date)] (Job $SLURM_JOB_ID / Task $SLURM_ARRAY_TASK_ID) Start → ${EGO_DIR}"
echo "Partition: ${SLURM_JOB_PARTITION}  Account: ${SLURM_JOB_ACCOUNT}  Cores: ${SLURM_CPUS_PER_TASK}  Mem: ${SLURM_MEM_PER_NODE:-N/A}"

"${PY}" "${ROOT}/process_one_edge.py" --ego-dir "${EGO_DIR}"
RC=$?

echo "[$(date)] (Job $SLURM_JOB_ID / Task $SLURM_ARRAY_TASK_ID) Done (rc=${RC})"
exit "${RC}"