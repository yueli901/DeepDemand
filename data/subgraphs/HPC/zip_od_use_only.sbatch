#!/bin/bash
# ====================== CSD3 Slurm job script ======================

# ---------- Slurm account & queueing ----------
#SBATCH -A YJIN-SL3-CPU
#SBATCH -p cclake
#SBATCH -t 01:00:00
#SBATCH -J tgz_od_use_only

# Working dir + absolute log paths
#SBATCH -D /rds/user/yl901/hpc-work/deepdemand
#SBATCH -o /rds/user/yl901/hpc-work/deepdemand/logs/%x_%j.out
#SBATCH -e /rds/user/yl901/hpc-work/deepdemand/logs/%x_%j.err

# Parallel compression with pigz
#SBATCH -c 32
#SBATCH --mem=3G

set -euo pipefail

# Base path and root of the tree that contains the 4000+ dirs
BASE_DIR="/rds/user/yl901/hpc-work/deepdemand/data"
ROOT_REL="subgraphs/subgraphs"              # relative to BASE_DIR
OUTFILE="${BASE_DIR}/od_use_only.tar.gz"    # output archive

echo "[$(date)] Collecting paths and compressing only od_use.csv (preserve dir structure)"
echo "BASE_DIR:  ${BASE_DIR}"
echo "ROOT_REL:  ${ROOT_REL}"
echo "OUTFILE:   ${OUTFILE}"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"

# Ensure pigz available
if ! command -v pigz >/dev/null 2>&1; then
  module load pigz || true
fi
command -v pigz >/dev/null 2>&1 || { echo "pigz not found"; exit 1; }

# Work from BASE_DIR so stored paths are relative (structure preserved on extract)
cd "${BASE_DIR}"

# Find only the target files, write a null-delimited list (safe for any weird chars)
find "${ROOT_REL}" -type f -name 'od_use.csv' -print0 > /tmp/od_use_list.$SLURM_JOB_ID.txt

# Create a tar.gz that stores only those files, with their relative paths intact
tar --use-compress-program="pigz -p ${SLURM_CPUS_PER_TASK} -9" \
    -cf "${OUTFILE}" \
    --null -T /tmp/od_use_list.$SLURM_JOB_ID.txt

# Optional: show result
ls -lh "${OUTFILE}" || true
echo "[$(date)] Done"