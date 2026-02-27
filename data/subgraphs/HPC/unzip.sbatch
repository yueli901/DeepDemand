#!/bin/bash
# ====================== CSD3 Slurm job script ======================

# ---------- Slurm account & queueing ----------
#SBATCH -A YJIN-SL3-CPU
#SBATCH -p cclake-himem
#SBATCH -t 06:00:00
#SBATCH -J unzip2

# Working dir + absolute log paths
#SBATCH -D /rds/user/yl901/hpc-work/deepdemand
#SBATCH -o /rds/user/yl901/hpc-work/deepdemand/logs/%x_%j.out
#SBATCH -e /rds/user/yl901/hpc-work/deepdemand/logs/%x_%j.err

# One core per task
#SBATCH -c 1
#SBATCH --mem=6G

# Path to the archive
ZIPFILE=/rds/user/yl901/hpc-work/deepdemand/data/subgraphs/subgraphs.zip

# Get the directory where the zip lives
ZIPDIR=$(dirname "$ZIPFILE")

echo "[$(date)] Unzipping $ZIPFILE into $ZIPDIR"

# Disable zip bomb detection for large archives
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
unzip -q "$ZIPFILE" -d "$ZIPDIR"

echo "[$(date)] Done"