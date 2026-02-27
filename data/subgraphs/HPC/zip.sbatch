#!/bin/bash
# ====================== CSD3 Slurm job script ======================

# ---------- Slurm account & queueing ----------
#SBATCH -A YJIN-SL3-CPU
#SBATCH -p cclake-himem
#SBATCH -t 06:00:00
#SBATCH -J zip2

# Working dir + absolute log paths
#SBATCH -D /rds/user/yl901/hpc-work/deepdemand
#SBATCH -o /rds/user/yl901/hpc-work/deepdemand/logs/%x_%j.out
#SBATCH -e /rds/user/yl901/hpc-work/deepdemand/logs/%x_%j.err

# One core per task
#SBATCH -c 1
#SBATCH --mem=6G

# Path to the folder you want to zip
ZIPFOLDER=/rds/user/yl901/hpc-work/deepdemand/data/subgraphs/subgraphs

# Name of the output archive
ZIPFILE=${ZIPFOLDER}.zip

echo "[$(date)] Zipping $ZIPFOLDER into $ZIPFILE"

# Create the zip (recursive, quiet)
zip -qr "$ZIPFILE" "$ZIPFOLDER"

echo "[$(date)] Done"