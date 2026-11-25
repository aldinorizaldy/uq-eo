#!/bin/bash
# Diagnose the low per-chip F1 on Sen1Floods11: pooled vs per-chip F1, broken
# down by how much water each chip has, for every cache that exists (default
# Prithvi, Prithvi without location coords if extracted, U-Net). See
# scripts/diagnose_sen1floods11_f1.py. Includes the no-coords cache only if
# slurm/extract_sen1floods11_cache_prithvi_nocoords.sbatch has finished.
#
# NOT submitted automatically by Claude — review and submit yourself:
#   sbatch slurm/diagnose_sen1floods11_f1.sbatch
#
# No --gres=gpu: pure NumPy on cached arrays.
#SBATCH --job-name=eouq-diagnose-s1f11
#SBATCH --partition=gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=/home/rizald42/projects/PROTEUS/uq-eo/slurm/logs/%x-%j.out
#SBATCH --error=/home/rizald42/projects/PROTEUS/uq-eo/slurm/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/home/rizald42/projects/PROTEUS/uq-eo"
mkdir -p "$REPO_ROOT/slurm/logs" "$REPO_ROOT/data" "$REPO_ROOT/outputs"

module load apptainer

apptainer exec \
    --pwd /workspace \
    --bind "$REPO_ROOT":/workspace \
    --bind "$REPO_ROOT/data":/data \
    --bind "$REPO_ROOT/outputs":/outputs \
    "$REPO_ROOT/container/eouq.sif" \
    python -m scripts.diagnose_sen1floods11_f1 \
        --output /outputs/sen1floods11_f1_diagnosis.json
