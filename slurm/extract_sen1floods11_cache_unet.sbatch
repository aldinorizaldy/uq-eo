#!/bin/bash
# Extract and cache per-chip U-Net ensemble outputs for Sen1Floods11.
# See scripts/extract_sen1floods11_cache.py. Run AFTER
# slurm/train_sen1floods11_unet.sbatch has produced checkpoints.
#
# NOT submitted automatically by Claude — review and submit yourself:
#   sbatch slurm/extract_sen1floods11_cache_unet.sbatch
#SBATCH --job-name=eouq-extract-s1f11-unet
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/home/rizald42/projects/PROTEUS/uq-eo/slurm/logs/%x-%j.out
#SBATCH --error=/home/rizald42/projects/PROTEUS/uq-eo/slurm/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/home/rizald42/projects/PROTEUS/uq-eo"
mkdir -p "$REPO_ROOT/slurm/logs" "$REPO_ROOT/data" "$REPO_ROOT/outputs"

module load apptainer

apptainer exec --nv \
    --pwd /workspace \
    --bind "$REPO_ROOT":/workspace \
    --bind "$REPO_ROOT/data":/data \
    --bind "$REPO_ROOT/outputs":/outputs \
    "$REPO_ROOT/container/eouq.sif" \
    python -m scripts.extract_sen1floods11_cache \
        --model unet \
        --data-root /data/sen1floods11 \
        --decoder-dir /outputs/sen1floods11_unet \
        --output-dir /outputs/sen1floods11_cache_unet
