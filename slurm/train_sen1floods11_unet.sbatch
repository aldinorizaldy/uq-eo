#!/bin/bash
# Train the Sen1Floods11 U-Net standard ensemble (non-FM baseline for the
# milestone-4 model-agnostic check). See scripts/train_sen1floods11_unet.py.
#
# NOT submitted automatically by Claude — review and submit yourself:
#   sbatch slurm/train_sen1floods11_unet.sbatch
#
# --time is a guess: 512x512 chips are ~5x the pixel count of burn scars'
# 224x224 working size, which took ~22min for 5 members there.
#SBATCH --job-name=eouq-train-s1f11-unet
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
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
    python -m scripts.train_sen1floods11_unet \
        --data-root /data/sen1floods11 \
        --output-dir /outputs/sen1floods11_unet
