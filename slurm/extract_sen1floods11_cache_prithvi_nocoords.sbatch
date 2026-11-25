#!/bin/bash
# Diagnostic: re-extract Prithvi-EO-2.0 outputs with location_coords=None (its
# config.yaml says it was fine-tuned with use_metadata: false), to compare against
# the default extraction that passes lon/lat like the HF reference script does.
# Writes to outputs/sen1floods11_cache_prithvi_nocoords/ -- the default cache is
# left untouched. See scripts/extract_sen1floods11_cache.py.
#
# NOT submitted automatically by Claude — review and submit yourself:
#   sbatch slurm/extract_sen1floods11_cache_prithvi_nocoords.sbatch
#
# --mem/--time are guesses: Prithvi-EO-2.0-300M is far bigger than the
# SSL4EO-S12 ViT-S (23M) used for burn scars, and TTA means 5 forward
# passes per chip at inference time.
#SBATCH --job-name=eouq-extract-s1f11-prithvi-nocoords
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
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
        --model prithvi \
        --data-root /data/sen1floods11 \
        --output-dir /outputs/sen1floods11_cache_prithvi_nocoords \
        --no-location-coords
