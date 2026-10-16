#!/bin/bash
# Train the burn-scars decoder ensemble (bootstrap ensemble over the frozen
# SSL4EO-S12 ViT-S/16 backbone). See scripts/train_burn_scars_ensemble.py.
#
# NOT submitted automatically by Claude — review and submit yourself:
#   sbatch slurm/train_burn_scars_ensemble.sbatch
#
# --time is a guess (max cluster wall-clock limit still unconfirmed, see
# NOTES.md) — 5 members x 20 epochs x 485 bootstrap-resampled chips on a
# frozen backbone should comfortably fit, but raise it if it times out.
#SBATCH --job-name=eouq-train-burnscars
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/rizald42/projects/PROTEUS/uq-eo/slurm/logs/%x-%j.out
#SBATCH --error=/home/rizald42/projects/PROTEUS/uq-eo/slurm/logs/%x-%j.err

set -euo pipefail

# Hardcoded, not derived from BASH_SOURCE[0]: Slurm stages the submitted
# script into a per-job spool dir on the compute node at runtime, so
# BASH_SOURCE[0]-based path detection breaks (see NOTES.md / smoke_test.sbatch).
REPO_ROOT="/home/rizald42/projects/PROTEUS/uq-eo"
mkdir -p "$REPO_ROOT/slurm/logs" "$REPO_ROOT/data" "$REPO_ROOT/outputs"

module load apptainer

apptainer exec --nv \
    --pwd /workspace \
    --bind "$REPO_ROOT":/workspace \
    --bind "$REPO_ROOT/data":/data \
    --bind "$REPO_ROOT/outputs":/outputs \
    "$REPO_ROOT/container/eouq.sif" \
    python scripts/train_burn_scars_ensemble.py \
        --data-root /data/burn_scars \
        --output-dir /outputs/burn_scars_decoders
