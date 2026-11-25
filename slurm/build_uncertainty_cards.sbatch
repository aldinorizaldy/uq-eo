#!/bin/bash
# Build L5 Uncertainty Cards (markdown + JSON) from the evaluation JSONs already in
# outputs/ -- one per experiment (burn scars, flood/Prithvi, flood/U-Net). No
# recomputation; takes seconds. Rerun after any evaluate_* job. See
# scripts/build_uncertainty_cards.py.
#
# NOT submitted automatically by Claude — review and submit yourself:
#   sbatch slurm/build_uncertainty_cards.sbatch
#
# No --gres=gpu: pure NumPy on cached arrays.
#SBATCH --job-name=eouq-cards
#SBATCH --partition=gpu-a100
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
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
    python -m scripts.build_uncertainty_cards \
        --outputs-dir /outputs
