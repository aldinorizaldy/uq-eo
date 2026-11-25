#!/bin/bash
# Conformal prediction under geographic shift on Sen1Floods11: calibrate on other
# continents (val) vs on half the African test chips, evaluate coverage and FNR
# risk control on the other African half, per model. See
# scripts/evaluate_sen1floods11_conformal.py. Needs the Prithvi + U-Net caches.
#
# NOT submitted automatically by Claude — review and submit yourself:
#   sbatch slurm/evaluate_sen1floods11_conformal.sbatch
#
# No --gres=gpu: pure NumPy on cached arrays.
#SBATCH --job-name=eouq-conformal-s1f11
#SBATCH --partition=gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
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
    python -m scripts.evaluate_sen1floods11_conformal \
        --output /outputs/sen1floods11_conformal_evaluation.json
