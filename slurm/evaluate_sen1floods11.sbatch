#!/bin/bash
# Run the SHRUG-FM-style UQ evaluation for both Sen1Floods11 adapters
# (Prithvi + U-Net) -- the actual milestone-4 model-agnostic comparison.
# See scripts/evaluate_sen1floods11.py. Run AFTER both
# slurm/extract_sen1floods11_cache_prithvi.sbatch and
# slurm/extract_sen1floods11_cache_unet.sbatch.
#
# NOT submitted automatically by Claude — review and submit yourself:
#   sbatch slurm/evaluate_sen1floods11.sbatch
#
# No --gres=gpu: pure NumPy/scikit-learn/FAISS on cached arrays.
#SBATCH --job-name=eouq-evaluate-s1f11
#SBATCH --partition=gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
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
    python -m scripts.evaluate_sen1floods11 \
        --output /outputs/sen1floods11_evaluation.json
