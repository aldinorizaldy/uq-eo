#!/bin/bash
# Run the SHRUG-FM-style UQ evaluation (L1+L2 signals, fusion, risk-coverage/
# AURC/AUDP) on the cached burn-scars ensemble outputs. See
# scripts/evaluate_burn_scars.py. Run AFTER
# slurm/extract_burn_scars_cache.sbatch has written outputs/burn_scars_cache/.
#
# NOT submitted automatically by Claude — review and submit yourself:
#   sbatch slurm/evaluate_burn_scars.sbatch
#
# No --gres=gpu: this is pure NumPy/scikit-learn/FAISS on cached arrays,
# never touches a GPU. Left on partition=gpu-a100 since that's the only
# partition confirmed so far (see NOTES.md).
#SBATCH --job-name=eouq-evaluate-burnscars
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
    python -m scripts.evaluate_burn_scars \
        --cache-dir /outputs/burn_scars_cache \
        --output /outputs/burn_scars_evaluation.json
