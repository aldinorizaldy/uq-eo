#!/bin/bash
# Extract and cache per-chip probs/embeddings/coords/F1 for burn scars using
# the trained decoder ensemble. See scripts/extract_burn_scars_cache.py.
# Run AFTER slurm/train_burn_scars_ensemble.sbatch has produced checkpoints
# in outputs/burn_scars_decoders/ — this job fails fast with a clear error
# if none are found.
#
# NOT submitted automatically by Claude — review and submit yourself:
#   sbatch slurm/extract_burn_scars_cache.sbatch
#
#SBATCH --job-name=eouq-extract-burnscars
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/rizald42/projects/PROTEUS/uq-eo/slurm/logs/%x-%j.out
#SBATCH --error=/home/rizald42/projects/PROTEUS/uq-eo/slurm/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/home/rizald42/projects/PROTEUS/uq-eo"
mkdir -p "$REPO_ROOT/slurm/logs" "$REPO_ROOT/data" "$REPO_ROOT/outputs"

module load apptainer

# -m (not a bare script path): running the file directly only puts scripts/
# on sys.path, not /workspace, so `adapters`/`data` wouldn't import. -m with
# --pwd /workspace puts /workspace itself on the path.
apptainer exec --nv \
    --pwd /workspace \
    --bind "$REPO_ROOT":/workspace \
    --bind "$REPO_ROOT/data":/data \
    --bind "$REPO_ROOT/outputs":/outputs \
    "$REPO_ROOT/container/eouq.sif" \
    python -m scripts.extract_burn_scars_cache \
        --data-root /data/burn_scars \
        --decoder-dir /outputs/burn_scars_decoders \
        --output-dir /outputs/burn_scars_cache
