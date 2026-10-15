#!/bin/bash
# Run the eouq unit test suite (synthetic data, CPU-only) inside the container.
#
# NOT submitted automatically by Claude — review and submit yourself:
#   sbatch slurm/run_tests.sbatch
#
# No --gres=gpu: these tests never touch a GPU. Left on partition=gpu-a100
# since that's the only partition confirmed so far (see NOTES.md) — move to
# a CPU partition if/when one is confirmed.
#SBATCH --job-name=eouq-tests
#SBATCH --partition=gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_ROOT/slurm/logs" "$REPO_ROOT/data" "$REPO_ROOT/outputs"

module load apptainer

apptainer exec \
    --pwd /workspace \
    --bind "$REPO_ROOT":/workspace \
    --bind "$REPO_ROOT/data":/data \
    --bind "$REPO_ROOT/outputs":/outputs \
    "$REPO_ROOT/container/eouq.sif" \
    python -m pytest tests -v
