#!/bin/bash
# L0 sensor-noise experiment, CPU stage: degradation sweep, clean-fitted
# fusion under noise, and noise sensitivity as a signal, for both datasets.
# See scripts/evaluate_sensor_noise.py. Run AFTER all three
# slurm/extract_sensor_noise.sbatch jobs have finished.
#
# NOT submitted automatically by Claude — review and submit yourself:
#   sbatch slurm/evaluate_sensor_noise.sbatch
#SBATCH --job-name=eouq-eval-sensor-noise
#SBATCH --partition=gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/rizald42/projects/PROTEUS/uq-eo/slurm/logs/%x-%j.out
#SBATCH --error=/home/rizald42/projects/PROTEUS/uq-eo/slurm/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/home/rizald42/projects/PROTEUS/uq-eo"
mkdir -p "$REPO_ROOT/slurm/logs" "$REPO_ROOT/outputs"

module load apptainer

for DATASET in burn_scars sen1floods11; do
    apptainer exec \
        --pwd /workspace \
        --bind "$REPO_ROOT":/workspace \
        --bind "$REPO_ROOT/outputs":/outputs \
        "$REPO_ROOT/container/eouq.sif" \
        python -m scripts.evaluate_sensor_noise --dataset "$DATASET"
done
