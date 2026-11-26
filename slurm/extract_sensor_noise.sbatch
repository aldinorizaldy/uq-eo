#!/bin/bash
# L0 sensor-noise experiment, GPU stage: re-run one model on noise-perturbed
# inputs (degradation sweep on test + 5-draw sensitivity signal on all
# splits). See scripts/extract_sensor_noise.py. Needs the model's existing
# clean cache. Takes dataset and model as arguments:
#
# NOT submitted automatically by Claude — review and submit yourself:
#   sbatch slurm/extract_sensor_noise.sbatch burn_scars ssl4eo
#   sbatch slurm/extract_sensor_noise.sbatch sen1floods11 prithvi
#   sbatch slurm/extract_sensor_noise.sbatch sen1floods11 unet
#SBATCH --job-name=eouq-sensor-noise
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/rizald42/projects/PROTEUS/uq-eo/slurm/logs/%x-%j.out
#SBATCH --error=/home/rizald42/projects/PROTEUS/uq-eo/slurm/logs/%x-%j.err

set -euo pipefail

DATASET="${1:?usage: sbatch slurm/extract_sensor_noise.sbatch <burn_scars|sen1floods11> <ssl4eo|prithvi|unet>}"
MODEL="${2:?usage: sbatch slurm/extract_sensor_noise.sbatch <burn_scars|sen1floods11> <ssl4eo|prithvi|unet>}"
echo "sensor-noise extraction: dataset=$DATASET model=$MODEL"

REPO_ROOT="/home/rizald42/projects/PROTEUS/uq-eo"
mkdir -p "$REPO_ROOT/slurm/logs" "$REPO_ROOT/data" "$REPO_ROOT/outputs"

module load apptainer

apptainer exec --nv \
    --pwd /workspace \
    --bind "$REPO_ROOT":/workspace \
    --bind "$REPO_ROOT/data":/data \
    --bind "$REPO_ROOT/outputs":/outputs \
    "$REPO_ROOT/container/eouq.sif" \
    python -m scripts.extract_sensor_noise --dataset "$DATASET" --model "$MODEL"
