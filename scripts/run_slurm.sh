#!/bin/bash
#SBATCH --job-name=vlm-bench
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# ── User config ────────────────────────────────────────────────────────────────
# Set these to your actual scratch/work paths on the cluster.
SCRATCH=/scratch/$USER
REPO=$HOME/3D-vision-Benchmarking-Spatial-and-State-Reasoning-Skills-of-VLMs-for-Robotics
VENV=$REPO/3dvision
# ──────────────────────────────────────────────────────────────────────────────

# Point HuggingFace cache and outputs to scratch (avoids filling home quota)
export HF_HOME=$SCRATCH/hf_cache
export VLM_OUTPUT_DIR=$SCRATCH/vlm_outputs
export VLM_LOG_DIR=$SCRATCH/vlm_logs

mkdir -p $HF_HOME $VLM_OUTPUT_DIR $VLM_LOG_DIR

source $VENV/bin/activate

cd $REPO/src
python main.py \
    --task ${TASK:-failure_mode} \
    --model ${MODEL:-qwen-3b} \
    --split ${SPLIT:-test} \
    ${LIMIT:+--limit $LIMIT} \
    ${LOCAL_DATA:+--local-data $LOCAL_DATA}
