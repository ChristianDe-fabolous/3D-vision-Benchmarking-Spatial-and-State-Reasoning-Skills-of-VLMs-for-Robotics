#!/bin/bash
# ── ETH ISG Student Cluster — VLM Benchmark Job ───────────────────────────────
#
# Usage:
#   sbatch scripts/run_slurm.sh
#
# Override defaults via env vars:
#   TASK=failure_mode MODEL=qwen-7b-int8 GPU=2080ti sbatch scripts/run_slurm.sh
#
# Resume a previous run (same RUN_ID picks up where it left off):
#   RUN_ID=my_run RESUME=1 sbatch scripts/run_slurm.sh
# ──────────────────────────────────────────────────────────────────────────────

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=vlm-bench
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --account=3dvision          # ← your course/project tag here
#SBATCH --time=24:00                # HH:MM — max 7 days (168:00); gb10 max is 24:00
#SBATCH --gpus=2080ti:1             # 11GB VRAM — fits qwen-7b-int8 / qwen-3b
#SBATCH --cpus-per-task=4           # 2080ti: max 4 cores
#SBATCH --mem=36G                   # 2080ti: max 36GB RAM

# ── GPU / model combinations ──────────────────────────────────────────────────
# qwen-3b          → 1080ti  (~6GB VRAM, bf16)   2 cores, 24GB RAM, max 7 days
# qwen-7b-int8     → 2080ti  (~8GB VRAM, int8)   4 cores, 36GB RAM, max 7 days
# qwen-7b          → 5060ti  (~14GB VRAM, bf16)  3 cores, 24GB RAM, max 7 days
# qwen-32b-int8    → gb10    (~32GB VRAM, int8)  20 cores, 116GB RAM, max 24h
#
# Uncomment the block matching your model and comment out the lines above:
##SBATCH --gpus=1080ti:1            # qwen-3b
##SBATCH --cpus-per-task=2
##SBATCH --mem=24G
#
##SBATCH --gpus=5060ti:1            # qwen-7b full precision
##SBATCH --cpus-per-task=3
##SBATCH --mem=24G
#
##SBATCH --gpus=gb10:1              # qwen-32b-int8 — 128GB shared VRAM, max 24:00
##SBATCH --cpus-per-task=20
##SBATCH --mem=116G
# ─────────────────────────────────────────────────────────────────────────────

# ── User config ───────────────────────────────────────────────────────────────
SCRATCH=/work/scratch/$USER
REPO=$HOME/3D-vision-Benchmarking-Spatial-and-State-Reasoning-Skills-of-VLMs-for-Robotics
VENV=$REPO/3dvision

# Job parameters (override via env vars)
TASK=${TASK:-failure_mode}          # failure_mode | multiview
MODEL=${MODEL:-qwen-7b-int8}        # qwen-3b | qwen-7b | qwen-7b-int8 | qwen3-2b
PROMPT=${PROMPT:-paper}             # default | paper | paper_cot | test
SPLIT=${SPLIT:-test}
# ─────────────────────────────────────────────────────────────────────────────

# ── Environment setup ─────────────────────────────────────────────────────────
. /etc/profile.d/modules.sh
module add cuda/12.9

# HF cache → scratch (keeps large downloads off the 20GB home quota)
# WARNING: scratch retention depends on usage size:
#   <10GB → 7 days | 10-50GB → 2 days | >50GB → 1 day
# Model + dataset will likely exceed 50GB → 1-day retention.
# Re-download happens automatically on next job if cache is gone.
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets

# Outputs → persistent scratch (copy to /work/users or home when done)
export VLM_OUTPUT_DIR=$SCRATCH/vlm_outputs
export VLM_LOG_DIR=$SCRATCH/vlm_logs

mkdir -p $HF_HOME $HF_DATASETS_CACHE $VLM_OUTPUT_DIR $VLM_LOG_DIR

# ── Activate venv ─────────────────────────────────────────────────────────────
source $VENV/bin/activate

# ── Print job info ────────────────────────────────────────────────────────────
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Task:      $TASK | Model: $MODEL | Prompt: $PROMPT"
echo "Scratch:   $SCRATCH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Build python command ──────────────────────────────────────────────────────
CMD="python src/main.py \
    --task $TASK \
    --model $MODEL \
    --prompt $PROMPT \
    --split $SPLIT"

# Optional limit (e.g. LIMIT=100 sbatch ...)
[ -n "$LIMIT" ] && CMD="$CMD --limit $LIMIT"

# Resume: pass RUN_ID + RESUME=1 to continue a previous run
[ -n "$RUN_ID" ] && CMD="$CMD --run-id $RUN_ID"
[ -n "$RESUME" ] && CMD="$CMD --resume"

# HF token (set in env or hardcode here)
[ -n "$HF_TOKEN" ] && CMD="$CMD --hf-token $HF_TOKEN"

# ── Run ───────────────────────────────────────────────────────────────────────
cd $REPO
echo "Running: $CMD"
eval $CMD

echo "Done. Outputs in $VLM_OUTPUT_DIR"
