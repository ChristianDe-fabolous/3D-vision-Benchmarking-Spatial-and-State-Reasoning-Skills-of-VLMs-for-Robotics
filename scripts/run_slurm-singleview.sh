#!/bin/bash
# ── ETH ISG Student Cluster — Action Phase VLM Benchmark ─────────────────────
#
# Usage:
#   sbatch scripts/run_slurm.sh
#
# Override defaults via env vars:
#   MODEL=qwen-7b-int8 sbatch scripts/run_slurm.sh
#   DATASET=data/action_phase_dataset_singleview.jsonl sbatch scripts/run_slurm.sh
#   COT=1 sbatch scripts/run_slurm.sh
#
# Resume a previous run:
#   RUN_ID=my_run sbatch scripts/run_slurm.sh
#
# Smoke test (first 10 questions):
#   SMOKE=1 sbatch scripts/run_slurm.sh
# ─────────────────────────────────────────────────────────────────────────────

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=vlm-bench
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --account=3dv
#SBATCH --gpus=5060ti:1
#SBATCH --time=12:00:00
#SBATCH --mail-user=cdeubel@ethz.ch
#SBATCH --mail-type=END,FAIL

# ── GPU / model reference ─────────────────────────────────────────────────────
# phi-3.5-vision-int8  → 1080ti  (~2GB VRAM)    --gpus=1080ti:1
# phi-3.5-vision       → 1080ti  (~4GB VRAM)    --gpus=1080ti:1
# gemma4-e2b-int8      → 1080ti  (~5GB VRAM)    --gpus=1080ti:1
# qwen3-4b             → 2080ti  (~8GB VRAM)    --gpus=2080ti:1
# gemma4-e4b-int8      → 2080ti  (~8GB VRAM)    --gpus=2080ti:1
# qwen3-8b             → 5060ti  (~16GB VRAM)   --gpus=5060ti:1  ← default
# qwen3-8b-thinking    → 5060ti  (~16GB VRAM)   --gpus=5060ti:1
# gemma4-e2b           → 5060ti  (~10GB VRAM)   --gpus=5060ti:1
# gemma4-e4b           → 5060ti  (~16GB VRAM)   --gpus=5060ti:1
# nvlm-12b-int8        → 5060ti  (~12GB VRAM)   --gpus=5060ti:1
# phi-4-vision         → 5060ti  (~10GB VRAM)   --gpus=5060ti:1
# nvlm-12b             → gb10    (~24GB VRAM)   --gpus=gb10:1
# gemma4-26b-int8      → gb10    (~27GB VRAM)   --gpus=gb10:1
# gemma4-31b-int8      → gb10    (~31GB VRAM)   --gpus=gb10:1
# qwen3-30b            → gb10    (~60GB VRAM)   --gpus=gb10:1
# qwen3-30b-thinking   → gb10    (~60GB VRAM)   --gpus=gb10:1
# gemma4-26b           → gb10    (~54GB VRAM)   --gpus=gb10:1
# gemma4-31b           → gb10    (~62GB VRAM)   --gpus=gb10:1
# ─────────────────────────────────────────────────────────────────────────────

REPO=/work/courses/3dv/team29/3D-vision-Benchmarking-Spatial-and-State-Reasoning-Skills-of-VLMs-for-Robotics

# ── Defaults (override via env vars at sbatch time) ───────────────────────────
MODEL="${MODEL:-qwen3-8b}"
DATASET="${DATASET:-data/action_phase_dataset_singleview.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-4}"

# ── Environment ───────────────────────────────────────────────────────────────
module load cuda/13.0
source ~/.bashrc   # sets HF_HOME, TRANSFORMERS_CACHE, etc.
if [ "$(uname -m)" = "aarch64" ]; then
    source "$REPO/.venv-arm64/bin/activate"
else
    source "$REPO/.venv/bin/activate"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Job info ──────────────────────────────────────────────────────────────────
echo "========================================"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"
echo "Model   : $MODEL"
echo "Dataset : $DATASET"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO"

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python src/main.py \
    --task action_phase \
    --model $MODEL \
    --action-phase-data $DATASET \
    --batch-size $BATCH_SIZE"

[ -n "$ACTION_PHASE_TYPE" ] && CMD="$CMD --action-phase-type $ACTION_PHASE_TYPE"
[ -n "$RUN_ID" ]            && CMD="$CMD --run-id $RUN_ID --resume"
[ -n "$LIMIT" ]             && CMD="$CMD --limit $LIMIT"
[ "${COT:-0}"   = "1" ]     && CMD="$CMD --cot"
[ "${SMOKE:-0}" = "1" ]     && CMD="$CMD --smoke"

echo "CMD: $CMD"
eval $CMD

echo "========================================"
echo "Job $SLURM_JOB_ID done."
