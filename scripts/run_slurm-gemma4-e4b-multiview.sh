#!/bin/bash
# Gemma4-E4B (8B BF16) on the multiview consistency dataset — 5060ti (~16GB VRAM), batch_size=1 (fills GPU)
#SBATCH --job-name=gemma4-e4b-multiview
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --account=pmlr_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=12:00:00
#SBATCH --mail-user=cdeubel@ethz.ch
#SBATCH --mail-type=END,FAIL

REPO=/work/courses/3dv/team29/3D-vision-Benchmarking-Spatial-and-State-Reasoning-Skills-of-VLMs-for-Robotics

MODEL="${MODEL:-gemma4-e4b}"
DATASET="${DATASET:-data/multiview_consistency_dataset.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-1}"

module load cuda/13.0
source ~/.bashrc   # sets HF_HOME, TRANSFORMERS_CACHE, etc.
if [ "$(uname -m)" = "aarch64" ]; then
    source "$REPO/.venv-arm64/bin/activate"
else
    source "$REPO/.venv/bin/activate"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"
echo "Model   : $MODEL"
echo "Dataset : $DATASET"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO"

# Multiview consistency dataset is built on the action_phase schema
# (id, scene_id, question_type, images, choices, answer, ...), so it runs
# through the same action_phase task loader.
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
