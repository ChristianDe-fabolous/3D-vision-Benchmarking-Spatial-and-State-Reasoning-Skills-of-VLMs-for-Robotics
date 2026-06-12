#!/bin/bash
# LoRA-finetuned Qwen3-VL on multiview consistency dataset — 5060ti (~16GB VRAM), batch_size=1
# Default adapter is the completed qwen3-4b LoRA run (outputs/lora-qwen3-vl-4b).
# Override MODEL/LORA_ADAPTER together to evaluate the qwen3-8b adapter once trained.
#SBATCH --job-name=qwen3-lora-multiview
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --account=cil_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=12:00:00
#SBATCH --mail-user=cdeubel@ethz.ch
#SBATCH --mail-type=END,FAIL

REPO=/work/courses/3dv/team29/3D-vision-Benchmarking-Spatial-and-State-Reasoning-Skills-of-VLMs-for-Robotics

MODEL="${MODEL:-qwen3-4b}"
LORA_ADAPTER="${LORA_ADAPTER:-outputs/lora-qwen3-vl-4b}"
DATASET="${DATASET:-data/multiview_consistency_dataset.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-1}"

module load cuda/13.0
source ~/.bashrc
export LD_LIBRARY_PATH=""
if [ "$(uname -m)" = "aarch64" ]; then
    source "$REPO/.venv-arm64/bin/activate"
else
    source "$REPO/.venv/bin/activate"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDNN_V8_API_ENABLED=0

echo "========================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURM_NODELIST"
echo "Model        : $MODEL"
echo "LoRA adapter : $LORA_ADAPTER"
echo "Dataset      : $DATASET"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO"

# Multiview consistency dataset is built on the action_phase schema
# (id, scene_id, question_type, images, choices, answer, ...), so it runs
# through the same action_phase task loader. --lora-adapter merges the
# fine-tuned LoRA weights into the base model before inference, so we can
# check whether action_phase fine-tuning transfers to the multiview task.
CMD="python src/main.py \
    --task action_phase \
    --model $MODEL \
    --lora-adapter $LORA_ADAPTER \
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
