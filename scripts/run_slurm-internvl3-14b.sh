#!/bin/bash
# InternVL3-14B BF16 (~28GB) — gb10 (128GB), batch_size=4
#SBATCH --job-name=internvl3-14b
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --time=24:00:00
#SBATCH --mail-user=cdeubel@ethz.ch
#SBATCH --mail-type=END,FAIL

REPO=/work/courses/3dv/team29/3D-vision-Benchmarking-Spatial-and-State-Reasoning-Skills-of-VLMs-for-Robotics

MODEL="${MODEL:-internvl3-14b}"
DATASET="${DATASET:-data/action_phase_dataset.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-4}"

module load cuda/13.0
source ~/.bashrc
source "$REPO/.venv-internvl/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"
echo "Model   : $MODEL"
echo "Dataset : $DATASET"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO"

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

[ "${CLEAN_CACHE:-1}" = "1" ] && rm -rf "$HF_HOME/hub/models--OpenGVLab--InternVL3_5-14B"
echo "HF cache after cleanup: $(du -sh $HF_HOME 2>/dev/null | cut -f1)"
