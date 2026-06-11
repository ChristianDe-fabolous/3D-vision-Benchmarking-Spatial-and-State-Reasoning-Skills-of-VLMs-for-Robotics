#!/bin/bash
# LoRA fine-tune Qwen3-VL-8B-Instruct on action_phase_dataset — gb10 (128GB)
#SBATCH --job-name=lora-qwen3-8b
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --account=3dv
#SBATCH --gpus=gb10:1
#SBATCH --time=12:00:00
#SBATCH --mail-user=cdeubel@ethz.ch
#SBATCH --mail-type=END,FAIL

REPO=/work/courses/3dv/team29/3D-vision-Benchmarking-Spatial-and-State-Reasoning-Skills-of-VLMs-for-Robotics

DATASET="${DATASET:-data/action_phase_dataset.jsonl}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lora-qwen3-vl-8b}"
N_TRAIN_SCENES="${N_TRAIN_SCENES:-20}"
N_VAL_SCENES="${N_VAL_SCENES:-7}"
EPOCHS="${EPOCHS:-3}"

module load cuda/13.0
source ~/.bashrc   # sets HF_HOME, TRANSFORMERS_CACHE, etc.
export LD_LIBRARY_PATH=""
if [ "$(uname -m)" = "aarch64" ]; then
    source "$REPO/.venv-arm64/bin/activate"
else
    source "$REPO/.venv/bin/activate"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDNN_V8_API_ENABLED=0

echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "Dataset    : $DATASET"
echo "Output dir : $OUTPUT_DIR"
echo "Seed       : $SEED"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO"

CMD="python scripts/train_lora_qwen3_8b.py \
    --data $DATASET \
    --seed $SEED \
    --output-dir $OUTPUT_DIR \
    --n-train-scenes $N_TRAIN_SCENES \
    --n-val-scenes $N_VAL_SCENES \
    --epochs $EPOCHS"

echo "CMD: $CMD"
eval $CMD

echo "========================================"
echo "Job $SLURM_JOB_ID done."
