#!/bin/bash
# ── ETH ISG Student Cluster — Action Phase VLM Benchmark ─────────────────────
#
# Usage:
#   sbatch scripts/run_slurm.sh
#
# Override defaults via env vars:
#   MODEL=qwen-7b-int8 GPU=2080ti sbatch scripts/run_slurm.sh
#   DATASET=data/action_phase_dataset_singleview.jsonl sbatch scripts/run_slurm.sh
#
# Resume a previous run:
#   RUN_ID=my_run RESUME=1 sbatch scripts/run_slurm.sh
#
# Smoke test (first 10 questions with description):
#   SMOKE=1 sbatch scripts/run_slurm.sh
# ─────────────────────────────────────────────────────────────────────────────

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=vlm-bench
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --account=3dvision
#SBATCH --time=12:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G

# ── GPU / model reference ─────────────────────────────────────────────────────
# qwen-3b        → 1080ti  (~6GB VRAM)   --cpus-per-task=2  --mem=24G
# qwen-7b-int8   → 2080ti  (~8GB VRAM)   --cpus-per-task=4  --mem=36G
# qwen-7b        → 5060ti  (~14GB VRAM)  --cpus-per-task=3  --mem=24G  ← default
# qwen-32b-int8  → gb10    (~32GB VRAM)  --cpus-per-task=20 --mem=116G --time=24:00:00
# ─────────────────────────────────────────────────────────────────────────────

# ── User config ───────────────────────────────────────────────────────────────
SCRATCH=/work/scratch/$USER
REPO=$HOME/3D-vision-Benchmarking-Spatial-and-State-Reasoning-Skills-of-VLMs-for-Robotics
VENV=$REPO/3dvision

MODEL=${MODEL:-qwen-7b}
DATASET=${DATASET:-data/action_phase_dataset.jsonl}
# ACTION_PHASE_TYPE: action_phase_id | progress | phase_success | task_success (unset = all)
# ─────────────────────────────────────────────────────────────────────────────

# ── Environment ───────────────────────────────────────────────────────────────
. /etc/profile.d/modules.sh
module add cuda/12.9

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export VLM_OUTPUT_DIR=$SCRATCH/vlm_outputs
export VLM_LOG_DIR=$SCRATCH/vlm_logs

mkdir -p $HF_HOME $HF_DATASETS_CACHE $VLM_OUTPUT_DIR $VLM_LOG_DIR

source $VENV/bin/activate

# ── Job info ──────────────────────────────────────────────────────────────────
echo "========================================"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"
echo "Model   : $MODEL"
echo "Dataset : $DATASET"
echo "Output  : $VLM_OUTPUT_DIR"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd $REPO

# ── Smoke test ────────────────────────────────────────────────────────────────
if [ -n "$SMOKE" ]; then
    SMOKE_RUN_ID=${RUN_ID:-action_phase_${MODEL}_v1}_smoke
    echo "Running smoke test (10 questions) -> $SMOKE_RUN_ID"
    python src/main.py \
        --task action_phase \
        --model $MODEL \
        --action-phase-data $DATASET \
        --image-root $REPO \
        --limit 10 \
        --describe \
        --run-id $SMOKE_RUN_ID \
        --resume
    echo "Smoke done. Check $VLM_OUTPUT_DIR/$SMOKE_RUN_ID/results.jsonl"
    exit 0
fi

# ── Build command ─────────────────────────────────────────────────────────────
RUN_ID=${RUN_ID:-action_phase_${MODEL}_v1}

CMD="python src/main.py \
    --task action_phase \
    --model $MODEL \
    --action-phase-data $DATASET \
    --image-root $REPO \
    --run-id $RUN_ID \
    --resume"

[ -n "$ACTION_PHASE_TYPE" ] && CMD="$CMD --action-phase-type $ACTION_PHASE_TYPE"
[ -n "$LIMIT" ]             && CMD="$CMD --limit $LIMIT"
[ -n "$HF_TOKEN" ]          && CMD="$CMD --hf-token $HF_TOKEN"

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Running: $CMD"
eval $CMD

# ── Results summary ───────────────────────────────────────────────────────────
RESULTS=$VLM_OUTPUT_DIR/$RUN_ID/results.jsonl
if [ -f "$RESULTS" ]; then
    echo ""
    echo "========================================"
    echo "Results summary"
    echo "========================================"
    python3 - "$RESULTS" << 'PYEOF'
import json, sys
from collections import Counter

results = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(f"Total answered: {len(results)}")

by_type = {}
for r in results:
    by_type.setdefault(r["question_type"], []).append(r)

print("\nAccuracy by question type:")
for qt, rows in sorted(by_type.items()):
    acc = sum(r["correct"] for r in rows) / len(rows)
    print(f"  {qt:25s}  {acc:.1%}  (n={len(rows)})")

print("\nAccuracy by (type, variant):")
by_tv = {}
for r in results:
    key = (r["question_type"], r.get("variant", "?"))
    by_tv.setdefault(key, []).append(r)
for (qt, v), rows in sorted(by_tv.items()):
    acc = sum(r["correct"] for r in rows) / len(rows)
    print(f"  {qt:25s}/{v}  {acc:.1%}  (n={len(rows)})")

print("\nAnswer distribution:")
for qt, rows in sorted(by_type.items()):
    pred = Counter(r["predicted_label"] for r in rows)
    gt   = Counter(r["ground_truth_label"] for r in rows)
    print(f"  {qt}")
    print(f"    GT  : {dict(gt)}")
    print(f"    Pred: {dict(pred)}")
PYEOF
fi

echo ""
echo "Done. Outputs in $VLM_OUTPUT_DIR/$RUN_ID"
