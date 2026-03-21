# Benchmarking Spatial and State Reasoning Skills of VLMs for Robotics

ETH Zurich — 3D Vision Course Project, SS26

Evaluates Vision-Language Models (VLMs) on two robotics-focused reasoning tasks using the [Robo2VLM-1](https://huggingface.co/datasets/keplerccc/Robo2VLM-1) dataset (~107 GB, streamed — never fully downloaded).

---
## Research Tasks

### 1. Failure Mode Detection
Can a VLM correctly identify the current state of a robotic task from an image?
Questions ask things like *"Has the robot successfully completed the task?"* with multiple-choice answers. Some samples are augmented with reversed image order (as if the robot did the opposite action).

### 2. Multi-View Consistency
Can a VLM extract spatial information from multiple camera angles?
Questions involve two camera perspectives (ext1/ext2) presented as a single composite image. Tasks include 3D point correspondence and depth reasoning.

---

## Setup

```bash
python -m venv 3dvision
source 3dvision/bin/activate
pip install -r requirements.txt
```

Requires a GPU with sufficient VRAM for Qwen2.5-VL-7B (~16 GB in bfloat16).

---

## Running the Pipeline

```bash
cd src
python main.py --task <task> --model <model> [options]
```

### Arguments

| Argument | Required | Values | Default | Description |
|---|---|---|---|---|
| `--task` | yes | `failure_mode`, `multiview` | — | Which evaluation task to run |
| `--model` | no | `qwen-3b`, `qwen-7b` | `qwen-3b` | Which VLM to use |
| `--split` | no | `train`, `test` | `test` | Dataset split |
| `--limit` | no | int | None | Stop after N samples (useful for testing) |
| `--local-data` | no | path | None | Load from a local directory instead of HuggingFace |
| `--analyse-categories` | no | flag | off | Include per-answer-category accuracy breakdown in summary |

### Examples

```bash
# Quick smoke test on Colab / local (3B, first 10 samples)
python main.py --task failure_mode --model qwen-3b --limit 10

# Full run on cluster (7B)
python main.py --task failure_mode --model qwen-7b

# Multiview task from local data
python main.py --task multiview --model qwen-3b --local-data /path/to/data
```

---

## Outputs

Each run writes to `outputs/<run_id>/` where `run_id` is `<timestamp>_<task>_<model>_<prompt>`:

```
outputs/
  <run_id>/
    config.json                   # exact run parameters
    results.jsonl                 # one JSON line per evaluated sample
    summary.json                  # aggregated accuracy metrics
    question_type_issues.txt      # questions with multiple types or no type assigned
logs/
  <run_id>.log                    # debug-level log of every inference
```

### Result Entry Schema (`results.jsonl`)

```json
{
  "run_id": "2026-03-17_10-00-00_failure_mode_qwen-3b_default",
  "model_id": "qwen-3b",
  "entry_id": "42",
  "task": "failure_mode",
  "question": "Has the robot successfully completed the task?",
  "choices": ["Yes", "No", "Cannot be determined"],
  "ground_truth_index": 1,
  "ground_truth_label": "No",
  "response_raw": "B",
  "predicted_index": 1,
  "predicted_label": "No",
  "correct": true,
  "scene_id": "14346",
  "question_types": ["task_success"],
  "question_type": "task_success"
}
```

`question_types` is the full list of all matched types (a question can match more than one). `question_type` is the primary type (first match) used for per-type metric breakdowns.

---

## Repository Structure

```
src/
  main.py               # CLI entry point
  pipeline.py           # main evaluation loop
  config.py             # paths, model IDs, QUESTION_TYPES, invalid entry IDs

  data/
    dataset.py          # HuggingFace streaming loader, Sample dataclass
    failure_mode.py     # prompt builder for failure mode task
    multiview.py        # prompt builder for multiview task

  tasks/
    base.py             # BaseTask: parse_response, evaluate
    failure_mode.py     # FailureModeTask
    multiview.py        # MultiviewTask

  models/
    base.py             # BaseVLM abstract interface
    qwen.py             # Qwen2.5-VL-7B-Instruct wrapper

  evaluation/
    metrics.py          # accuracy, scene_analysis, answer_distribution, etc.
    results.py          # save_config, save_summary, save_type_issues

  utils/
    cache.py            # ResponseCache (JSONL-based skip logic, not currently active)
    logging.py          # setup_logger, SampleLogger

scripts/
  list_questions.py                      # unique question templates + counts
  list_answer_categories.py             # unique answer choice sets + counts
  list_questions_with_answers.py        # questions with their answer sets (single pass)
  list_questions_with_answers_chunked.py # same but with checkpointing + resume
  merge_questions_with_answers_chunks.py # merge chunk files into final text output
  list_question_type_overlaps.py        # questions matching multiple types or no type
  run_slurm.sh                          # SLURM job template (TODO: adapt to cluster)
```

All dataset analysis scripts write to `dataset_analysis/`. See NOTES.md for details.

---

## Dataset Notes

- **Known wrong answers**: entries `#5, #35, #37, #67, #69` are skipped automatically (see `INVALID_ENTRIES` in `config.py`)
- **Choices**: stored as a Python list string in HF; parsed to `["Yes", "No", ...]`. Number of choices varies per question (2–5). Labels A/B/C/D/E are assigned at prompt time.
- **Failure mode augmentation**: some samples have images in reverse order — the model should still identify the correct state
- **Multiview images**: the `image` field is a pre-composed side-by-side composite of both camera views

---

## Cluster (SLURM)

> **TODO:** SLURM setup has not been tested yet. The script at `scripts/run_slurm.sh` is a starting point but needs to be adapted to the target cluster (partition names, memory/GPU requirements, paths).

Output and log dirs are overridable via env vars so runs write to scratch rather than filling home quota:

```bash
export HF_HOME=/scratch/$USER/hf_cache       # HuggingFace model weights
export VLM_OUTPUT_DIR=/scratch/$USER/vlm_outputs
export VLM_LOG_DIR=/scratch/$USER/vlm_logs
```

A sbatch template is at `scripts/run_slurm.sh`. Customise the paths at the top, then submit with optional overrides:

```bash
# Defaults: failure_mode, qwen-3b, test split
sbatch scripts/run_slurm.sh

# Override task/model via env vars
TASK=multiview MODEL=qwen-7b sbatch scripts/run_slurm.sh
```

## Google Colab

Use `qwen-3b` (fits on a free T4, ~8 GB VRAM). A ready-to-run notebook is available at `notebooks/run_minimal_inference.ipynb` — open it in Colab and follow the cells.

---

## Adding a New Model

1. Create `src/models/<name>.py` implementing `BaseVLM` (`load()` and `infer()`)
2. Add a constant to `config.py`: `MODEL_<NAME> = "<name>"`
3. Register it in `main.py`: add to `build_model()` and the `--model` choices list

---

## Google Drive (data / shared files)

https://drive.google.com/drive/folders/1gMF-vDXdjZAspC9u8j9JzT4NB8wk7aGd
