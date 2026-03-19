# Implementation Notes

---

## Question Filtering

The dataset contains many question types across many robotic tasks. We do not use all of them — only the subset relevant to our two research tasks.

Filtering and question type assignment happen in a single step via `_classify_task_and_type()` in `src/data/dataset.py`. The function matches question text against `QUESTION_TYPES` in `config.py`, which is the single source of truth:

```python
QUESTION_TYPES = {
    "failure_mode": {
        "goal_state":  ["successfully completed", "goal state"],
        "grasp_state": ["has the robot", ...],
    },
    "multiview": {
        "correspondence": ["3d location", "same 3d", "corresponding"],
        "depth":          ["left image", "right image"],
    },
}
```

A question is matched against all patterns across all tasks and types. The first match determines both the task and the question type. If nothing matches, the question is skipped. This means `QUESTION_TYPES` serves double duty: it filters the 600k questions down to only what we care about, and it assigns a type name used for per-type metric breakdowns.

**`QUESTION_TYPES` is currently empty** — it needs to be filled in before type-level metrics appear. Use `scripts/list_questions.py` to discover all question templates first (see below).

### Fallback when QUESTION_TYPES is empty

When `QUESTION_TYPES` is not yet configured, `_classify_task_and_type()` falls back to `TASK_KEYWORDS` (hardcoded in `data/dataset.py`) for task assignment only — no question type is assigned. This preserves current filtering behaviour during development. Once `QUESTION_TYPES` is fully populated, the fallback and `TASK_KEYWORDS` can be removed.

---

## Dataset Image Format

Source: [`generation/vqa.py`](https://github.com/KeplerC/robo2VLM/blob/main/generation/vqa.py) in the robo2VLM repo.

The `image` field in the HuggingFace dataset always contains exactly one image. For questions involving multiple camera views, the images are stitched into a single composite before storage — the model never receives separate images.

### Single-camera questions

A single annotated camera frame is used directly. For spatial questions (relative depth, relative direction) colored markers or arrows are drawn onto the frame to indicate reference points.

### Multi-view questions (`vqa_multi_view_correspondence`, lines 924–1094)

Two camera views are horizontally concatenated side by side into one image. The camera name (e.g. `ext1`, `ext2`) is **burned as text directly into the image pixels** using `cv2.putText()` (lines 1032–1034). The question text also explicitly names the views:

> "In the left image (ext1 camera), a red dot is marked. Which point in the right image (ext2 camera) corresponds to the same 3D location?"

So the model has two redundant cues: the label in the image and the label in the question text. This is also why our keyword filter (`TASK_KEYWORDS` in `data/dataset.py:28`) uses strings like `"left image"`, `"ext1"`, `"ext2"` — they come directly from these question templates.

### All-camera stitched questions (`stitch_all_cameras`, lines 1471–1577)

All available cameras are combined into a grid layout, padded to uniform size with black space. Each tile has the camera name burned into the top-left corner in yellow text. Used for question types like task success state, gripper state, and action understanding.

---

## Exploration Scripts

Before filling in `QUESTION_TYPES` or `ALLOWED_QUESTION_PATTERNS` in `config.py`, you need to know what questions and answer formats actually exist in the dataset. Two scripts exist for this.

### `scripts/list_questions.py`

Streams through the dataset and extracts all distinct question templates (normalised by stripping scene-specific prefixes). Prints each template with how often it appears and an example.

```bash
python scripts/list_questions.py --limit 5000
python scripts/list_questions.py --local-data /path/to/data
```

Output is saved to `outputs/question_templates.txt`.

**Why this exists:** the dataset has ~107 GB of images and thousands of question variants. Before you can decide which questions to include in your evaluation (and how to classify them into types), you need to see the full vocabulary of question phrasings. This script lets you do that without downloading the full dataset.

### `scripts/list_answer_categories.py`

Streams the dataset and groups questions by their set of answer choices (e.g. `["Yes", "No"]` vs `["Yes", "No", "Cannot be determined"]`). Prints each category with a count and example questions from different scenes.

```bash
python scripts/list_answer_categories.py --limit 5000
python scripts/list_answer_categories.py --no-examples   # counts only
```

Output is saved to `outputs/answer_categories.txt`.

**Why this exists:** the number of answer choices varies per question, which affects both prompt design and the random baseline (1/2 vs 1/3 vs 1/5). Knowing which categories exist and how many questions each has is necessary for deciding which to include and for interpreting results (a 50% accuracy on a binary question is very different from 50% on a 5-way question).

---

## Logging and Output Storage

Every run produces outputs in two locations, both configurable via env vars (`VLM_OUTPUT_DIR`, `VLM_LOG_DIR` — see README for cluster usage).

```
outputs/
  cache.jsonl                    # persistent cross-run response cache (all runs)
  <run_id>/
    config.json                  # exact CLI arguments for this run
    results.jsonl                # one JSON line per evaluated sample, written live
    summary.json                 # aggregated metrics (written at end of run)

logs/
  <run_id>.log                   # debug-level log of every inference step
```

`run_id` is `<timestamp>_<task>_<model>_<prompt>`, e.g. `2026-03-19_14-30-00_failure_mode_qwen-3b_default`.

### `results.jsonl` (`src/utils/logging.py:34`)

Written one line at a time during the run (each write is immediately flushed). This means results are not lost if the run crashes. Each line contains the question, choices, ground truth, raw model response, parsed prediction, and whether it was correct. Metadata fields (e.g. `scene_id`, `question_type`) are included if available.

### `cache.jsonl` (`src/utils/cache.py`)

A single shared file across all runs. Keyed by `(entry_id, model_id, prompt_id)`. When a run starts, already-processed keys are loaded and skipped. This allows resuming interrupted runs and avoids re-running the same sample with the same model and prompt. To force a re-run of a specific sample, delete its line from `cache.jsonl`.

### `summary.json` (`src/evaluation/results.py`)

Written once at the end of the run. Contains all metrics described below.

### `*.log` (`src/utils/logging.py:6`)

Human-readable debug log. Console output is at INFO level (one line per sample showing correct/wrong and the raw response). The file handler is at DEBUG level and includes additional detail.

---

## Metrics

All metrics are computed in `src/evaluation/metrics.py` and written into `summary.json`.

### Overall accuracy

```
correct / total
```

Also reported: number of wrong predictions, number of unparseable responses (model output that could not be matched to A/B/C/D/E), and the random baseline (expected accuracy from uniform random guessing, accounting for variable choice counts).

### Accuracy by question type (`by_question_type`)

Questions are classified into named types via `QUESTION_TYPES` in `config.py`. This is a nested dict: `task → {type_name → [keyword patterns]}`. A question is assigned the first matching type.

Each question type is intended to correspond to one research hypothesis — e.g. "does the model understand spatial depth?" maps to one type, "can it identify object state?" to another. **Note:** the mapping between types and hypotheses is not tracked in the code — it lives in the research design and should be documented separately once types are defined.

`by_question_type` is a flat dict of accuracy per type:

```json
"by_question_type": {
  "state_recognition": 0.62,
  "spatial_depth": 0.41
}
```

Both this and `question_type_analysis` (below) only appear in `summary.json` when `QUESTION_TYPES` is populated in `config.py`.

### Question type outlier analysis (`question_type_analysis`)

Same outlier logic as scene analysis, but applied across question types. Reports the accuracy for every question type plus which types are statistical outliers — performing unusually well or badly compared to the mean across types.

```json
"question_type_analysis": {
  "total_types": 3,
  "mean_accuracy": 0.55,
  "std": 0.12,
  "outlier_std_threshold": 1.0,
  "all_types": [
    {"question_type": "state_recognition", "accuracy": 0.62, "n": 80},
    {"question_type": "spatial_depth",     "accuracy": 0.41, "n": 60}
  ],
  "outliers_above": [...],
  "outliers_below": [...]
}
```

### Scene analysis (`scene_analysis`)

Each dataset entry belongs to a scene (a specific robot + environment combination). The scene ID is parsed from the entry ID (`data/dataset.py:110`). Scene analysis reports accuracy for every scene and flags statistical outliers.

Scenes with fewer than `SCENE_MIN_QUESTIONS` questions (default: 5, `config.py:47`) are excluded to avoid noise from very small samples. The remaining scenes are compared to the mean; scenes more than `SCENE_OUTLIER_STD` standard deviations above or below the mean are listed as outliers.

```json
"scene_analysis": {
  "total_scenes": 120,
  "excluded_scenes": 14,
  "excluded_questions": 42,
  "included_scenes": 106,
  "mean_accuracy": 0.612,
  "std": 0.183,
  "outlier_std_threshold": 1.0,
  "outliers_above": [
    {"scene_id": "14346", "accuracy": 0.923, "n": 13}
  ],
  "outliers_below": [
    {"scene_id": "3301", "accuracy": 0.167, "n": 6}
  ]
}
```

This metric helps detect whether model failures are uniformly distributed or concentrated in specific scenes — which would suggest scene-specific visual properties (lighting, clutter, viewpoint) driving errors rather than the question type.

### Currently unused metrics

`scene_analysis_by_question_type` and `cross_bucket_scene_analysis` exist in `src/evaluation/metrics.py` but are not called. They are not written to `summary.json`.

### Answer category analysis (`--analyse-categories`)

Only computed when `--analyse-categories` is passed to `main.py`. Groups questions by their exact set of choices (order-independent) and reports accuracy and random baseline per group. Useful for checking whether accuracy differences between runs are driven by a shift in which question categories are included rather than genuine model improvement.

---

## Question Type Configuration Workflow

The intended workflow for setting up question type breakdowns:

1. Run `scripts/list_questions.py` to see all question templates.
2. Decide which templates belong to which task and which question type (one type per research hypothesis).
3. Fill in `QUESTION_TYPES` in `config.py` — this simultaneously controls filtering and type assignment.
4. Once fully configured, remove `TASK_KEYWORDS` and the fallback branch from `data/dataset.py`.
5. Run a full evaluation — `summary.json` will now include `by_question_type`, `question_type_analysis`, and `scene_analysis`.
