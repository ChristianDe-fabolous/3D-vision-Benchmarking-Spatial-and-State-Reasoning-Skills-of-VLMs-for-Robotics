# Implementation Notes

---

## Question Filtering and Type Assignment

The dataset contains many question types across many robotic tasks. We do not use all of them — only the subset relevant to our two research tasks.

Filtering and question type assignment happen in a single step via `_classify_task_and_types()` in `src/data/dataset.py`. The function matches question text against `QUESTION_TYPES` in `config.py`, which is the single source of truth:

```python
QUESTION_TYPES = {
    "failure_mode": {
        "task_success":       ["successfully completed the task"],
        "grasp_phase_current": ["which phase of the grasp action is shown"],
        "grasp_phase_next":   ["what will be the robot's next action phase"],
        "gripper_state":      ["is the robot's gripper open"],
        "grasp_stability":    ["is the robot's grasp of the", "stable"],
        "obstacle_detection": ["is there any obstacle blocking the robot from reaching"],
    },
    "multiview": {
        "cross_view_correspondence": ["corresponding to the same 3d location", "same 3d location"],
        "relative_depth":            ["which colored point is closest", "which colored point is farthest",
                                      "closest to the camera", "farthest from the camera"],
    },
}
```

A question is matched against all patterns within its task. **All matching type names are returned** — a question can belong to more than one type. The full list is stored as `question_types` in the result entry; `question_type` holds the first match and is used for single-type metric breakdowns. Questions that match no pattern in any task are skipped entirely.

### Multi-type overlap and untyped questions

Every run writes `question_type_issues.txt` to the run output directory with two sections:
- **Multi-type overlap**: questions that matched more than one type, listed with their id, choices, correct answer, and which types matched.
- **Untyped but included**: questions that matched a task via the `TASK_KEYWORDS` fallback but no `QUESTION_TYPES` pattern — they appear in runs without a type assignment.

The same report can be generated for the full dataset without running inference using `scripts/list_question_type_overlaps.py`.

### Fallback when QUESTION_TYPES is empty

When `QUESTION_TYPES` is not yet configured, `_classify_task_and_types()` falls back to `TASK_KEYWORDS` (hardcoded in `data/dataset.py`) for task assignment only — no question type is assigned. Once `QUESTION_TYPES` is fully populated, the fallback and `TASK_KEYWORDS` can be removed.

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

## Dataset Analysis Scripts

All scripts write output to `dataset_analysis/` (gitignored). They all support `--split` (default: `train`), `--limit`, and `--local-data`. All stream from HuggingFace and only fetch the columns they need (no images downloaded).

Progress is printed every 1000 rows.

### `scripts/list_questions.py`

Streams through the dataset and extracts all distinct question templates (normalised by stripping scene-specific prefixes). Output saved to `dataset_analysis/question_templates_<split>.txt`.

```bash
python scripts/list_questions.py
python scripts/list_questions.py --split test --limit 5000
```

**Caveat:** the number of distinct templates will appear much higher than the actual number of question types — many questions reference scene-specific objects. When defining keyword patterns for `QUESTION_TYPES`, match the shared structural part, not the object-specific parts.

### `scripts/list_answer_categories.py`

Groups questions by their set of answer choices and reports counts and example questions per category. Output saved to `dataset_analysis/answer_categories_<split>.txt`.

```bash
python scripts/list_answer_categories.py
python scripts/list_answer_categories.py --no-examples
```

**Caveat:** color-based questions (relative depth, relative direction) produce many near-identical categories because the set of colors varies per scene.

### `scripts/list_questions_with_answers.py`

Lists every unique question with occurrence count, and below it all distinct answer choice sets that appear with that question (also counted). Questions and choice sets sorted by descending count. If a question has more than 5 distinct answer sets, only the top 4 are shown and the rest collapsed into `(others)`. Output saved to `dataset_analysis/questions_with_answers_<split>.txt`.

```bash
python scripts/list_questions_with_answers.py
```

### `scripts/list_questions_with_answers_chunked.py` + `merge_questions_with_answers_chunks.py`

Same as above but with checkpointing — writes a JSONL delta file every `--checkpoint` rows (default 100k) so a crash does not lose all progress. On restart it detects the state file and resumes automatically via `.skip()`.

```bash
# Run (resumes automatically if interrupted)
python scripts/list_questions_with_answers_chunked.py --checkpoint 100000

# Start fresh, ignoring existing state
python scripts/list_questions_with_answers_chunked.py --fresh

# Merge all chunks into final text output
python scripts/merge_questions_with_answers_chunks.py
```

Chunk files: `dataset_analysis/chunks/<split>/chunk_<start>_<end>.jsonl`
State file: `dataset_analysis/chunks/<split>/state.json`

### `scripts/list_question_type_overlaps.py`

Scans the dataset and reports:
1. Questions matching more than one question type (per current `QUESTION_TYPES` config)
2. Questions matching a task via `TASK_KEYWORDS` fallback but no question type

Each entry is listed with id, question, choices, and correct answer. Output saved to `dataset_analysis/question_type_overlaps_<split>.txt`.

```bash
python scripts/list_question_type_overlaps.py
python scripts/list_question_type_overlaps.py --split test
```

---

## Logging and Output Storage

Every run produces outputs in two locations, both configurable via env vars (`VLM_OUTPUT_DIR`, `VLM_LOG_DIR` — see README for cluster usage).

```
outputs/
  <run_id>/
    config.json                  # exact CLI arguments for this run
    results.jsonl                # one JSON line per evaluated sample, written live
    summary.json                 # aggregated metrics (written at end of run)
    question_type_issues.txt     # multi-type overlaps and untyped questions in this run

logs/
  <run_id>.log                   # debug-level log of every inference step
```

`run_id` is `<timestamp>_<task>_<model>_<prompt>`, e.g. `2026-03-19_14-30-00_failure_mode_qwen-3b_default`.

### `results.jsonl` (`src/utils/logging.py:34`)

Written one line at a time during the run (each write is immediately flushed). This means results are not lost if the run crashes. Each line contains the question, choices, ground truth, raw model response, parsed prediction, and whether it was correct. Metadata fields included if available:

- `scene_id` — parsed from entry ID
- `question_types` — list of all matched type names (can be more than one); first entry is used as primary for per-type metrics

### `summary.json` (`src/evaluation/results.py`)

Written once at the end of the run. Contains all metrics described below.

### `question_type_issues.txt` (`src/evaluation/results.py`)

Written once at the end of every run. Two sections:
- Questions that matched multiple question types (with id, choices, correct answer, matched types)
- Questions that were included in the run but matched no question type (via fallback)

If both sections are empty the file still exists so you can confirm it was checked.

### Response cache (not currently active)

`src/utils/cache.py` contains a `ResponseCache` implementation that would persist all responses to `outputs/cache.jsonl` keyed by `(entry_id, model_id, prompt_id)`, allowing interrupted runs to resume without re-running already-processed samples. It is not used at the moment. Could be worth enabling once runs get long enough that resumability matters.

### `*.log` (`src/utils/logging.py:6`)

Human-readable debug log. Console output is at INFO level (one line per sample showing correct/wrong and the raw response). The file handler is at DEBUG level and includes additional detail.

---

## Response Parsing

The model is expected to reply with a single letter (A, B, C, D, or E). Parsing is done in `src/tasks/base.py:20` (`parse_response`):

1. Strip whitespace from the response.
2. Take the first character and uppercase it.
3. Check whether it is a valid label for this question (e.g. only A/B/C for a 3-choice question).
4. If valid, return the 0-based index (A→0, B→1, …). If not, return `None`.

A `None` result counts as **unparseable** — not wrong, tracked separately in `summary.json`. This matters because an unparseable response could mean the model refused to answer, produced verbose output instead of a letter, or genuinely didn't know. The prompts instruct the model to reply with only the letter to minimise this.

**Caveat:** the parser only checks the first character. A response like `"A real interesting question"` would be parsed as `A`. The full raw response is always stored in `response_raw` so re-parsing with a stricter parser is possible without re-running inference.

`evaluate()` (`tasks/base.py:31`) calls `parse_response` and returns `True` only if the parsed index matches `sample.correct_answer`.

---

## Metrics

All metrics are computed in `src/evaluation/metrics.py` and written into `summary.json`.

### Overall accuracy

```
correct / total
```

Also reported: number of wrong predictions, number of unparseable responses (model output that could not be matched to A/B/C/D/E), and the random baseline (expected accuracy from uniform random guessing, accounting for variable choice counts).

### Accuracy by question type (`by_question_type`)

Questions are classified into named types via `QUESTION_TYPES` in `config.py`. A question can match multiple types — `question_type` (the first match) is used here for per-type accuracy. Only present when `QUESTION_TYPES` is populated.

```json
"by_question_type": {
  "task_success": 0.62,
  "grasp_phase_current": 0.41
}
```

### Question type outlier analysis (`question_type_analysis`)

Same outlier logic as scene analysis, but applied across question types. Reports accuracy for every type plus which are statistical outliers. Only present when `QUESTION_TYPES` is populated.

```json
"question_type_analysis": {
  "total_types": 3,
  "mean_accuracy": 0.55,
  "std": 0.12,
  "outlier_std_threshold": 1.0,
  "all_types": [
    {"question_type": "task_success",        "accuracy": 0.62, "n": 80},
    {"question_type": "grasp_phase_current", "accuracy": 0.41, "n": 60}
  ],
  "outliers_above": [...],
  "outliers_below": [...]
}
```

### Scene analysis (`scene_analysis`)

Each dataset entry belongs to a scene (a specific robot + environment combination). The scene ID is parsed from the entry ID (`data/dataset.py:110`). Scene analysis reports accuracy for every scene and flags statistical outliers.

Scenes with fewer than `SCENE_MIN_QUESTIONS` questions (default: 5, `config.py`) are excluded to avoid noise from very small samples. The remaining scenes are compared to the mean across all included scenes; scenes more than `SCENE_OUTLIER_STD` standard deviations above or below are listed as outliers.

```json
"scene_analysis": {
  "total_scenes": 120,
  "excluded_scenes": 14,
  "excluded_questions": 42,
  "included_scenes": 106,
  "mean_accuracy": 0.612,
  "std": 0.183,
  "outlier_std_threshold": 1.0,
  "outliers_above": [{"scene_id": "14346", "accuracy": 0.923, "n": 13}],
  "outliers_below": [{"scene_id": "3301",  "accuracy": 0.167, "n": 6}]
}
```

This metric helps detect whether model failures are uniformly distributed or concentrated in specific scenes — which would suggest scene-specific visual properties (lighting, clutter, viewpoint) driving errors rather than the question type.

### Answer distribution (`answer_distribution`)

Always present. For each distinct answer label, tracks how often it appears as the ground truth, how often the model predicted it, and the model's accuracy on questions where that label is correct. Sorted by ground truth frequency.

```json
"answer_distribution": [
  {"label": "No",  "ground_truth_count": 412, "predicted_count": 380, "accuracy_when_gt": 0.871},
  {"label": "Yes", "ground_truth_count": 201, "predicted_count": 230, "accuracy_when_gt": 0.791},
  {"label": "Cannot be determined", "ground_truth_count": 38, "predicted_count": 12, "accuracy_when_gt": 0.421}
]
```

Useful for detecting label imbalance (e.g. "No" dominating the ground truth) and whether the model is biased towards predicting certain answers regardless of the question.

### Answer category analysis (`--analyse-categories`)

Only computed when `--analyse-categories` is passed to `main.py`. Groups questions by their exact set of choices (order-independent) and reports accuracy and random baseline per group. Useful for checking whether accuracy differences between runs are driven by a shift in which question categories are included rather than genuine model improvement.

### Currently unused metrics

`scene_analysis_by_question_type` and `cross_bucket_scene_analysis` exist in `src/evaluation/metrics.py` but are not called. They are not written to `summary.json`.

---

## Question Type Configuration Workflow

`QUESTION_TYPES` in `config.py` is now populated. The current categories are:

**failure_mode**: `task_success`, `grasp_phase_current`, `grasp_phase_next`, `gripper_state`, `grasp_stability`, `obstacle_detection`

**multiview**: `cross_view_correspondence`, `relative_depth`

If categories need to be revised:

1. Run `scripts/list_questions_with_answers_chunked.py` + `merge_questions_with_answers_chunks.py` to get the full question/answer inventory.
2. Run `scripts/list_question_type_overlaps.py` to check for overlapping or unmatched questions.
3. Update patterns in `QUESTION_TYPES` in `config.py`.
4. Re-run the overlap script to verify the result.
5. Once fully configured with no untyped questions, remove `TASK_KEYWORDS` and the fallback branch from `data/dataset.py`.
