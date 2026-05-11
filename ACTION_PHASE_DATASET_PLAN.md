# Action Phase Dataset — Redesign Plan

## Image Format

**All questions use the 2×2 tile image** for each timestep: combine `_top_left`, `_top_right`,
`_bottom_left`, `_bottom_right` quadrant images into a single composite. Reconstruct at
build time via PIL (stitch 4 images into 2×2 grid). Store stitched tiles in
`data/action_phase_images/tiles/`.

Some entries may have only `_left` / `_right` (2-view tiles) — stitch those as 1×2.

**Step inclusion:** a timestep is included if it has a `step` value set in the annotations,
regardless of whether `action_phase` is set. Q1, Q3, Q5 internally filter to steps that also
have an `action_phase`. Q2 and Q6 use all steps with a step number and a stitchable tile.

---

## Terminology

| Symbol | Meaning |
|--------|---------|
| `f` | First step of the scene |
| `l` | Last step of the scene |
| `m` | A middle step (any step that is not f or l) |
| `r` | A tile from a **different** scene (random) |
| `N` | Number of valid steps in a scene |

"Valid step" = has a non-empty `action_phase` that is not `\`.

---

## Data Sources

- Master annotations: `data/multiview_tiles/merged_annotations.jsonl` — produced by `scripts/merge_annotations.py` from the hardcoded list of annotator files; run this first before building the dataset
- Scene order: derived from `data/multiview_tiles/entries.jsonl` (first appearance of each `scene_id` with `split_mode=="4"` and task-completion question) — matches the order seen in `annotate_tiles.py`
- Tile images: `data/multiview_tiles/images/{original_id}_{position}.jpg`
- Stitched tiles: `data/action_phase_images/tiles/{original_id}.jpg` — generated at build time
- Black tile: `data/action_phase_images/tiles/_black.jpg` — generated at build time for Cannot-be-determined entries
- Scene task strings: extracted from `scenes/scene_{scene_id}.jsonl` (only if question contains "Has the robot successfully completed"); otherwise derived from scene_id
- Allowed scenes: `ALLOWED_SCENE_INDICES` in `build_action_phase_dataset.py` — 1-based indices into the entries.jsonl scene order

---

## Q1 — action_phase_id

**Question:** "Is the robot currently performing the following action phase? Action phase: {X}"  
**Choices:** Yes / No / Cannot be determined

For each step i with a phase-annotated tile, generate entries across four categories.
Steps without an `action_phase` annotation are skipped.

### Yes entries — 2N per scene

| variant | image | prompt context | claimed phase | answer |
|---------|-------|---------------|--------------|--------|
| A | tile_i | none | phase_i | Yes |
| B | tile_i | full ordered sequence | phase_i | Yes |

### Cannot be determined — N + P per scene

| image | claimed phase | answer | note |
|-------|--------------|--------|------|
| black image | phase_i | Cannot be determined | one per step (N total) |
| random tile from different scene | phase_k | Cannot be determined | one per unique phase in scene (P total) |

### No entries — up to 4N per scene

For each step i, pair tile_i with a different claimed phase:

| claimed phase | condition | answer |
|--------------|-----------|--------|
| first phase | first phase ≠ phase_i | No |
| last phase | last phase ≠ phase_i | No |
| random phase strictly before step i | such phase exists and ≠ phase_i | No |
| random phase strictly after step i | such phase exists and ≠ phase_i | No |

**Total per scene:** 2N (Yes) + N + P (Cannot) + up to 4N (No)

---

## Q2 — progress

**Question:** "Did the robot make progress toward completing the overall goal from Image 1 to Image 2?"  
**Choices:** Yes / No / Cannot be determined  
**Images:** two tiles shown (Image 1 = earlier claimed, Image 2 = later claimed)

Up to **6 pair types** per scene. For types 1–3 and 6, also produce the **swapped** order
(so 2 entries per type). For types 4–5, no swap.

| # | Image 1 | Image 2 | Answer | Swap? |
|---|---------|---------|--------|-------|
| 1 | tile_f | tile_l | Yes | Yes → also (l, f) = No |
| 2 | tile_f | tile_m | Yes | Yes → also (m, f) = No |
| 3 | tile_l | tile_m | No | Yes → also (m, l) = Yes |
| 4 | tile_f | tile_r | Cannot be determined | No |
| 5 | tile_r1 | tile_r2 (different scene) | Cannot be determined | No |
| 6 | tile_m1 | tile_m2 (m1.step < m2.step) | Yes | Yes → also (m2, m1) = No |

Types 2, 3, 6 require at least one middle step → skipped if N < 3.  
Type 6 requires at least two middle steps → skipped if N < 4.  
r1, r2 sampled from two different other scenes.

Maximum entries per scene: **10** (2+2+2+1+1+2).

---

## Q3 — next_action

**Question:** "Based on what you see in the image, what action phase should the robot perform next?"  
**Choices:** all phases shuffled in scene + "Nothing to do, task already succeeded" + "Cannot be determined"

### Variant A — trivial (no claimed phase, no sequence)

One entry per step. Image = tile_i. No additional context.  
Answer = true next phase of step i (or "task succeeded" if i = last).

Produces **N entries** per scene.

### Variant B — with full ordered sequence as context

Same as Variant A, but the prompt also includes the complete ordered list of all action phases.  
Tests if providing the sequence as context improves accuracy.

Produces **N entries** per scene.

### Variant C — with claimed current phase

For each image (step i), generate entries with 4 different claimed current phases.
Give all action phases in correct order as part of prompt.
Answer is always the true next of step i, regardless of claimed phase.

| claimed phase | condition | per image |
|--------------|-----------|-----------|
| first phase (phase_f) | always | 1 |
| last phase (phase_l) | always | 1 |
| random phase strictly before step i | if such a phase exists | 1 |
| random phase strictly after step i | if such a phase exists | 1 |

Produces up to **3N entries** per scene.

---

## Q5 — phase_success

**Question:** "Has the robot successfully completed the following action phase? Action phase: {X}"  
**Choices:** Yes / No / Cannot be determined

For each image (step i), generate entries with up to 5 claimed phases:

| claimed phase | condition | answer |
|--------------|-----------|--------|
| phase_i (correct, true phase of image) | always | Yes |
| first phase | always | Yes if i = f, else No |
| last phase | always | Yes if i = l, else No |
| random phase strictly before step i | if such phase exists | No (image hasn't reached it yet → wait, image IS at step i which is AFTER the before-phase → Yes) |
| random phase strictly after step i | if such phase exists | No |

Answer rule: Yes if `step(image) >= step(claimed_phase)`, No otherwise.

Produces up to **5N entries** per scene (fewer when no before/after phases exist for edge steps).

---

## Q6 — task_success (new)

**Question:** "Did the robot successfully complete the overall task?"  
**Choices:** Yes / No / Cannot be determined

**Ground truth:** derived from the **"End goal done"** checkbox, stored as
`goal_understandable` (per tile position: `top_left`, `top_right`, `bottom_left`).
If ANY tile position has `goal_understandable = True` for a given `original_id` → task succeeded.

Aggregate with OR across **all annotation files** (multiple annotators may have annotated the
same `original_id`). `task_completed`, `mod1_solved`, `mod2_solved` are NOT used.

| image | ground truth | answer |
|-------|-------------|--------|
| tile_t where `goal_understandable[any position]` is True in any annotation file | robot completed task | Yes |
| tile_t where all `goal_understandable` positions are False in all annotation files | robot has not completed task | No |
| tile_r (random tile from a different scene) | unknown | Cannot be determined |

**Annotation files to merge:** all `*.jsonl` files in `data/multiview_tiles/` except
`entries.jsonl`. Deduplicate by `original_id`, OR-aggregate `goal_understandable` per position.

---

## Q7 - 
---

## Output Schema (per entry)

```json
{
  "id": 0,
  "scene_id": "...",
  "question_type": "action_phase_id | progress | next_action | phase_success | task_success",
  "variant": "A | B | C | D",
  "task": "...",
  "question": "...",
  "tile_ids": ["original_id_1", "original_id_2"],
  "images": ["data/action_phase_images/tiles/original_id_1.jpg"],
  "choices": ["A. Yes", "B. No", "C. Cannot be determined"],
  "answer": "A",
  "answer_text": "Yes",
  "metadata": {
    "image_step": 3,
    "image_phase": "...",
    "label_phase": "...",
    "label_step": 1,
    "special_image": null
  }
}
```

---

## Open Questions

1. **Middle step selection for Q2**: if N > 3 (multiple middles), pick one at random or use all pairs?
   Current plan: pick one representative middle step per scene (e.g., the median step).

2. **Q3 Variant D vs Variant A**: these are identical entries — merge into one with a flag,
   or keep as separate `variant` values for cleaner filtering?

3. **Q5 retention**: original N²+2N scheme kept unchanged. Should it also be pruned to
   first/last/middle like Q1 and Q2?

4. **Tile stitching order**: confirm 2×2 layout is top_left | top_right / bottom_left | bottom_right.
