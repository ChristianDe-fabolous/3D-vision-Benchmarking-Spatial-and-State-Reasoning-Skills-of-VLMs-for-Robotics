# Annotation Instructions

We are annotating DROID robot manipulation scenes to build a benchmark for evaluating Vision-Language Models (VLMs) as robot supervisors. The goal is to label what the robot is doing at each step and whether it is safe to proceed to the next action phase.

---

## Setup

Make sure all dependencies are installed and the scenes are fetched:

```bash
pip install datasets pillow matplotlib tkinter numpy
```

If we want to get new scenes then we can run, however the hand selected ones are already in this repo, so NO NEED:

```bash
python scripts/fetch_scenes.py --scenes-file scenes.txt
```

This resumes automatically if interrupted.

---

## Pipeline

There are two annotation steps. Run them in order.

---

## Step 1 — Classify images per scene (NOT IMPORTANT AS WE DO THAT AUTOMATICALLY)

```bash
python scripts/annotate_scenes.py --scenes-file scenes.txt
```

This opens a viewer showing all images for one scene at a time in a grid.

**What to do:**

- **Left-click** on images that show a distinct action phase (subtask step). Click them in temporal order — the number shown on each image is its step order. These are the images you will annotate in detail in Step 2.
- **Right-click** on images that show two camera views (multiview). These are handled separately.
- **Skip** scenes where no images are useful (press `S`).

**Controls:**

| Key | Action |
|-----|--------|
| Left-click | Mark as subtask step (ordered) |
| Right-click | Mark as multiview |
| `→` / `Enter` | Save and go to next scene |
| `←` | Save and go to previous scene |
| `C` | Clear all selections for current scene |
| `S` | Skip without saving |
| `Q` | Quit |

**What counts as a subtask step:**

Select one representative image per action phase. A good scene has 3–6 distinct phases, for example:
1. Robot approaching the object
2. Robot grasping the object
3. Robot transporting the object
4. Robot placing/releasing the object

Do not select multiple images of the exact same pose. One clear image per phase is enough.

Annotations are saved to `data/annotations.jsonl` automatically. You can stop and resume at any time.

---

## Step 2 — Annotate action phases and ordering

First, run the auto-annotation to detect multiview images and create tiles:

```bash
python scripts/auto_annotate_multiview.py --scenes-file scenes.txt
python scripts/tile_multiview.py
```

Then open the tile annotation viewer:

```bash
python scripts/annotate_tiles.py
```

This shows all "has the robot successfully completed the task" questions for one scene at a time, with the image split into tiles side by side.

**What to do for each question row:**

- **Step**: Enter a number (1, 2, 3, ...) indicating where this question falls in the temporal sequence of the scene. Two questions at the same moment get the same step number — their description field will be shared automatically.
- **Action phase**: Describe what is visually happening in the image. Write the observable state, not the action label. Example: *"gripper closed around the red cup, cup lifted above table"* not *"picking up cup"*.
- **Safe to proceed**: Is the scene in a valid state to move to the next action phase?
  - `Yes` — previous step completed correctly, ready to proceed
  - `No` — something went wrong, should not proceed
  - `Unsure` — ambiguous, cannot tell from image alone
- **Task completed**: Check this box on the step where the overall task is finished.
- **Mod 1 / Mod 2**: Optional question modifications. Write a slightly altered version of the task goal that would change the correct answer. Leave empty if nothing comes to mind.
  - Check **Solved here** if the modified task would be considered complete at this step.

**Controls:**

| Key | Action |
|-----|--------|
| `→` | Save and go to next scene |
| `←` | Go to previous scene |
| Save button | Save current scene |
| Mouse wheel | Scroll through questions |

Annotations are saved to `data/multiview_tiles/tile_annotations.jsonl`.

---

## Key principles

- Always describe what you **see** in the image, not what you think happened.
- If two questions have the same step number, they happened at the same time — only fill in the description once.
- When in doubt about **Safe to proceed**, use `Unsure`.
- Mark **Task completed** only once per scene, on the step where the final goal is achieved.

---

## Output files

| File | Contents |
|------|----------|
| `data/annotations.jsonl` | Step 1 output: subtask + multiview selections per scene |
| `data/multiview_tiles/entries.jsonl` | Tiled images ready for annotation |
| `data/multiview_tiles/tile_annotations.jsonl` | Step 2 output: action phases, ordering, safe-to-proceed labels |
