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

First, run the auto-annotation to detect multiview images and create tiles (ALSO DONE ALREADY):

```bash
python scripts/auto_annotate_multiview.py --scenes-file scenes.txt
python scripts/tile_multiview.py
```

Then open the tile annotation viewer (HERE THE WORK BEGINS):

```bash
python scripts/annotate_tiles.py
```

This shows all "has the robot successfully completed the task" questions for one scene at a time, with the image split into tiles side by side.

**What to do for each question row:**

- **Step**: Enter a number (1, 2, 3, ...) indicating where this question falls in the temporal sequence of the scene. Two questions at the same moment get the same step number — their description field will be shared automatically.
Only the order of the numbers matter: 1, 2, 3 = 100, 123, 10000314. After that restart the script **annotate_tiles.py**. The sets will now appear in order.
- **Action phase**: Name the discrete robot action happening at this step. Use action-phase labels like *"approach object with open gripper"*, *"close gripper around object"*, *"transport object to target"*, *"place object and release"*. These are the same answer choices the VLM will see — write them in a form that is unambiguous and could stand alone.
- **Safe to proceed**: Was this action phase fully and correctly completed, such that the robot could immediately start the next phase?
  - `Yes` — phase completed correctly, safe to continue
  - `No` — something went wrong, should not proceed
  - `Unsure` — ambiguous, cannot tell from image alone
- **Task completed**: Check this box on the step where the overall end goal is achieved.
- **Mod 1 / Mod 2**: Optional question modifications. Write a slightly altered version of the task goal that would change the correct answer. Leave empty if nothing comes to mind.
Example: Trivially true: remove x 
  - Check **Solved here** if the modified task would be considered complete at this step.

**Per-tile understandability** — for each tile (top-left, top-right, bottom-left), check two boxes:

- **End goal done**: Can you determine from this tile alone whether the robot has successfully completed the overall end goal task?
- **Phase done**: Can you determine from this tile alone whether the robot has successfully completed the *current action phase* (as you labelled it above)?

Check the box only if the answer is clearly readable from that tile. If the view is occluded, blurry, or ambiguous — leave it unchecked.

**Controls:**

| Key | Action |
|-----|--------|
| `→` | Save and go to next scene |
| `←` | Go to previous scene |
| Save button | Save current scene |
| Mouse wheel | Scroll through questions |
| Top Right Box | Use to jump to scene index |


Annotations are saved to `data/multiview_tiles/tile_annotations.jsonl`.


## Examples instructions
Type 1: Pick and Place / Removal

Used for: Removing objects from containers, bags, or racks.

    Locate Target: Identify the specific object and its container.

    Approach: Move the end-effector to the object's position.

    Grasp: Secure the object.

    Extract: Lift/pull the object out of the container.

    Transport: Move the object to the target destination (if specified).

    Release: Place the object down and retract.

Type 2: Tool Use (Wiping/Capping)

Used for: Removing a lid, using an eraser, or spraying.

    Pre-Grasp: Align with the handle or lid.

    Engage: Attach to the component (e.g., grip the lid).

    Detach/Apply: Perform the primary action (e.g., lift lid or press eraser to board).

    Manipulate: Execute the secondary motion (e.g., wiping stroke).

    Re-home: Return the tool/lid to its original or new location.


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
