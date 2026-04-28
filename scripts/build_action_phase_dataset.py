#!/usr/bin/env python3
"""
Build action-phase dataset from tile annotations.

Question types
--------------
Q1  action_phase_id  — Is this image showing action phase X?
                       Choices: Yes / No / Cannot be determined
                       N² + 2N pairs per scene.

Q2  progress         — Did the robot make progress from Image 1 to Image 2?
                       Model is told explicitly which image comes first.
                       Choices: Yes / No / Cannot be determined
                       N² + 2N pairs per scene.

Q3  next_action      — Given claimed current phase and image, what phase is next?
                       Answer based on true step of the image, not claimed phase.
                       Choices: all phases + "Nothing to do, task already succeeded"
                                + "Cannot be determined"
                       N² + 2N pairs per scene.

Q5  phase_success    — Did the robot succeed at action phase X?
                       Yes if step(image) >= step(phase), No if step(image) < step(phase).
                       Choices: Yes / No / Cannot be determined
                       N² + 2N pairs per scene.

Usage
-----
  python scripts/build_action_phase_dataset.py
  python scripts/build_action_phase_dataset.py --output data/my_dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


CANNOT  = "Cannot be determined"
_MAX_FNAME = 120


# ── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def strip_task(question: str) -> str:
    return question.split("Has the robot successfully completed")[0].strip().rstrip(".")


def safe_fname(scene_id: str) -> str:
    if len(scene_id) <= _MAX_FNAME:
        return scene_id
    m = re.search(r"_(\d+)$", scene_id)
    suffix = f"_{m.group(1)}" if m else ""
    return scene_id[:_MAX_FNAME - len(suffix)] + suffix


def load_scene_images(scene_id: str, scenes_dir: Path) -> dict[str, str]:
    """Return {original_id: image_path} from the scene JSONL file."""
    path = scenes_dir / f"scene_{safe_fname(scene_id)}.jsonl"
    if not path.exists():
        return {}
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                e = json.loads(line)
                eid = e.get("id", "")
                img = e.get("image_path", "")
                if eid and img:
                    result[eid] = img
    return result


def rebase(img: str, root: Path) -> str:
    if not img or Path(img).exists():
        return img
    for marker in ("scenes/", "data/"):
        idx = img.find(marker)
        if idx != -1:
            return str(root / img[idx:])
    return img


IMG_DIR = "data/action_phase_images"


def make_black_image(root: Path) -> str:
    """Create black image in the images dir and return relative path."""
    img_dir = root / IMG_DIR
    img_dir.mkdir(parents=True, exist_ok=True)
    path = img_dir / "black_image.jpg"
    if not path.exists():
        Image.new("RGB", (320, 240), (0, 0, 0)).save(path)
    return f"{IMG_DIR}/black_image.jpg"


def copy_image(src: str, root: Path) -> str:
    """Copy image to IMG_DIR and return relative path. Returns src unchanged if copy fails."""
    src_path = Path(src)
    if not src_path.exists():
        return src
    img_dir = root / IMG_DIR
    img_dir.mkdir(parents=True, exist_ok=True)
    dst = img_dir / src_path.name
    if not dst.exists():
        shutil.copy2(src_path, dst)
    return f"{IMG_DIR}/{src_path.name}"


def make_choices(options: list[str]) -> tuple[list[str], dict[str, str]]:
    """Shuffle options, return (labelled_choices, value→letter map)."""
    shuffled = options[:]
    random.shuffle(shuffled)
    labelled = [f"{chr(65 + i)}. {c}" for i, c in enumerate(shuffled)]
    letter_of = {c: chr(65 + i) for i, c in enumerate(shuffled)}
    return labelled, letter_of


# ── scene data ───────────────────────────────────────────────────────────────

def build_steps(scene_anns: list[dict], img_by_oid: dict[str, str], root: Path) -> list[dict]:
    """Return steps sorted by step number. Copies images to IMG_DIR, stores relative paths."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for ann in scene_anns:
        step = str(ann.get("step", "")).strip()
        if step and ann.get("action_phase", "").strip():
            groups[step].append(ann)

    steps = []
    for step_key in sorted(groups, key=lambda s: int(s) if s.isdigit() else 999):
        anns = groups[step_key]
        action_phase = anns[0]["action_phase"].strip()

        images = []
        for ann in anns:
            oid = ann["original_id"]
            abs_img = rebase(img_by_oid.get(oid, ""), root)
            if abs_img and Path(abs_img).exists():
                images.append(copy_image(abs_img, root))

        if not images:
            continue

        steps.append({
            "step":         int(step_key) if step_key.isdigit() else step_key,
            "action_phase": action_phase,
            "images":       images,
        })
    return steps


# ── Q1: is this image showing phase X? ───────────────────────────────────────

def make_q1(scene_id: str, task: str, steps: list[dict],
            black_img: str, rng_imgs: list[str]) -> list[dict]:
    entries = []
    base_choices = ["Yes", "No", CANNOT]

    # N² pairs
    for step_i in steps:
        for step_j in steps:
            correct = "Yes" if step_i["step"] == step_j["step"] else "No"
            choices, lof = make_choices(base_choices)
            entries.append({
                "scene_id":      scene_id,
                "question_type": "action_phase_id",
                "task":          task,
                "question": (
                    f"The robot's goal is: {task}\n\n"
                    f"Is the robot currently performing the following action phase?\n\n"
                    f"Action phase: {step_j['action_phase']}"
                ),
                "images":      step_i["images"],
                "choices":     choices,
                "answer":      lof[correct],
                "answer_text": correct,
                "image_step":  step_i["step"],
                "image_phase": step_i["action_phase"],
                "label_phase": step_j["action_phase"],
                "label_step":  step_j["step"],
            })

    # +N black pairs and +N random pairs
    for step_j in steps:
        for img, label in [(black_img, CANNOT), (random.choice(rng_imgs), CANNOT)]:
            choices, lof = make_choices(base_choices)
            entries.append({
                "scene_id":      scene_id,
                "question_type": "action_phase_id",
                "task":          task,
                "question": (
                    f"The robot's goal is: {task}\n\n"
                    f"Is the robot currently performing the following action phase?\n\n"
                    f"Action phase: {step_j['action_phase']}"
                ),
                "images":      [img],
                "choices":     choices,
                "answer":      lof[label],
                "answer_text": label,
                "image_step":  None,
                "image_phase": None,
                "label_phase": step_j["action_phase"],
                "label_step":  step_j["step"],
                "special_image": "black" if img == black_img else "random_scene",
            })

    return entries


# ── Q2: did the robot make progress? ─────────────────────────────────────────

def make_q2(scene_id: str, task: str, steps: list[dict],
            black_img: str, rng_imgs: list[str]) -> list[dict]:
    entries = []
    base_choices = ["Yes", "No", CANNOT]

    # N² pairs — all ordered (a, b) combinations
    for step_a in steps:
        for step_b in steps:
            correct = "Yes" if step_a["step"] < step_b["step"] else "No"
            choices, lof = make_choices(base_choices)
            entries.append({
                "scene_id":      scene_id,
                "question_type": "progress",
                "task":          task,
                "question": (
                    f"The robot's goal is: {task}\n\n"
                    f"Image 1 (earlier): shows the robot during '{step_a['action_phase']}'\n"
                    f"Image 2 (later):   shows the robot during '{step_b['action_phase']}'\n\n"
                    f"Did the robot make progress toward completing the overall goal "
                    f"from Image 1 to Image 2?"
                ),
                "images":       step_a["images"][:1] + step_b["images"][:1],
                "choices":      choices,
                "answer":       lof[correct],
                "answer_text":  correct,
                "step_a":       step_a["step"],
                "step_b":       step_b["step"],
                "phase_a":      step_a["action_phase"],
                "phase_b":      step_b["action_phase"],
            })

    # +N black second image, +N random second image
    for step_a in steps:
        for img, label in [(black_img, CANNOT), (random.choice(rng_imgs), CANNOT)]:
            choices, lof = make_choices(base_choices)
            entries.append({
                "scene_id":      scene_id,
                "question_type": "progress",
                "task":          task,
                "question": (
                    f"The robot's goal is: {task}\n\n"
                    f"Image 1 (earlier): shows the robot during '{step_a['action_phase']}'\n"
                    f"Image 2 (later):   [second image]\n\n"
                    f"Did the robot make progress toward completing the overall goal "
                    f"from Image 1 to Image 2?"
                ),
                "images":        step_a["images"][:1] + [img],
                "choices":       choices,
                "answer":        lof[label],
                "answer_text":   label,
                "step_a":        step_a["step"],
                "step_b":        None,
                "phase_a":       step_a["action_phase"],
                "phase_b":       None,
                "special_image": "black" if img == black_img else "random_scene",
            })

    return entries


# ── Q3: what action phase comes next? ────────────────────────────────────────

TASK_DONE = "Nothing to do, task already succeeded"


def make_q3(scene_id: str, task: str, steps: list[dict],
            black_img: str, rng_imgs: list[str]) -> list[dict]:
    entries = []
    all_phases   = [s["action_phase"] for s in steps]
    last_step    = steps[-1]["step"]
    base_choices = all_phases + [TASK_DONE, CANNOT]

    def next_phase(step_i: dict) -> str:
        idx = next((i for i, s in enumerate(steps) if s["step"] == step_i["step"]), None)
        if idx is None or idx + 1 >= len(steps):
            return TASK_DONE
        return steps[idx + 1]["action_phase"]

    # N² pairs: (image_i, claimed_phase_j)
    for step_i in steps:
        true_next = next_phase(step_i)
        for step_j in steps:
            choices, lof = make_choices(base_choices)
            entries.append({
                "scene_id":      scene_id,
                "question_type": "next_action",
                "task":          task,
                "question": (
                    f"The robot's goal is: {task}\n\n"
                    f"The robot is currently performing: {step_j['action_phase']}\n\n"
                    f"Based on what you see in the image, what action phase should "
                    f"the robot perform next?"
                ),
                "images":         step_i["images"],
                "choices":        choices,
                "answer":         lof[true_next],
                "answer_text":    true_next,
                "image_step":     step_i["step"],
                "image_phase":    step_i["action_phase"],
                "claimed_phase":  step_j["action_phase"],
                "claimed_step":   step_j["step"],
            })

    # +N black pairs and +N random pairs — per claimed phase_j
    for step_j in steps:
        for img, label in [(black_img, CANNOT), (random.choice(rng_imgs), CANNOT)]:
            choices, lof = make_choices(base_choices)
            entries.append({
                "scene_id":      scene_id,
                "question_type": "next_action",
                "task":          task,
                "question": (
                    f"The robot's goal is: {task}\n\n"
                    f"The robot is currently performing: {step_j['action_phase']}\n\n"
                    f"Based on what you see in the image, what action phase should "
                    f"the robot perform next?"
                ),
                "images":        [img],
                "choices":       choices,
                "answer":        lof[label],
                "answer_text":   label,
                "image_step":    None,
                "image_phase":   None,
                "claimed_phase": step_j["action_phase"],
                "claimed_step":  step_j["step"],
                "special_image": "black" if img == black_img else "random_scene",
            })

    return entries


# ── Q5: did the robot succeed at phase X? ────────────────────────────────────

def make_q5(scene_id: str, task: str, steps: list[dict],
            black_img: str, rng_imgs: list[str]) -> list[dict]:
    entries = []
    base_choices = ["Yes", "No", CANNOT]

    # N² pairs
    for step_i in steps:
        for step_j in steps:
            correct = "Yes" if step_i["step"] >= step_j["step"] else "No"
            choices, lof = make_choices(base_choices)
            entries.append({
                "scene_id":      scene_id,
                "question_type": "phase_success",
                "task":          task,
                "question": (
                    f"The robot's goal is: {task}\n\n"
                    f"Has the robot successfully completed the following action phase?\n\n"
                    f"Action phase: {step_j['action_phase']}"
                ),
                "images":      step_i["images"],
                "choices":     choices,
                "answer":      lof[correct],
                "answer_text": correct,
                "image_step":  step_i["step"],
                "image_phase": step_i["action_phase"],
                "label_phase": step_j["action_phase"],
                "label_step":  step_j["step"],
            })

    # +N black and +N random
    for step_j in steps:
        for img, label in [(black_img, CANNOT), (random.choice(rng_imgs), CANNOT)]:
            choices, lof = make_choices(base_choices)
            entries.append({
                "scene_id":      scene_id,
                "question_type": "phase_success",
                "task":          task,
                "question": (
                    f"The robot's goal is: {task}\n\n"
                    f"Has the robot successfully completed the following action phase?\n\n"
                    f"Action phase: {step_j['action_phase']}"
                ),
                "images":        [img],
                "choices":       choices,
                "answer":        lof[label],
                "answer_text":   label,
                "image_step":    None,
                "image_phase":   None,
                "label_phase":   step_j["action_phase"],
                "label_step":    step_j["step"],
                "special_image": "black" if img == black_img else "random_scene",
            })

    return entries


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", default="data/multiview_tiles/tile_annotations.jsonl")
    p.add_argument("--scenes-dir",  default="scenes")
    p.add_argument("--output",      default="data/action_phase_dataset.jsonl")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--append",      action="store_true",
                   help="Append to existing output, skipping scenes already present")
    args = p.parse_args()

    random.seed(args.seed)
    root       = Path(__file__).parent.parent
    scenes_dir = root / args.scenes_dir

    annotations = load_jsonl(root / args.annotations)

    # Group annotations by scene
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for ann in annotations:
        by_scene[ann["scene_id"]].append(ann)

    # Load original images and task descriptions from scene JSONL files
    task_by_scene:   dict[str, str]       = {}
    images_by_scene: dict[str, list[str]] = defaultdict(list)
    img_by_oid_by_scene: dict[str, dict[str, str]] = {}

    for scene_id in by_scene:
        img_by_oid = load_scene_images(scene_id, scenes_dir)
        img_by_oid_by_scene[scene_id] = img_by_oid
        for oid, img in img_by_oid.items():
            img = rebase(img, root)
            if img and Path(img).exists():
                images_by_scene[scene_id].append(copy_image(img, root))

        # Task from first scene entry that has a question
        scene_file = scenes_dir / f"scene_{safe_fname(scene_id)}.jsonl"
        if scene_file.exists() and scene_id not in task_by_scene:
            with open(scene_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        e = json.loads(line)
                        if e.get("question"):
                            task_by_scene[scene_id] = strip_task(e["question"])
                            break

    black_img = make_black_image(root)
    out_path  = root / args.output

    # Load existing entries when appending
    existing:       list[dict] = []
    existing_scenes: set[str]  = set()
    if args.append and out_path.exists():
        existing = load_jsonl(out_path)
        existing_scenes = {e["scene_id"] for e in existing}
        print(f"Appending — {len(existing)} existing entries, "
              f"{len(existing_scenes)} existing scenes")

    dataset:  list[dict]     = []
    counters: dict[str, int] = defaultdict(int)

    all_scene_ids = list(images_by_scene.keys())

    for scene_id, scene_anns in by_scene.items():
        if scene_id in existing_scenes:
            continue  # already in output

        img_by_oid = img_by_oid_by_scene.get(scene_id, {})
        steps = build_steps(scene_anns, img_by_oid, root)
        if len(steps) < 2:
            continue

        task = task_by_scene.get(scene_id, "")

        other_scenes = [s for s in all_scene_ids if s != scene_id and images_by_scene[s]]
        if not other_scenes:
            continue
        rng_imgs = [random.choice(images_by_scene[s]) for s in other_scenes]

        scene_entries = (
            make_q1(scene_id, task, steps, black_img, rng_imgs) +
            make_q2(scene_id, task, steps, black_img, rng_imgs) +
            make_q3(scene_id, task, steps, black_img, rng_imgs) +
            make_q5(scene_id, task, steps, black_img, rng_imgs)
        )

        for entry in scene_entries:
            counters[entry["question_type"]] += 1

        dataset.extend(scene_entries)

    # Assign ids continuing from existing
    id_offset = max((e["id"] for e in existing), default=-1) + 1
    for i, entry in enumerate(dataset):
        entry["id"] = id_offset + i

    mode = "a" if args.append else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

    total = len(existing) + len(dataset)
    print(f"Written {len(dataset)} new entries ({total} total) → {out_path}")
    for qt, cnt in sorted(counters.items()):
        print(f"  {qt}: {cnt}")


if __name__ == "__main__":
    main()
