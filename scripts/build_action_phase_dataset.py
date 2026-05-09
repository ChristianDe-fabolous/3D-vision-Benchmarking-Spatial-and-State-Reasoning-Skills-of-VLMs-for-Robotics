#!/usr/bin/env python3
"""
Build action-phase dataset from tile annotations (redesigned).

Question types
--------------
Q1  action_phase_id
    A: N correct pairs (naive, one per step)
    B: first/last phase × correct/random/middle image (up to 7 entries/scene)

Q2  progress
    Up to 6 pair types; types 1-3,6 include swapped order; types 4-5 do not.

Q3  next_action
    A: trivial N entries (no context)
    B: N entries with full ordered sequence as context
    C: up to 4N entries with claimed current phase (first, last, before, after)

Q5  phase_success
    For each image: up to 5 claimed phases (correct, first, last, random-before, random-after)

Q6  task_success
    Ground truth: goal_understandable (End goal done) OR-aggregated across all annotation files.
    Random cross-scene tile -> Cannot be determined.

Usage
-----
  python scripts/build_action_phase_dataset.py
  python scripts/build_action_phase_dataset.py --output data/my_dataset.jsonl --append
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image


CANNOT    = "Cannot be determined"
TASK_DONE = "Task already succeeded"
DOING     = "Is currently performing it"
TILES_SUBDIR     = "data/action_phase_images/tiles"
ANNOTATIONS_FILE = "data/multiview_tiles/merged_annotations.jsonl"
ENTRIES_FILE     = "data/multiview_tiles/entries.jsonl"
_MAX_FNAME       = 120

# ── allowed scenes — 1-based indices into the sorted scene list ───────────────
ALLOWED_SCENE_INDICES: list[int] = [
    1, 4, 7, 9, 13, 21, 22, 23, 24, 28, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 63, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 84, 93, 95, 97, 101, 105, 107, 111, 115
]
# ─────────────────────────────────────────────────────────────────────────────


# ── generic helpers ───────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def safe_fname(scene_id: str) -> str:
    if len(scene_id) <= _MAX_FNAME:
        return scene_id
    m = re.search(r"_(\d+)$", scene_id)
    suffix = f"_{m.group(1)}" if m else ""
    return scene_id[:_MAX_FNAME - len(suffix)] + suffix


def strip_task(question: str) -> str:
    return question.split("Has the robot successfully completed")[0].strip().rstrip(".")


def make_choices(options: list[str], correct: str | None = None,
                 n: int | None = None) -> tuple[list[str], dict[str, str]]:
    """
    Shuffle options; return (labelled list, value->letter map).
    If n is given, sample n options total, always including `correct`.
    """
    pool = options[:]
    if n is not None and correct is not None and len(pool) > n:
        others = [o for o in pool if o != correct]
        pool   = random.sample(others, min(n - 1, len(others))) + [correct]
    random.shuffle(pool)
    labelled  = [f"{chr(65 + i)}. {c}" for i, c in enumerate(pool)]
    letter_of = {c: chr(65 + i) for i, c in enumerate(pool)}
    return labelled, letter_of


# ── tile stitching ────────────────────────────────────────────────────────────

def stitch_tile(original_id: str, src_dir: Path, out_dir: Path) -> str | None:
    """
    Combine quadrant images into a single composite tile.
    Returns relative path (from project root) or None if no images found.
    """
    out_path = out_dir / f"{original_id}.jpg"
    rel_path = f"{TILES_SUBDIR}/{original_id}.jpg"

    if out_path.exists():
        return rel_path

    quads = {pos: src_dir / f"{original_id}_{pos}.jpg"
             for pos in ("top_left", "top_right", "bottom_left", "bottom_right")}
    found = {k: v for k, v in quads.items() if v.exists()}

    if len(found) == 4:
        tl = Image.open(quads["top_left"]).convert("RGB")
        tr = Image.open(quads["top_right"]).convert("RGB")
        bl = Image.open(quads["bottom_left"]).convert("RGB")
        br = Image.open(quads["bottom_right"]).convert("RGB")
        w  = tl.width + tr.width
        h  = tl.height + bl.height
        composite = Image.new("RGB", (w, h))
        composite.paste(tl, (0, 0))
        composite.paste(tr, (tl.width, 0))
        composite.paste(bl, (0, tl.height))
        composite.paste(br, (tl.width, tl.height))
        composite.save(out_path, quality=90)
        return rel_path

    if found:
        for pos in ("top_left", "top_right", "bottom_left", "bottom_right"):
            if pos in found:
                Image.open(found[pos]).convert("RGB").save(out_path, quality=90)
                return rel_path

    # try _left / _right layout
    lr = {pos: src_dir / f"{original_id}_{pos}.jpg" for pos in ("left", "right")}
    lr_found = {k: v for k, v in lr.items() if v.exists()}
    if lr_found:
        imgs = [Image.open(v).convert("RGB") for v in lr_found.values()]
        if len(imgs) == 2:
            li, ri = imgs
            composite = Image.new("RGB", (li.width + ri.width, max(li.height, ri.height)))
            composite.paste(li, (0, 0))
            composite.paste(ri, (li.width, 0))
            composite.save(out_path, quality=90)
        else:
            imgs[0].save(out_path, quality=90)
        return rel_path

    return None


# ── annotation loading ────────────────────────────────────────────────────────

def scene_order_from_entries(root: Path) -> list[str]:
    """Return scene_ids in the same order annotate_tiles.py presents them."""
    fpath = root / ENTRIES_FILE
    if not fpath.exists():
        raise SystemExit(f"ERROR: entries file not found: {fpath}")
    seen:  set[str]  = set()
    order: list[str] = []
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("split_mode") != "4":
                continue
            if "successfully completed" not in e.get("question", "").lower():
                continue
            sid = e.get("scene_id", "")
            if sid and sid not in seen:
                seen.add(sid)
                order.append(sid)
    return order


def load_merged_annotations(root: Path) -> dict[str, dict]:
    """Load the pre-merged master annotations file."""
    fpath = root / ANNOTATIONS_FILE
    if not fpath.exists():
        raise SystemExit(f"ERROR: master annotations not found: {fpath}\n"
                         "Run scripts/merge_annotations.py first.")
    return {e["original_id"]: e for e in load_jsonl(fpath) if e.get("original_id")}


def is_goal_done(ann: dict) -> bool:
    return any(ann.get("goal_understandable", {}).values())


# ── single-view helpers ───────────────────────────────────────────────────────

_POSITIONS = ("top_left", "top_right", "bottom_left")

def _crop_box(pos: str, img_size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = img_size
    hw, hh = w // 2, h // 2
    return {
        "top_left":    (0,  0,  hw, hh),
        "top_right":   (hw, 0,  w,  hh),
        "bottom_left": (0,  hh, hw, h),
    }[pos]


def extract_single_view(tile_rel: str, pos: str, tile_out: Path, src_imgs: Path) -> str:
    stem     = Path(tile_rel).stem
    out_path = tile_out / f"{stem}_{pos}.jpg"
    rel_path = f"{TILES_SUBDIR}/{stem}_{pos}.jpg"
    if out_path.exists():
        return rel_path
    # prefer original source image — no double compression
    orig = src_imgs / f"{stem}_{pos}.jpg"
    if orig.exists():
        import shutil
        shutil.copy2(orig, out_path)
        return rel_path
    # fallback: crop from stitched tile (one extra compression step)
    src = tile_out / f"{stem}.jpg"
    if not src.exists():
        return tile_rel
    img = Image.open(src).convert("RGB")
    img.crop(_crop_box(pos, img.size)).save(out_path, quality=95)
    return rel_path


def views_ok_for_entry(entry: dict, merged_anns: dict[str, dict]) -> list[str]:
    if entry.get("special_image") or entry.get("special_a") or entry.get("special_b"):
        return []
    check_key = ("goal_understandable"
                 if entry["question_type"] == "task_success"
                 else "phase_understandable")
    if entry["question_type"] == "progress":
        oid_a = entry.get("original_id_a")
        oid_b = entry.get("original_id_b")
        if not oid_a or not oid_b:
            return []
        ann_a = merged_anns.get(oid_a, {})
        ann_b = merged_anns.get(oid_b, {})
        return [p for p in _POSITIONS
                if ann_a.get(check_key, {}).get(p) and ann_b.get(check_key, {}).get(p)]
    else:
        oid = entry.get("original_id")
        if not oid:
            return []
        ann = merged_anns.get(oid, {})
        return [p for p in _POSITIONS if ann.get(check_key, {}).get(p)]


def expand_with_single_views(
    entries: list[dict],
    merged_anns: dict[str, dict],
    tile_out: Path,
    src_imgs: Path,
) -> list[dict]:
    result = []
    for entry in entries:
        views = views_ok_for_entry(entry, merged_anns)
        if len(views) < 2:
            continue
        for pos in views[:2]:
            e = dict(entry)
            e["view"] = pos
            if entry["question_type"] == "progress":
                e["images"] = [
                    extract_single_view(entry["images"][0], pos, tile_out, src_imgs),
                    extract_single_view(entry["images"][1], pos, tile_out, src_imgs),
                ]
            else:
                e["images"] = [extract_single_view(entry["images"][0], pos, tile_out, src_imgs)]
            result.append(e)
    return result


# ── scene helpers ─────────────────────────────────────────────────────────────

def build_steps(scene_anns: list[dict], src_dir: Path, out_dir: Path) -> list[dict]:
    """
    Group annotations by step, sort, stitch tiles.
    Returns list of {step, action_phase, tile, original_id}.
    action_phase may be "" for steps where no phase was annotated.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for ann in scene_anns:
        step = str(ann.get("step", "")).strip()
        if step:
            groups[step].append(ann)

    steps = []
    for step_key in sorted(groups, key=lambda s: int(s) if s.isdigit() else 999):
        anns = groups[step_key]
        phase = next(
            (a.get("action_phase", "").strip()
             for a in anns
             if a.get("action_phase", "").strip() and a.get("action_phase", "").strip() != "\\"),
            ""
        )

        tile_path = None
        rep_oid   = None
        for ann in anns:
            oid = ann["original_id"]
            tp  = stitch_tile(oid, src_dir, out_dir)
            if tp:
                tile_path = tp
                rep_oid   = oid
                break

        if not tile_path:
            continue

        steps.append({
            "step":         int(step_key) if step_key.isdigit() else step_key,
            "action_phase": phase,
            "tile":         tile_path,
            "original_id":  rep_oid,
        })
    return steps


def _task_from_scene_id(scene_id: str) -> str:
    s = scene_id
    if s.startswith("droid_"):
        s = s[6:]
    s = re.sub(r"_\d+$", "", s)
    return s.replace("_", " ")


def get_task(scene_id: str, scenes_dir: Path) -> str:
    scene_file = scenes_dir / f"scene_{safe_fname(scene_id)}.jsonl"
    if scene_file.exists():
        with open(scene_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    e = json.loads(line)
                    q = e.get("question", "")
                    if "Has the robot successfully completed" in q:
                        return strip_task(q)
    return _task_from_scene_id(scene_id)


def _next_phase(steps: list[dict], step_i: dict) -> str:
    idx = next((i for i, s in enumerate(steps) if s["step"] == step_i["step"]), None)
    if idx is None or idx + 1 >= len(steps):
        return TASK_DONE
    return steps[idx + 1]["action_phase"]


def _dedup_steps(candidates: list[dict]) -> list[dict]:
    seen, out = set(), []
    for s in candidates:
        key = s["action_phase"]
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


# ── Q1: action_phase_id ───────────────────────────────────────────────────────

def make_black_tile(out_dir: Path, size: tuple[int, int] = (1280, 960)) -> str:
    out_path = out_dir / "_black.jpg"
    rel_path = f"{TILES_SUBDIR}/_black.jpg"
    if not out_path.exists():
        Image.new("RGB", size, (0, 0, 0)).save(out_path, quality=90)
    return rel_path


def _q1_entry(scene_id, task, tile, img_step, img_phase,
              label_phase, label_step, ans_text, seq_text=None, special=None,
              original_id=None) -> dict:
    base    = ["Yes", "No", CANNOT]
    ch, lof = make_choices(base)
    seq_block = f"\nThe action phases in order are:\n{seq_text}\n" if seq_text else ""
    entry = {
        "scene_id":      scene_id,
        "question_type": "action_phase_id",
        "variant":       "B" if seq_text else "A",
        "task":          task,
        "question": (
            f"The robot's goal is: {task}\n"
            f"{seq_block}\n"
            f"Is the robot currently performing the following action phase?\n\n"
            f"Action phase: {label_phase}"
        ),
        "images":      [tile],
        "choices":     ch,
        "answer":      lof[ans_text],
        "answer_text": ans_text,
        "image_step":  img_step,
        "image_phase": img_phase,
        "label_phase": label_phase,
        "label_step":  label_step,
        "view":        "combined",
    }
    if special:
        entry["special_image"] = special
    if original_id:
        entry["original_id"] = original_id
    return entry


def make_q1(scene_id: str, task: str, steps: list[dict],
            rng_tiles: list[str]) -> list[dict]:
    steps = [s for s in steps if s["action_phase"]]
    if not steps:
        return []
    entries  = []
    seq_text = "\n".join(f"{i+1}. {s['action_phase']}" for i, s in enumerate(steps))

    # unique phases in order (for random-tile Cannot entries)
    unique_phases = list(dict.fromkeys(s["action_phase"] for s in steps))

    for step_i in steps:
        ph_i       = step_i["action_phase"]
        label_step = step_i["step"]

        oid_i = step_i.get("original_id")

        # ── Yes A: tile_i + phase_i, no context ──────────────────────────────
        entries.append(_q1_entry(
            scene_id, task,
            step_i["tile"], step_i["step"], ph_i,
            ph_i, label_step, "Yes", original_id=oid_i,
        ))

        # ── Yes B: tile_i + phase_i, with full sequence context ───────────────
        entries.append(_q1_entry(
            scene_id, task,
            step_i["tile"], step_i["step"], ph_i,
            ph_i, label_step, "Yes", seq_text=seq_text, original_id=oid_i,
        ))

        # ── No A+B: tile_i + randomly chosen different phase ──────────────────
        other_phases = [p for p in unique_phases if p != ph_i]
        if other_phases:
            wrong_phase = random.choice(other_phases)
            wrong_step  = next(s["step"] for s in steps if s["action_phase"] == wrong_phase)
            entries.append(_q1_entry(
                scene_id, task,
                step_i["tile"], step_i["step"], ph_i,
                wrong_phase, wrong_step, "No", original_id=oid_i,
            ))
            entries.append(_q1_entry(
                scene_id, task,
                step_i["tile"], step_i["step"], ph_i,
                wrong_phase, wrong_step, "No", seq_text=seq_text, original_id=oid_i,
            ))

    # ── Cannot: one random cross-scene tile per unique phase ──────────────────
    for phase in unique_phases:
        if not rng_tiles:
            break
        rng_tile   = random.choice(rng_tiles)
        label_step = next(s["step"] for s in steps if s["action_phase"] == phase)
        entries.append(_q1_entry(
            scene_id, task,
            rng_tile, None, None,
            phase, label_step, CANNOT, special="random_scene",
        ))

    return entries


# ── Q2: progress ──────────────────────────────────────────────────────────────

def _prog_entry(scene_id, task, tile_a, step_a, phase_a,
                tile_b, step_b, phase_b, ans_text,
                seq_text=None, special_a=None, special_b=None,
                original_id_a=None, original_id_b=None) -> dict:
    base      = ["Yes", "No", CANNOT]
    ch, lof   = make_choices(base)
    seq_block = f"The action phases in order are:\n{seq_text}\n\n" if seq_text else ""
    e = {
        "scene_id":      scene_id,
        "question_type": "progress",
        "variant":       "B" if seq_text else "A",
        "task":          task,
        "question": (
            f"The robot's goal is: {task}\n\n"
            f"{seq_block}"
            f"The following two images were taken during the robot's task execution.\n"
            f"Did the robot make progress toward completing the overall goal "
            f"from Image 1 to Image 2?"
        ),
        "images":      [tile_a, tile_b],
        "choices":     ch,
        "answer":      lof[ans_text],
        "answer_text": ans_text,
        "step_a":      step_a,
        "step_b":      step_b,
        "phase_a":     phase_a,
        "phase_b":     phase_b,
        "view":        "combined",
    }
    if special_a:
        e["special_a"] = special_a
    if special_b:
        e["special_b"] = special_b
    if original_id_a:
        e["original_id_a"] = original_id_a
    if original_id_b:
        e["original_id_b"] = original_id_b
    return e


def make_q2(scene_id: str, task: str, steps: list[dict], rng_tiles: list[str]) -> list[dict]:
    entries = []
    N = len(steps)

    phase_steps = [s for s in steps if s["action_phase"]]
    seq_text    = ("\n".join(f"{i+1}. {s['action_phase']}" for i, s in enumerate(phase_steps))
                   if phase_steps else None)

    def add_pair(a, b, ans):
        entries.append(_prog_entry(
            scene_id, task,
            a["tile"], a["step"], a["action_phase"],
            b["tile"], b["step"], b["action_phase"],
            ans,
            original_id_a=a.get("original_id"),
            original_id_b=b.get("original_id"),
        ))
        if seq_text:
            entries.append(_prog_entry(
                scene_id, task,
                a["tile"], a["step"], a["action_phase"],
                b["tile"], b["step"], b["action_phase"],
                ans, seq_text=seq_text,
                original_id_a=a.get("original_id"),
                original_id_b=b.get("original_id"),
            ))

    # Fixed: first → last (Yes) and last → first (No)
    add_pair(steps[0], steps[-1], "Yes")
    add_pair(steps[-1], steps[0], "No")

    if N < 2:
        return entries

    # All (i, j) pairs with i < j, excluding (0, N-1) already covered
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)
                 if not (i == 0 and j == N - 1)]

    if not all_pairs:
        return entries

    # Sort by distance: close = small index gap, far = large index gap
    all_pairs.sort(key=lambda p: p[1] - p[0])
    close_pairs = all_pairs[:max(1, len(all_pairs) // 3)]
    far_pairs   = all_pairs[-(max(1, len(all_pairs) // 3)):]

    selected: list[tuple[int, int]] = []
    if len(close_pairs) >= 2:
        selected += random.sample(close_pairs, 2)
    elif close_pairs:
        selected += close_pairs
    if len(far_pairs) >= 2:
        candidates = [p for p in far_pairs if p not in selected]
        selected  += random.sample(candidates, min(2, len(candidates)))
    elif far_pairs:
        selected += [p for p in far_pairs if p not in selected]

    for i, j in selected:
        add_pair(steps[i], steps[j], "Yes")
        add_pair(steps[j], steps[i], "No")

    # Cannot: 3 middle tiles each paired with a random cross-scene tile
    middles = steps[1:-1]
    sampled_middles = random.sample(middles, min(3, len(middles)))
    for m in sampled_middles:
        if not rng_tiles:
            break
        rng_tile = random.choice(rng_tiles)
        for st in ([None, seq_text] if seq_text else [None]):
            entries.append(_prog_entry(
                scene_id, task,
                m["tile"], m["step"], m["action_phase"],
                rng_tile, None, None,
                CANNOT, st, None, "random_scene",
            ))

    return entries


# ── Q3: next_action ───────────────────────────────────────────────────────────

def make_q3(scene_id: str, task: str, steps: list[dict],
            black_tile: str, rng_tiles: list[str]) -> list[dict]:
    steps = [s for s in steps if s["action_phase"]]
    if not steps:
        return []
    entries      = []
    seen_ph: set[str] = set()
    all_phases: list[str] = []
    for s in steps:
        if s["action_phase"] not in seen_ph:
            all_phases.append(s["action_phase"])
            seen_ph.add(s["action_phase"])
    seq_text = "\n".join(f"{i+1}. {s['action_phase']}" for i, s in enumerate(steps))

    def q3_choices(correct: str):
        """Up to 4 action phases + CANNOT + TASK_DONE (always fixed).
        If correct is an action phase it is force-included in the 4 phase slots.
        If correct is CANNOT or TASK_DONE those are already fixed — sample 4 freely."""
        pool = all_phases[:]
        if correct in pool:
            others = [p for p in pool if p != correct]
            pool   = random.sample(others, min(3, len(others))) + [correct]
        else:
            pool = random.sample(pool, min(4, len(pool)))
        pool += [CANNOT, TASK_DONE]
        random.shuffle(pool)
        labelled  = [f"{chr(65 + i)}. {c}" for i, c in enumerate(pool)]
        letter_of = {c: chr(65 + i) for i, c in enumerate(pool)}
        return labelled, letter_of

    for step_i in steps:
        true_next = _next_phase(steps, step_i)
        idx_i     = next(i for i, s in enumerate(steps) if s["step"] == step_i["step"])
        before    = [s["action_phase"] for s in steps[:idx_i]]
        after     = [s["action_phase"] for s in steps[idx_i + 1:]]
        oid_i     = step_i.get("original_id")

        # Sample choices once per step — shared across A, B, and all C entries
        ch, lof = q3_choices(true_next)

        # Variant A: trivial, no context
        entries.append({
            "scene_id":      scene_id,
            "question_type": "next_action",
            "variant":       "A",
            "task":          task,
            "question": (
                f"The robot's goal is: {task}\n\n"
                f"Based on what you see in the image, what action phase should "
                f"the robot perform next?"
            ),
            "images":        [step_i["tile"]],
            "choices":       ch,
            "answer":        lof[true_next],
            "answer_text":   true_next,
            "image_step":    step_i["step"],
            "image_phase":   step_i["action_phase"],
            "claimed_phase": None,
            "original_id":   oid_i,
            "view":          "combined",
        })

        # Variant B: with full ordered sequence
        entries.append({
            "scene_id":      scene_id,
            "question_type": "next_action",
            "variant":       "B",
            "task":          task,
            "question": (
                f"The robot's goal is: {task}\n\n"
                f"The action phases in order are:\n{seq_text}\n\n"
                f"Based on what you see in the image, what action phase should "
                f"the robot perform next?"
            ),
            "images":        [step_i["tile"]],
            "choices":       ch,
            "answer":        lof[true_next],
            "answer_text":   true_next,
            "image_step":    step_i["step"],
            "image_phase":   step_i["action_phase"],
            "claimed_phase": None,
            "original_id":   oid_i,
            "view":          "combined",
        })

        # Variant C: claimed current phase (first, last, random-before, random-after)
        claims_raw = [steps[0]["action_phase"], steps[-1]["action_phase"]]
        if before:
            claims_raw.append(random.choice(before))
        if after:
            claims_raw.append(random.choice(after))

        seen_claims: set[str] = set()
        for claimed in claims_raw:
            if claimed in seen_claims:
                continue
            seen_claims.add(claimed)
            entries.append({
                "scene_id":      scene_id,
                "question_type": "next_action",
                "variant":       "C",
                "task":          task,
                "question": (
                    f"The robot's goal is: {task}\n\n"
                    f"The action phases in order are:\n{seq_text}\n\n"
                    f"The robot is currently performing: {claimed}\n\n"
                    f"Based on what you see in the image, what action phase should "
                    f"the robot perform next?"
                ),
                "images":        [step_i["tile"]],
                "choices":       ch,
                "answer":        lof[true_next],
                "answer_text":   true_next,
                "image_step":    step_i["step"],
                "image_phase":   step_i["action_phase"],
                "claimed_phase": claimed,
                "original_id":   oid_i,
                "view":          "combined",
            })

    # ── Cannot: black image — one per step ───────────────────────────────────
    q_base = (
        f"The robot's goal is: {task}\n\n"
        f"Based on what you see in the image, what action phase should "
        f"the robot perform next?"
    )
    for _ in steps:
        ch, lof = q3_choices(CANNOT)
        entries.append({
            "scene_id":      scene_id,
            "question_type": "next_action",
            "variant":       "A",
            "task":          task,
            "question":      q_base,
            "images":        [black_tile],
            "choices":       ch,
            "answer":        lof[CANNOT],
            "answer_text":   CANNOT,
            "image_step":    None,
            "image_phase":   None,
            "claimed_phase": None,
            "special_image": "black_image",
            "view":          "combined",
        })

    # ── Cannot: random cross-scene tile — one per rng_tile ───────────────────
    for rng_tile in rng_tiles:
        ch, lof = q3_choices(CANNOT)
        entries.append({
            "scene_id":      scene_id,
            "question_type": "next_action",
            "variant":       "A",
            "task":          task,
            "question":      q_base,
            "images":        [rng_tile],
            "choices":       ch,
            "answer":        lof[CANNOT],
            "answer_text":   CANNOT,
            "image_step":    None,
            "image_phase":   None,
            "claimed_phase": None,
            "special_image": "random_scene",
            "view":          "combined",
        })

    return entries


# ── Q5: phase_success ─────────────────────────────────────────────────────────

def make_q5(scene_id: str, task: str, steps: list[dict]) -> list[dict]:
    steps = [s for s in steps if s["action_phase"]]
    if not steps:
        return []
    entries  = []
    base     = ["Yes", "No", CANNOT]
    f, l     = steps[0], steps[-1]
    seq_text = "\n".join(f"{i+1}. {s['action_phase']}" for i, s in enumerate(steps))

    for step_i in steps:
        idx_i  = next(i for i, s in enumerate(steps) if s["step"] == step_i["step"])
        before = steps[:idx_i]
        after  = steps[idx_i + 1:]

        candidates = [step_i]
        if f["step"] != step_i["step"]:
            candidates.append(f)
        if l["step"] != step_i["step"]:
            candidates.append(l)
        if before:
            candidates.append(random.choice(before))
        if after:
            candidates.append(random.choice(after))

        oid_i = step_i.get("original_id")
        for claimed in _dedup_steps(candidates):
            if step_i["step"] > claimed["step"]:
                correct = "Yes"
            elif step_i["step"] == claimed["step"]:
                correct = DOING
            else:
                correct = "No"
            for variant, q_text in (
                ("A", (
                    f"The robot's goal is: {task}\n\n"
                    f"Has the robot successfully completed the following action phase?\n\n"
                    f"Action phase: {claimed['action_phase']}"
                )),
                ("B", (
                    f"The robot's goal is: {task}\n\n"
                    f"The action phases in order are:\n{seq_text}\n\n"
                    f"Has the robot successfully completed the following action phase?\n\n"
                    f"Action phase: {claimed['action_phase']}"
                )),
            ):
                ch, lof = make_choices(["Yes", "No", CANNOT, DOING])
                entries.append({
                    "scene_id":      scene_id,
                    "question_type": "phase_success",
                    "variant":       variant,
                    "task":          task,
                    "question":      q_text,
                    "images":        [step_i["tile"]],
                    "choices":       ch,
                    "answer":        lof[correct],
                    "answer_text":   correct,
                    "image_step":    step_i["step"],
                    "image_phase":   step_i["action_phase"],
                    "label_phase":   claimed["action_phase"],
                    "label_step":    claimed["step"],
                    "original_id":   oid_i,
                    "view":          "combined",
                })

    return entries


# ── Q6: task_success ──────────────────────────────────────────────────────────

def make_q6(scene_id: str, task: str, steps: list[dict],
            merged_anns: dict[str, dict], rng_tiles: list[str]) -> list[dict]:
    entries    = []
    base       = ["Yes", "No", CANNOT]
    phase_steps = [s for s in steps if s["action_phase"]]
    seq_text    = ("\n".join(f"{i+1}. {s['action_phase']}" for i, s in enumerate(phase_steps))
                   if phase_steps else None)

    for step_i in steps:
        ann = merged_anns.get(step_i["original_id"])
        if ann is None:
            continue
        ans_text = "Yes" if is_goal_done(ann) else "No"
        for variant, seq in (("A", None), ("B", seq_text)):
            if variant == "B" and not seq_text:
                continue
            seq_block = f"The action phases in order are:\n{seq}\n\n" if seq else ""
            ch, lof   = make_choices(base)
            entries.append({
                "scene_id":      scene_id,
                "question_type": "task_success",
                "variant":       variant,
                "task":          task,
                "question": (
                    f"The robot's goal is: {task}\n\n"
                    f"{seq_block}"
                    f"Did the robot successfully complete the overall task?"
                ),
                "images":      [step_i["tile"]],
                "choices":     ch,
                "answer":      lof[ans_text],
                "answer_text": ans_text,
                "image_step":  step_i["step"],
                "image_phase": step_i["action_phase"],
                "original_id": step_i["original_id"],
                "view":        "combined",
            })

    # Random cross-scene tiles -> Cannot be determined (at most 2 per scene, variant A only)
    for rng_tile in rng_tiles[:2]:
        ch, lof = make_choices(base)
        entries.append({
            "scene_id":      scene_id,
            "question_type": "task_success",
            "variant":       "A",
            "task":          task,
            "question": (
                f"The robot's goal is: {task}\n\n"
                f"Did the robot successfully complete the overall task?"
            ),
            "images":        [rng_tile],
            "choices":       ch,
            "answer":        lof[CANNOT],
            "answer_text":   CANNOT,
            "image_step":    None,
            "image_phase":   None,
            "special_image": "random_scene",
            "view":          "combined",
        })

    return entries


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations-dir", default="data/multiview_tiles",
                   help="Directory with annotation JSONL files and images/ subfolder")
    p.add_argument("--scenes-dir",      default="scenes")
    p.add_argument("--output",             default="data/action_phase_dataset.jsonl")
    p.add_argument("--single-view-output", default=None,
                   help="If set, also write single-view variant dataset to this path "
                        "(default: data/action_phase_dataset_singleview.jsonl alongside --output)")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--append",          action="store_true",
                   help="Append new scenes; skip scenes already present in output")
    args = p.parse_args()

    random.seed(args.seed)

    root       = Path(__file__).parent.parent
    ann_dir    = root / args.annotations_dir
    src_imgs   = ann_dir / "images"
    scenes_dir = root / args.scenes_dir
    out_path   = root / args.output
    tile_out   = root / TILES_SUBDIR
    tile_out.mkdir(parents=True, exist_ok=True)

    sv_out_path = (
        root / args.single_view_output
        if args.single_view_output
        else out_path.parent / (out_path.stem + "_singleview" + out_path.suffix)
    )

    print("Loading annotations…")
    merged_anns = load_merged_annotations(root)

    by_scene: dict[str, list[dict]] = defaultdict(list)
    for ann in merged_anns.values():
        by_scene[ann["scene_id"]].append(ann)
    print(f"  {len(merged_anns)} original_ids across {len(by_scene)} scenes")

    existing:        list[dict] = []
    existing_scenes: set[str]   = set()
    if args.append and out_path.exists():
        existing        = load_jsonl(out_path)
        existing_scenes = {e["scene_id"] for e in existing}
        print(f"Appending — {len(existing)} existing entries, {len(existing_scenes)} existing scenes")

    if not ALLOWED_SCENE_INDICES:
        raise SystemExit("ERROR: ALLOWED_SCENE_INDICES is empty — add indices before building.")

    all_sorted_scenes = scene_order_from_entries(root)
    n_scenes = len(all_sorted_scenes)
    print(f"  {n_scenes} scenes in entries.jsonl order")

    selected_scene_ids: list[str] = []
    for idx in ALLOWED_SCENE_INDICES:
        if 1 <= idx <= n_scenes:
            selected_scene_ids.append(all_sorted_scenes[idx - 1])
        else:
            print(f"  WARNING: index {idx} out of range (1–{n_scenes}), skipped")
    print(f"  {len(selected_scene_ids)} scenes selected via ALLOWED_SCENE_INDICES")

    print("Building steps and stitching tiles…")
    scene_steps:     dict[str, list[dict]] = {}
    scene_rep_tiles: dict[str, str]        = {}
    for idx, scene_id in zip(ALLOWED_SCENE_INDICES, selected_scene_ids):
        anns  = by_scene[scene_id]
        steps = build_steps(anns, src_imgs, tile_out)
        scene_steps[scene_id] = steps
        if steps:
            scene_rep_tiles[scene_id] = steps[0]["tile"]
        else:
            raw_steps = {str(a.get("step", "")).strip() for a in anns if str(a.get("step", "")).strip()}
            print(f"  WARNING [{idx}] {scene_id}: 0 steps with tiles")
            print(f"    entries={len(anns)}  annotated step values={sorted(raw_steps)}")

    all_scene_ids = [sid for sid, steps in scene_steps.items() if steps]
    print(f"  {len(all_scene_ids)} / {len(selected_scene_ids)} scenes have at least one stitched tile")

    black_tile = make_black_tile(tile_out)

    dataset:  list[dict]     = []
    counters: dict[str, int] = defaultdict(int)

    for scene_id, steps in scene_steps.items():
        if scene_id in existing_scenes:
            continue

        task      = get_task(scene_id, scenes_dir)
        other_ids = [s for s in all_scene_ids if s != scene_id]
        rng_tiles = [scene_rep_tiles[s]
                     for s in random.sample(other_ids, min(2, len(other_ids)))]

        scene_entries = (
            make_q1(scene_id, task, steps, rng_tiles)
            + make_q2(scene_id, task, steps, rng_tiles)
            + make_q3(scene_id, task, steps, black_tile, rng_tiles)
            + make_q5(scene_id, task, steps)
            + make_q6(scene_id, task, steps, merged_anns, rng_tiles)
        )

        for entry in scene_entries:
            counters[entry["question_type"]] += 1

        dataset.extend(scene_entries)

    id_offset = max((e.get("id", -1) for e in existing), default=-1) + 1
    for i, entry in enumerate(dataset):
        entry["id"] = id_offset + i

    mode = "a" if args.append else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

    total = len(existing) + len(dataset)
    print(f"\nWritten {len(dataset)} new entries ({total} total) -> {out_path}")
    for qt, cnt in sorted(counters.items()):
        print(f"  {qt}: {cnt}")

    # ── sanity check: warn about entries missing image metadata ──────────────
    root_path = Path(__file__).parent.parent
    bad = 0
    for e in dataset:
        for img in e.get("images", []):
            if not (root_path / img).exists():
                print(f"  WARN missing image: id={e.get('id')} path={img}")
                bad += 1
    if bad == 0:
        print("  All image paths exist.")

    # ── single-view expansion ─────────────────────────────────────────────────
    print(f"\nExpanding single-view variants -> {sv_out_path}")
    sv_dataset = expand_with_single_views(dataset, merged_anns, tile_out, src_imgs)
    sv_id_offset = max((e.get("id", -1) for e in existing), default=-1) + 1
    for i, entry in enumerate(sv_dataset):
        entry["id"] = sv_id_offset + i

    sv_counters: dict[str, int] = defaultdict(int)
    for e in sv_dataset:
        sv_counters[e["question_type"]] += 1

    with open(sv_out_path, "w", encoding="utf-8") as f:
        for entry in sv_dataset:
            f.write(json.dumps(entry) + "\n")

    print(f"  {len(sv_dataset)} single-view entries written")
    for qt, cnt in sorted(sv_counters.items()):
        print(f"  {qt}: {cnt}")

    bad_sv = 0
    for e in sv_dataset:
        for img in e.get("images", []):
            if not (root_path / img).exists():
                print(f"  WARN missing sv image: id={e.get('id')} path={img}")
                bad_sv += 1
    if bad_sv == 0:
        print("  All single-view image paths exist.")


if __name__ == "__main__":
    main()
