#!/usr/bin/env python3
"""
Build multiview consistency dataset from action_phase_dataset.jsonl.

For every entry in action_phase_dataset.jsonl (view="combined"), three single-view
variants are generated using the pre-tiled images in data/multiview_tiles/images/:

  top_left    — external camera A
  top_right   — external camera B
  bottom_left — wrist camera

Entry expansion rules
---------------------
action_phase_id, phase_success, task_success  (single image):
  images: [data/multiview_tiles/images/<original_id>_<view>.jpg]

progress  (two images):
  BOTH images use the SAME view position.
  images: [<original_id_a>_<view>.jpg, <original_id_b>_<view>.jpg]

Entries with special_image / special_a / special_b (random_scene, black_image)
are skipped — they reference tiles from other scenes not in our tile directory.

Annotation quality from merged_annotations.jsonl is attached as metadata
(phase_understandable, goal_understandable per view) but does NOT filter entries.
Researchers can filter downstream if needed.

Usage:
  python scripts/build_multiview_consistency_dataset.py
  python scripts/build_multiview_consistency_dataset.py --output data/my_consistency.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

VIEWS = ["top_left", "top_right", "bottom_left"]

TILES_DIR      = "data/multiview_tiles/images"
SRC_DATASET    = "data/action_phase_dataset.jsonl"
ANNOTATIONS    = "data/multiview_tiles/merged_annotations.jsonl"
DEFAULT_OUTPUT = "data/multiview_consistency_dataset.jsonl"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_merged_annotations(path: Path) -> dict[str, dict]:
    """Keyed by original_id."""
    if not path.exists():
        print(f"  [WARN] annotations not found: {path}")
        return {}
    return {e["original_id"]: e for e in load_jsonl(path) if e.get("original_id")}


def tile_img_path(root: Path, original_id: str, view: str) -> Path:
    return root / TILES_DIR / f"{original_id}_{view}.jpg"


def tile_img_rel(original_id: str, view: str) -> str:
    return f"{TILES_DIR}/{original_id}_{view}.jpg"


def quality_for_view(ann: dict | None, view: str) -> tuple[bool | None, bool | None]:
    """Return (phase_understandable, goal_understandable) for the given tile view."""
    if ann is None:
        return None, None
    phase = ann.get("phase_understandable", {}).get(view)
    goal  = ann.get("goal_understandable", {}).get(view)
    return phase, goal


# ── expansion ─────────────────────────────────────────────────────────────────

def expand_single_image(
    entry: dict,
    root: Path,
    merged_anns: dict[str, dict],
    views: list[str],
) -> list[dict]:
    """
    Expand action_phase_id / phase_success / task_success entries.
    Each has one original_id → one tile image per view.
    """
    oid = entry.get("original_id")
    if not oid:
        return []

    ann = merged_anns.get(oid)
    results = []

    for view in views:
        img = tile_img_path(root, oid, view)
        if not img.exists():
            continue

        phase_ok, goal_ok = quality_for_view(ann, view)

        e = {k: v for k, v in entry.items() if k not in ("images", "view", "id")}
        e["view"]                  = view
        e["images"]                = [tile_img_rel(oid, view)]
        e["source_id"]             = entry["id"]
        e["phase_understandable"]  = phase_ok
        e["goal_understandable"]   = goal_ok
        results.append(e)

    return results


def expand_progress(
    entry: dict,
    root: Path,
    merged_anns: dict[str, dict],
    views: list[str],
) -> list[dict]:
    """
    Expand progress entries (two images).
    Both images use the SAME view position so the scene perspective is consistent.
    """
    oid_a = entry.get("original_id_a")
    oid_b = entry.get("original_id_b")
    if not oid_a or not oid_b:
        return []

    ann_a = merged_anns.get(oid_a)
    ann_b = merged_anns.get(oid_b)
    results = []

    for view in views:
        img_a = tile_img_path(root, oid_a, view)
        img_b = tile_img_path(root, oid_b, view)
        if not img_a.exists() or not img_b.exists():
            continue

        phase_a, goal_a = quality_for_view(ann_a, view)
        phase_b, goal_b = quality_for_view(ann_b, view)

        e = {k: v for k, v in entry.items() if k not in ("images", "view", "id")}
        e["view"]                    = view
        e["images"]                  = [tile_img_rel(oid_a, view), tile_img_rel(oid_b, view)]
        e["source_id"]               = entry["id"]
        e["phase_understandable_a"]  = phase_a
        e["phase_understandable_b"]  = phase_b
        e["goal_understandable_a"]   = goal_a
        e["goal_understandable_b"]   = goal_b
        results.append(e)

    return results


def expand_entry(
    entry: dict,
    root: Path,
    merged_anns: dict[str, dict],
    views: list[str] = VIEWS,
) -> list[dict]:
    """
    Dispatch to the correct expansion function based on question_type.
    Returns [] for special/distractor entries (random_scene, black_image).
    """
    if entry.get("special_image") or entry.get("special_a") or entry.get("special_b"):
        return []

    qt = entry["question_type"]

    if qt == "progress":
        return expand_progress(entry, root, merged_anns, views)
    else:
        return expand_single_image(entry, root, merged_anns, views)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Build multiview consistency dataset")
    p.add_argument("--source",      default=SRC_DATASET,    metavar="PATH",
                   help="source action_phase_dataset.jsonl (default: %(default)s)")
    p.add_argument("--annotations", default=ANNOTATIONS,    metavar="PATH",
                   help="merged_annotations.jsonl for tile quality metadata")
    p.add_argument("--output",      default=DEFAULT_OUTPUT, metavar="PATH")
    args = p.parse_args()

    root     = Path(__file__).parent.parent
    src_path = root / args.source
    ann_path = root / args.annotations
    out_path = root / args.output

    print(f"Source   : {src_path}")
    print(f"Tiles    : {root / TILES_DIR}")
    print(f"Output   : {out_path}")
    print()

    print("Loading source dataset…")
    source = load_jsonl(src_path)
    print(f"  {len(source)} entries ({sum(1 for e in source if e.get('special_image') or e.get('special_a') or e.get('special_b'))} special/distractor)")

    print("Loading merged annotations…")
    merged_anns = load_merged_annotations(ann_path)
    print(f"  {len(merged_anns)} annotated original_ids")

    print(f"Expanding to {len(VIEWS)} views: {VIEWS}…")

    dataset:         list[dict]         = []
    skipped_special: int                = 0
    skipped_missing: int                = 0
    qt_view_counts:  dict[str, Counter] = {}

    for entry in source:
        if entry.get("special_image") or entry.get("special_a") or entry.get("special_b"):
            skipped_special += 1
            continue

        expanded = expand_entry(entry, root, merged_anns)

        if not expanded:
            skipped_missing += 1
            continue

        for e in expanded:
            qt = e["question_type"]
            qt_view_counts.setdefault(qt, Counter())[e["view"]] += 1

        dataset.extend(expanded)

    # Assign fresh sequential IDs
    for i, e in enumerate(dataset):
        e["id"] = i

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print(f"  {len(dataset):>6} entries generated")
    print(f"  {skipped_special:>6} skipped  (special_image / random_scene / black_image)")
    print(f"  {skipped_missing:>6} skipped  (missing tile images)")
    print()
    for qt, counts in sorted(qt_view_counts.items()):
        total = sum(counts.values())
        print(f"  {qt} ({total} total):")
        for view in VIEWS:
            print(f"    {view:<14}: {counts.get(view, 0)}")

    # ── write ─────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in dataset:
            f.write(json.dumps(e) + "\n")

    print(f"\nWritten {len(dataset)} entries → {out_path}")

    # ── sanity check ──────────────────────────────────────────────────────────
    bad = sum(
        1 for e in dataset
        for img in e.get("images", [])
        if not (root / img).exists()
    )
    print("  All image paths exist." if bad == 0 else f"  WARNING: {bad} broken image paths.")


if __name__ == "__main__":
    main()
