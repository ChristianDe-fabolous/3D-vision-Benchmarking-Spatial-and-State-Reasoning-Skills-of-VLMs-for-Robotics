#!/usr/bin/env python3
"""
Merge tile annotation files into a single JSONL.

Hardcoded source files are listed in ANNOTATION_FILES below.
Only scenes with >= 3 distinct valid steps (across all files combined) are kept.
Scenes that appear in more than one source file are printed.

Usage
-----
  python scripts/merge_annotations.py
  python scripts/merge_annotations.py --output data/multiview_tiles/merged_annotations.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


# ── edit this list to control which files are merged ─────────────────────────
ANNOTATION_FILES = [
    "data/multiview_tiles/tile_annotations_alex.jsonl",
    "data/multiview_tiles/tile_annotations_defne_7-5.jsonl",
    "data/multiview_tiles/annotations_31_43.jsonl",
    "data/multiview_tiles/annotations_31_43_anna.jsonl",
]

MIN_STEPS = 3
# ─────────────────────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def is_valid(ann: dict) -> bool:
    phase = ann.get("action_phase", "").strip()
    step  = str(ann.get("step", "")).strip()
    return bool(step) and bool(phase) and phase != "\\"


def merge_into(base: dict, new: dict) -> None:
    """Merge new entry into base in-place."""
    for field in ("goal_understandable", "phase_understandable"):
        bd = base.setdefault(field, {})
        for pos in ("top_left", "top_right", "bottom_left"):
            bd[pos] = bd.get(pos, False) or new.get(field, {}).get(pos, False)

    # prefer a valid action_phase if base has none
    if not is_valid(base) and is_valid(new):
        base["step"]         = new["step"]
        base["action_phase"] = new["action_phase"]

    for field in ("task_completed", "mod1_solved", "mod2_solved"):
        if isinstance(base.get(field), bool) and isinstance(new.get(field), bool):
            base[field] = base[field] or new[field]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/multiview_tiles/merged_annotations.jsonl")
    args = p.parse_args()

    root = Path(__file__).parent.parent

    # ── load all files ────────────────────────────────────────────────────────
    merged:        dict[str, dict]              = {}               # original_id → merged entry
    scene_sources: dict[str, set[str]]          = defaultdict(set) # scene_id → {filenames}
    per_file:      dict[str, list[tuple]]       = defaultdict(list) # original_id → [(fname, step, phase)]
    scene_file_entries: dict[tuple, list[dict]] = defaultdict(list) # (scene_id, fname) → entries

    for rel in ANNOTATION_FILES:
        fpath = root / rel
        if not fpath.exists():
            print(f"  WARNING: not found — {rel}")
            continue
        entries = load_jsonl(fpath)
        fname   = fpath.name
        print(f"  {fname}: {len(entries)} entries")
        for e in entries:
            oid = e.get("original_id", "")
            sid = e.get("scene_id", "")
            if not oid:
                continue
            scene_sources[sid].add(fname)
            per_file[oid].append((fname, str(e.get("step", "")), e.get("action_phase", "")))
            scene_file_entries[(sid, fname)].append(e)
            if oid not in merged:
                merged[oid] = e.copy()
            else:
                merge_into(merged[oid], e)

    # ── valid-step counts per (scene, file) ──────────────────────────────────
    def valid_steps_in(entries: list[dict]) -> int:
        return len({str(e["step"]) for e in entries if is_valid(e)})

    scene_file_vsteps: dict[tuple, int] = {
        key: valid_steps_in(ents) for key, ents in scene_file_entries.items()
    }

    # ── print step/phase conflicts (only when both files are "full" scenes) ───
    # build set of (scene_id, fname) pairs that are "full" (>= MIN_STEPS valid steps)
    full_pairs: set[tuple] = {k for k, v in scene_file_vsteps.items() if v >= MIN_STEPS}

    conflicts_found = 0
    for oid in sorted(per_file):
        rows = per_file[oid]
        if len(rows) < 2:
            continue
        if len({(r[1], r[2]) for r in rows}) == 1:
            continue  # all agree
        # get scene_id for this original_id
        sid = merged[oid].get("scene_id", "")
        # only flag if every contributing file is a full scene for this scene
        contributing_files = {r[0] for r in rows}
        if not all((sid, fn) in full_pairs for fn in contributing_files):
            continue  # at least one side is sparse → auto-resolved, no flag
        if conflicts_found == 0:
            print(f"\nConflicts (both files full, step/phase differ — resolve manually):")
        conflicts_found += 1
        print(f"  CONFLICT: {oid}")
        for fname, step, phase in rows:
            print(f"    {fname}: step={step!r}, phase={phase!r}")
    if conflicts_found == 0:
        print(f"\nNo conflicts between full annotation sets.")

    # ── group by scene, count valid steps ────────────────────────────────────
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for ann in merged.values():
        by_scene[ann["scene_id"]].append(ann)

    # ── print scenes annotated in 2+ files ───────────────────────────────────
    multi = {s: fs for s, fs in scene_sources.items() if len(fs) > 1}
    print(f"\nScenes annotated in 2+ files: {len(multi)} / {len(scene_sources)}")
    for sid in sorted(multi):
        print(f"  {sid}")
        for fn in sorted(multi[sid]):
            print(f"    - {fn}")

    # ── filter and write ──────────────────────────────────────────────────────
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept, dropped = 0, 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sid, anns in sorted(by_scene.items()):
            valid_steps = len({str(a["step"]) for a in anns if is_valid(a)})
            if valid_steps >= MIN_STEPS:
                for ann in anns:
                    f.write(json.dumps(ann) + "\n")
                kept += 1
            else:
                dropped += 1

    print(f"\nKept    scenes: {kept}  (>= {MIN_STEPS} valid steps)")
    print(f"Dropped scenes: {dropped}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
