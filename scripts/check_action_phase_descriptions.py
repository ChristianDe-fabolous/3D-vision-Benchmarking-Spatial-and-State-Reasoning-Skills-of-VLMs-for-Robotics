#!/usr/bin/env python3
"""
Scan jsonl dataset files for entries whose action-phase description is
genuinely absent, and print those entries.

Different question types store the action-phase text under different keys,
and not every question type has one at all:

  action_phase_id, phase_success  → `label_phase`            (single phase)
  progress                        → `phase_a` AND `phase_b`  (two phases)
  task_success                    → (none — not applicable, skipped)

Annotation-style files (no `question_type`, e.g. merged_annotations.jsonl)
store it under `action_phase` instead and are checked directly.

Only entries whose relevant field(s) are missing / None / empty / whitespace
are reported — i.e. the description is *absent*, not just "looks empty" due
to a schema that doesn't carry one for that question type (task_success is
skipped entirely for this reason).

This script is read-only: it does not modify any files.

Usage:
  python scripts/check_action_phase_descriptions.py
  python scripts/check_action_phase_descriptions.py path/to/file.jsonl [more files...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

DEFAULT_GLOBS = [
    "data/*.jsonl",
    "data/multiview_tiles/*.jsonl",
]

# question_type → list of fields that must all be non-empty for the
# action-phase description to count as present. question_types absent from
# this map (e.g. task_success) don't carry a phase description and are skipped.
QTYPE_PHASE_FIELDS = {
    "action_phase_id": ["label_phase"],
    "phase_success":   ["label_phase"],
    "progress":        ["phase_a", "phase_b"],
}

# Fallback field for files without a `question_type` (annotation-style files).
ANNOTATION_PHASE_FIELD = "action_phase"


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def is_empty(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def missing_fields(entry: dict) -> list[str] | None:
    """
    Return the list of empty/absent phase fields for `entry`, or None if this
    entry's question type doesn't carry a phase description at all (skip).
    """
    qt = entry.get("question_type")

    if qt is not None:
        fields = QTYPE_PHASE_FIELDS.get(qt)
        if fields is None:
            return None  # e.g. task_success — nothing to check
        empty = [f for f in fields if is_empty(entry.get(f))]
        return empty if empty else []

    # Annotation-style file: single `action_phase` field.
    if ANNOTATION_PHASE_FIELD in entry:
        return [ANNOTATION_PHASE_FIELD] if is_empty(entry.get(ANNOTATION_PHASE_FIELD)) else []

    return None


def describe(entry: dict) -> str:
    bits = []
    for key in ("id", "original_id", "scene_id", "question_type"):
        if key in entry:
            bits.append(f"{key}={entry[key]!r}")
    return " ".join(bits)


def check_file(path: Path) -> None:
    entries = load_jsonl(path)
    if not entries:
        print(f"{path.relative_to(ROOT)}: empty file, skipping")
        return

    checked = 0
    flagged: list[tuple[dict, list[str]]] = []
    affected_scenes: set[str] = set()

    for e in entries:
        empty_fields = missing_fields(e)
        if empty_fields is None:
            continue
        checked += 1
        if empty_fields:
            flagged.append((e, empty_fields))
            scene = e.get("scene_id")
            if scene is not None:
                affected_scenes.add(scene)

    print(f"{path.relative_to(ROOT)}")
    print(f"  {len(flagged)} / {checked} checked entries have an absent description")
    print(f"  {len(affected_scenes)} distinct scenes affected")
    for entry, empty_fields in flagged:
        print(f"    [{', '.join(empty_fields)} absent] {describe(entry)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Check jsonl files for absent action-phase descriptions (read-only)")
    p.add_argument("files", nargs="*", metavar="PATH",
                   help="jsonl files to check (default: all dataset/annotation jsonl files)")
    args = p.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        paths = sorted({p for pattern in DEFAULT_GLOBS for p in ROOT.glob(pattern)})

    for path in paths:
        check_file(path)
        print()


if __name__ == "__main__":
    main()
