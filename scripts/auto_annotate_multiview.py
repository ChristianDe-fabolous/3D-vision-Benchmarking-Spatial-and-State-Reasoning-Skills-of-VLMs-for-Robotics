#!/usr/bin/env python3
"""
Auto-annotate multiview entries by matching question text patterns.

  "successfully completed" questions  → 4-tile split
  "closest to the red dot" questions  → 2-tile split (left/right views)

Adds matched entries to data/annotations.jsonl under "multiview".
Existing annotations are not overwritten.

Usage:
  python scripts/auto_annotate_multiview.py
  python scripts/auto_annotate_multiview.py --scenes-file scenes.txt
  python scripts/auto_annotate_multiview.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PATTERNS = {
    "4": ["successfully completed"],
    "2": ["closest to the red dot"],
}


def match_split(question: str) -> str | None:
    q = question.lower()
    for split_mode, phrases in PATTERNS.items():
        if any(p in q for p in phrases):
            return split_mode
    return None


def load_scene_entries(path: Path) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_annotations(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                a = json.loads(line)
                result[a["scene_id"]] = a
    return result


def write_annotations(path: Path, annotations: dict):
    with open(path, "w", encoding="utf-8") as f:
        for a in annotations.values():
            f.write(json.dumps(a) + "\n")


def main():
    p = argparse.ArgumentParser(description="Auto-annotate multiview entries by question text")
    p.add_argument("--scenes-dir",  default="scenes",                 metavar="DIR")
    p.add_argument("--scenes-file", default=None,                     metavar="PATH")
    p.add_argument("--output",      default="data/annotations.jsonl", metavar="PATH")
    p.add_argument("--dry-run",     action="store_true")
    args = p.parse_args()

    root       = Path(__file__).parent.parent
    scenes_dir = Path(args.scenes_dir) if Path(args.scenes_dir).is_absolute() else root / args.scenes_dir
    out_path   = Path(args.output)     if Path(args.output).is_absolute()      else root / args.output

    if args.scenes_file:
        sf = Path(args.scenes_file) if Path(args.scenes_file).is_absolute() else root / args.scenes_file
        wanted = {l.strip() for l in sf.read_text().splitlines() if l.strip()}

        def _sid(path: Path) -> str | None:
            try:
                first = path.read_text(encoding="utf-8").split("\n")[0].strip()
                return json.loads(first).get("scene_id") if first else None
            except Exception:
                return None

        scene_files = [f for f in scenes_dir.glob("scene_*.jsonl") if _sid(f) in wanted]
    else:
        scene_files = list(scenes_dir.glob("scene_*.jsonl"))

    if not scene_files:
        print(f"No scene files found in {scenes_dir}")
        sys.exit(1)

    annotations = load_annotations(out_path)
    total_added = 0

    for scene_file in sorted(scene_files):
        entries = load_scene_entries(scene_file)
        if not entries:
            continue

        scene_id = entries[0].get("scene_id", scene_file.stem.removeprefix("scene_"))
        ann = annotations.setdefault(scene_id, {"scene_id": scene_id, "subtask": [], "multiview": []})
        existing_ids = {e["id"] for e in ann.get("multiview", [])} | {e["id"] for e in ann.get("subtask", [])}

        added = []
        for entry in entries:
            eid = entry["id"]
            if eid in existing_ids:
                continue
            split_mode = match_split(entry.get("question", ""))
            if split_mode:
                added.append((eid, entry.get("image_path", ""), split_mode))

        if added:
            print(f"\n{scene_id}")
            for eid, image_path, split_mode in added:
                print(f"  + {eid}  [split={split_mode}]")
                if not args.dry_run:
                    ann["multiview"].append({"id": eid, "image_path": image_path, "split_mode": split_mode})
            total_added += len(added)

    if not args.dry_run and total_added:
        write_annotations(out_path, annotations)
        print(f"\nSaved — {total_added} entries added to {out_path}")
    elif args.dry_run:
        print(f"\nDry run — {total_added} entries would be added")
    else:
        print("No matching entries found.")


if __name__ == "__main__":
    main()
