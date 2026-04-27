#!/usr/bin/env python3
"""
Split annotated multiview images into 4 equal tiles (2x2) and produce a JSONL
ready for VLM evaluation — same question fed once per tile.

Reads:
  data/annotations.jsonl     — multiview selections from annotate_scenes.py
  scenes/scene_<id>.jsonl    — question/choices/answer for each entry

Output:
  data/multiview_tiles/images/<id>_<pos>.jpg   — individual tiles
  data/multiview_tiles/entries.jsonl           — one entry per tile

Tile positions: top_left, top_right, bottom_left, bottom_right

Usage:
  python scripts/tile_multiview.py
  python scripts/tile_multiview.py --annotations data/annotations.jsonl
  python scripts/tile_multiview.py --scenes-dir scenes/ --output-dir data/multiview_tiles
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import re

from PIL import Image

TILE_POSITIONS_4 = ["top_left", "top_right", "bottom_left", "bottom_right"]

_MAX_SCENE_FNAME = 120


def _safe_fname(scene_id: str) -> str:
    if len(scene_id) <= _MAX_SCENE_FNAME:
        return scene_id
    m = re.search(r"_(\d+)$", scene_id)
    suffix = f"_{m.group(1)}" if m else ""
    return scene_id[: _MAX_SCENE_FNAME - len(suffix)] + suffix
TILE_POSITIONS_2 = ["left", "right"]

# Images with width/height >= this are treated as two views stitched side-by-side
WIDE_ASPECT_THRESHOLD = 3.0


def load_annotations(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_scene_entry(scenes_dir: Path, scene_id: str, entry_id: str) -> dict | None:
    path = scenes_dir / f"scene_{_safe_fname(scene_id)}.jsonl"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e["id"] == entry_id:
                return e
    return None


def split_tiles(img: Image.Image) -> list[tuple[str, Image.Image]]:
    w, h = img.size
    if w / h >= WIDE_ASPECT_THRESHOLD:
        # Side-by-side views: split left/right only
        mw = w // 2
        boxes = [(0, 0, mw, h), (mw, 0, w, h)]
        return list(zip(TILE_POSITIONS_2, [img.crop(b) for b in boxes]))
    else:
        mw, mh = w // 2, h // 2
        boxes = [
            (0,   0,   mw,  mh),
            (mw,  0,   w,   mh),
            (0,   mh,  mw,  h),
            (mw,  mh,  w,   h),
        ]
        return list(zip(TILE_POSITIONS_4, [img.crop(b) for b in boxes]))


def load_image(image_path: str) -> Image.Image | None:
    path = Path(image_path)
    if not path.exists():
        return None
    try:
        return Image.open(io.BytesIO(path.read_bytes())).convert("RGB")
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description="Tile multiview images 2x2 for VLM eval")
    p.add_argument("--annotations", default="data/annotations.jsonl", metavar="PATH")
    p.add_argument("--scenes-dir", default="scenes", metavar="DIR")
    p.add_argument("--output-dir", default="data/multiview_tiles", metavar="DIR")
    args = p.parse_args()

    project_root = Path(__file__).parent.parent
    ann_path = Path(args.annotations) if Path(args.annotations).is_absolute() else project_root / args.annotations
    scenes_dir = Path(args.scenes_dir) if Path(args.scenes_dir).is_absolute() else project_root / args.scenes_dir
    out_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else project_root / args.output_dir

    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    out_jsonl = out_dir / "entries.jsonl"

    annotations = load_annotations(ann_path)
    if not annotations:
        print(f"No annotations found at {ann_path}")
        sys.exit(1)

    # Load already-written tile IDs for resume
    written: set[str] = set()
    if out_jsonl.exists():
        with open(out_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    written.add(json.loads(line)["tile_id"])

    total_mv = sum(len(a.get("multiview", [])) for a in annotations)
    print(f"{len(annotations)} annotated scenes, {total_mv} multiview entries")
    if written:
        print(f"Resuming — {len(written)} tiles already written")

    written_count = 0
    skipped = 0

    with open(out_jsonl, "a", encoding="utf-8") as out_f:
        for ann in annotations:
            scene_id = ann["scene_id"]
            for mv in ann.get("multiview", []):
                entry_id = mv["id"]

                # Get question/choices from scene JSONL
                scene_entry = load_scene_entry(scenes_dir, scene_id, entry_id)
                if scene_entry is None:
                    print(f"  [WARN] entry {entry_id} not found in scene JSONL — skipping")
                    skipped += 1
                    continue

                # Load image
                img = load_image(mv.get("image_path", ""))
                if img is None:
                    print(f"  [WARN] image missing for {entry_id} — skipping")
                    skipped += 1
                    continue

                tiles = split_tiles(img)
                positions = [pos for pos, _ in tiles]

                # Skip if all tiles for this entry already written
                if all(f"{entry_id}_{pos}" in written for pos in positions):
                    continue

                for pos, tile in tiles:
                    tile_id = f"{entry_id}_{pos}"
                    if tile_id in written:
                        continue

                    safe_id = entry_id.replace("/", "_")
                    tile_path = images_dir / f"{safe_id}_{pos}.jpg"
                    tile.save(tile_path, "JPEG", quality=95, subsampling=0)

                    entry = {
                        "tile_id": tile_id,
                        "original_id": entry_id,
                        "scene_id": scene_id,
                        "tile_position": pos,
                        "split_mode": "2" if len(positions) == 2 else "4",
                        "image_path": str(tile_path),
                        "task": scene_entry.get("task"),
                        "question": scene_entry.get("question"),
                        "choices": scene_entry.get("choices"),
                        "correct_answer": scene_entry.get("correct_answer"),
                        "correct_choice": scene_entry.get("correct_choice"),
                        "question_types": scene_entry.get("question_types", []),
                        "question_group": scene_entry.get("question_group"),
                    }
                    out_f.write(json.dumps(entry) + "\n")
                    out_f.flush()
                    written.add(tile_id)
                    written_count += 1
                    print(f"  {tile_id}")

    print(f"\nDone. {written_count} tiles written ({skipped} entries skipped)")
    print(f"Output: {out_jsonl}")


if __name__ == "__main__":
    main()
