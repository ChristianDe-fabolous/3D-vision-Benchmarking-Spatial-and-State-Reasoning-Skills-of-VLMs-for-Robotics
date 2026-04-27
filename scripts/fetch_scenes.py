#!/usr/bin/env python3
"""
Fetch all images for a list of DROID scenes.

Reads data/index.jsonl for all entries belonging to the requested scenes,
streams the HuggingFace dataset, saves images, and writes one JSONL per scene.

Output:
  scenes/scene_<scene_id>.jsonl    — all entries for that scene + image_path
  scenes/images/<id>.jpg           — image files

Usage:
  python scripts/fetch_scenes.py --scenes-file scenes_combined.txt
  python scripts/fetch_scenes.py --scenes-file scenes_diverse.txt
  python scripts/fetch_scenes.py --hf-token hf_...

Resume: already-fetched IDs are skipped, so re-running continues where it stopped.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HF_DATASET = "keplerccc/Robo2VLM-1"
_MAX_FNAME  = 120


def _safe_fname(scene_id: str) -> str:
    if len(scene_id) <= _MAX_FNAME:
        return scene_id
    m = re.search(r"_(\d+)$", scene_id)
    suffix = f"_{m.group(1)}" if m else ""
    return scene_id[: _MAX_FNAME - len(suffix)] + suffix


def load_scene_ids(scenes_file: Path) -> list[str]:
    return [l.strip() for l in scenes_file.read_text().splitlines() if l.strip()]


def load_index_entries(index_path: Path, scene_ids: set[str]) -> dict[str, dict]:
    """Return {entry_id: entry} for all entries belonging to the requested scenes."""
    by_id: dict[str, dict] = {}
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("scene_id") in scene_ids:
                by_id[e["id"]] = e
    return by_id


def _load_written_ids(scenes_dir: Path, scene_ids: set[str]) -> set[str]:
    written: set[str] = set()
    for sid in scene_ids:
        path = scenes_dir / f"scene_{_safe_fname(sid)}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        written.add(json.loads(line)["id"])
                    except Exception:
                        pass
    return written


def main():
    p = argparse.ArgumentParser(description="Fetch scene images from HuggingFace")
    p.add_argument("--scenes-file", required=True, metavar="PATH",
                   help="plain text file with one scene ID per line")
    p.add_argument("--scenes-dir", default="scenes",          metavar="DIR")
    p.add_argument("--index",      default="data/index.jsonl", metavar="PATH")
    p.add_argument("--split",      default="train",            choices=["train", "test"])
    p.add_argument("--hf-token",   metavar="TOKEN")
    args = p.parse_args()

    try:
        from datasets import load_dataset as hf_load
        from PIL import Image
    except ImportError:
        print("Missing: pip install datasets pillow", file=sys.stderr)
        sys.exit(1)

    root       = Path(__file__).parent.parent
    sf         = Path(args.scenes_file) if Path(args.scenes_file).is_absolute() else root / args.scenes_file
    index_path = Path(args.index)       if Path(args.index).is_absolute()       else root / args.index
    scenes_dir = Path(args.scenes_dir)  if Path(args.scenes_dir).is_absolute()  else root / args.scenes_dir
    images_dir = scenes_dir / "images"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    scene_ids = load_scene_ids(sf)
    if not scene_ids:
        print(f"No scene IDs found in {sf}")
        sys.exit(1)
    print(f"{len(scene_ids)} scenes requested")

    print(f"Loading entries from index...", flush=True)
    by_id = load_index_entries(index_path, set(scene_ids))
    print(f"  {len(by_id)} entries found in index")

    written_ids = _load_written_ids(scenes_dir, set(scene_ids))
    remaining   = {eid: e for eid, e in by_id.items() if eid not in written_ids}
    if not remaining:
        print("All entries already fetched.")
        return
    if written_ids:
        print(f"Resuming — {len(written_ids)} already fetched, {len(remaining)} to go")

    load_kwargs = {}
    if args.hf_token:
        load_kwargs["token"] = args.hf_token

    streaming = args.split != "test"
    ds = hf_load(HF_DATASET, split=args.split, streaming=streaming, **load_kwargs)

    # Open one file handle per scene (lazy)
    handles: dict[str, object] = {}

    def _fh(scene_id: str):
        if scene_id not in handles:
            path = scenes_dir / f"scene_{_safe_fname(scene_id)}.jsonl"
            handles[scene_id] = open(path, "a", encoding="utf-8")
        return handles[scene_id]

    written = 0
    scanned = 0
    t0 = time.time()

    try:
        for row in ds:
            if not remaining:
                break

            scanned += 1
            row_id = row["id"]

            if row_id not in remaining:
                if scanned % 2000 == 0:
                    elapsed = time.time() - t0
                    rate = written / elapsed if elapsed > 0 else 0
                    print(f"\r  scanned {scanned}, {written}/{len(by_id)} fetched — {rate:.1f}/s",
                          end="", flush=True)
                continue

            entry = remaining.pop(row_id)

            image_path = None
            try:
                img: Image.Image = row["image"].convert("RGB")
                safe_id  = row_id.replace("/", "_")
                img_file = images_dir / f"{safe_id}.jpg"
                img.save(img_file, "JPEG", quality=95, subsampling=0)
                image_path = str(img_file)
            except Exception as exc:
                print(f"\n  [WARN] image save failed for {row_id}: {exc}")

            fh = _fh(entry["scene_id"])
            fh.write(json.dumps({**entry, "image_path": image_path}) + "\n")
            fh.flush()
            written += 1

            elapsed = time.time() - t0
            rate = written / elapsed if elapsed > 0 else 0
            print(f"\r  scanned {scanned}, {written}/{len(by_id)} fetched — {rate:.1f}/s",
                  end="", flush=True)
    finally:
        for fh in handles.values():
            fh.close()

    elapsed = time.time() - t0
    print(f"\rDone. {written} images saved in {elapsed / 60:.1f}min  ({scanned} rows scanned)")
    if remaining:
        print(f"  WARNING: {len(remaining)} IDs not found in dataset stream")


if __name__ == "__main__":
    main()
