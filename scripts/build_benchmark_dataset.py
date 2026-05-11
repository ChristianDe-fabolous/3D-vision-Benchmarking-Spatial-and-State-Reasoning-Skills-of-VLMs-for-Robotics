#!/usr/bin/env python3
"""
Build a train/test benchmark dataset from scenes in merged_annotations.jsonl.

1. Reads data/multiview_tiles/merged_annotations.jsonl → unique scene IDs.
2. Filters data/index.jsonl to entries in those scenes with the requested question types.
3. Splits scenes into train/test (no leakage — whole scenes go to one split).
4. Streams keplerccc/Robo2VLM-1 from HuggingFace to download images.
5. Writes data/benchmark/train.jsonl and data/benchmark/test.jsonl.

Usage:
  python scripts/build_benchmark_dataset.py
  python scripts/build_benchmark_dataset.py --n-scenes 50 --train-ratio 0.8
  python scripts/build_benchmark_dataset.py --question-types task_success_TS-S,relative_depth_SU
  python scripts/build_benchmark_dataset.py --hf-token hf_...

Resume: already-saved image files are skipped on re-run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

ANNOTATIONS_PATH = ROOT / "data" / "multiview_tiles" / "merged_annotations.jsonl"
INDEX_PATH       = ROOT / "data" / "index.jsonl"
HF_DATASET       = "keplerccc/Robo2VLM-1"

ALL_QUESTION_TYPES = [
    "action_direction_IP",
    "cross_view_correspondence_MV",
    "goal_configuration_TS-GL",
    "relative_depth_SU",
    "task_success_TS-S",
    "trajectory_understanding_TU",
]


def load_scene_ids(n_scenes: int | None) -> list[str]:
    seen: dict[str, None] = {}
    with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = d.get("scene_id", "")
            if sid:
                seen[sid] = None
    scenes = list(seen)
    if n_scenes is not None:
        scenes = scenes[:n_scenes]
    return scenes


def load_index_entries(scene_ids: set[str], question_types: set[str]) -> dict[str, list[dict]]:
    """Return {hf_id: [index_entry, ...]} for matching entries."""
    by_hf_id: dict[str, list[dict]] = {}
    with open(INDEX_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("scene_id") not in scene_ids:
                continue
            entry_qtypes = set(e.get("question_types", []))
            if not entry_qtypes.intersection(question_types):
                continue
            hf_id = e["id"]
            by_hf_id.setdefault(hf_id, []).append(e)
    return by_hf_id


def split_scenes(scenes: list[str], train_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    shuffled = list(scenes)
    rng.shuffle(shuffled)
    cutoff = int(len(shuffled) * train_ratio)
    return set(shuffled[:cutoff]), set(shuffled[cutoff:])


def write_entry(fh, entry: dict, image_path: str | None) -> None:
    fh.write(json.dumps({**entry, "image_path": image_path}) + "\n")
    fh.flush()


def main() -> None:
    p = argparse.ArgumentParser(description="Build train/test benchmark from annotated scenes")
    p.add_argument("--annotations", default=str(ANNOTATIONS_PATH), metavar="PATH",
                   help="merged_annotations.jsonl (default: %(default)s)")
    p.add_argument("--index",       default=str(INDEX_PATH),       metavar="PATH")
    p.add_argument("--output-dir",  default="data/benchmark",      metavar="DIR")
    p.add_argument("--n-scenes",    type=int, default=None,        metavar="N",
                   help="limit to first N scenes from annotations (default: all)")
    p.add_argument("--question-types", default=",".join(ALL_QUESTION_TYPES), metavar="TYPES",
                   help="comma-separated question type keys (default: all)")
    p.add_argument("--train-ratio", type=float, default=0.8,       metavar="R")
    p.add_argument("--seed",        type=int,   default=42,        metavar="S")
    p.add_argument("--hf-split",    default="train",               choices=["train", "test"],
                   help="which HF dataset split to stream from")
    p.add_argument("--hf-token",    metavar="TOKEN")
    args = p.parse_args()

    try:
        from datasets import load_dataset as hf_load
        from PIL import Image
    except ImportError:
        print("Missing: pip install datasets pillow", file=sys.stderr)
        sys.exit(1)

    out_dir    = Path(args.output_dir) if Path(args.output_dir).is_absolute() else ROOT / args.output_dir
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    question_types = {qt.strip() for qt in args.question_types.split(",") if qt.strip()}
    print(f"Question types: {sorted(question_types)}")

    # --- Load scenes ---
    print("Loading scene IDs from annotations...", flush=True)
    all_scenes = load_scene_ids(args.n_scenes)
    print(f"  {len(all_scenes)} scenes")

    train_scenes, test_scenes = split_scenes(all_scenes, args.train_ratio, args.seed)
    print(f"  {len(train_scenes)} train scenes, {len(test_scenes)} test scenes")

    # --- Index lookup ---
    print("Filtering index entries...", flush=True)
    by_hf_id = load_index_entries(set(all_scenes), question_types)
    if not by_hf_id:
        print("No matching entries found in index. Check --question-types.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(by_hf_id)} HF entries to fetch ({sum(len(v) for v in by_hf_id.values())} total rows)")

    # --- Load already-written entries to allow resume ---
    def _already_written(path: Path) -> set[str]:
        written: set[str] = set()
        if not path.exists():
            return written
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        written.add(json.loads(line)["id"])
                    except Exception:
                        pass
        return written

    train_path = out_dir / "train.jsonl"
    test_path  = out_dir / "test.jsonl"

    written_train = _already_written(train_path)
    written_test  = _already_written(test_path)
    written_all   = written_train | written_test

    remaining = {hid: entries for hid, entries in by_hf_id.items() if hid not in written_all}
    if not remaining:
        print("All entries already written. Done.")
        return
    if written_all:
        print(f"Resuming — {len(written_all)} already done, {len(remaining)} left")

    # --- Stream HF ---
    load_kwargs: dict = {}
    if args.hf_token:
        load_kwargs["token"] = args.hf_token

    streaming = args.hf_split != "test"
    print(f"Streaming {HF_DATASET} ({args.hf_split})...", flush=True)
    ds = hf_load(HF_DATASET, split=args.hf_split, streaming=streaming, **load_kwargs)

    fh_train = open(train_path, "a", encoding="utf-8")
    fh_test  = open(test_path,  "a", encoding="utf-8")

    written = 0
    scanned = 0
    t0 = time.time()

    try:
        for row in ds:
            if not remaining:
                break

            scanned += 1
            hf_id = row["id"]

            if hf_id not in remaining:
                if scanned % 5000 == 0:
                    elapsed = time.time() - t0
                    rate = written / elapsed if elapsed > 0 else 0
                    print(f"\r  scanned {scanned:,}, {written}/{len(by_hf_id)} fetched — {rate:.1f}/s",
                          end="", flush=True)
                continue

            entries = remaining.pop(hf_id)

            # Save image once per HF row
            img_file = images_dir / f"{hf_id.replace('/', '_')}.jpg"
            image_path: str | None = None
            if img_file.exists():
                image_path = str(img_file)
            else:
                try:
                    img: Image.Image = row["image"].convert("RGB")
                    img.save(img_file, "JPEG", quality=95, subsampling=0)
                    image_path = str(img_file)
                except Exception as exc:
                    print(f"\n  [WARN] image save failed for {hf_id}: {exc}")

            for entry in entries:
                scene_id = entry["scene_id"]
                fh = fh_train if scene_id in train_scenes else fh_test
                write_entry(fh, entry, image_path)
                written += 1

            elapsed = time.time() - t0
            rate = written / elapsed if elapsed > 0 else 0
            print(f"\r  scanned {scanned:,}, {written}/{sum(len(v) for v in by_hf_id.values())} fetched — {rate:.1f}/s",
                  end="", flush=True)
    finally:
        fh_train.close()
        fh_test.close()

    elapsed = time.time() - t0
    print(f"\rDone. {written} entries in {elapsed / 60:.1f}min  ({scanned:,} rows scanned)")

    if remaining:
        print(f"  WARNING: {len(remaining)} HF IDs not found in stream — may be in opposite split")

    # --- Summary ---
    def _count(path: Path) -> int:
        if not path.exists():
            return 0
        with open(path) as f:
            return sum(1 for l in f if l.strip())

    print(f"\nOutput in {out_dir}/")
    print(f"  train.jsonl : {_count(train_path):>6} entries")
    print(f"  test.jsonl  : {_count(test_path):>6} entries")
    print(f"  images/     : {len(list(images_dir.glob('*.jpg'))):>6} images")


if __name__ == "__main__":
    main()
