#!/usr/bin/env python3
"""
Download images for all index entries matching a given question type.

Reads data/index.jsonl for matching entries (metadata already there), then
streams the HuggingFace dataset and saves images for matching IDs.

Output:
  data/by_type/<qtype>/entries.jsonl   — entries + image_path, viewable via view_scene.py
  data/by_type/<qtype>/images/<id>.jpg — image files

Usage:
  python scripts/fetch_by_qtype.py --type grasp_phase_next_IP
  python scripts/fetch_by_qtype.py --type grasp_phase_next_IP --split test
  python scripts/fetch_by_qtype.py --type grasp_phase_next_IP --hf-token hf_...

Resume: already-fetched IDs are skipped, so re-running appends only new rows.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HF_DATASET = "keplerccc/Robo2VLM-1"


def load_matching(index_path: Path, qtype: str, split: str) -> dict[str, dict]:
    """One entry per scene — lowest q-number that matches the type."""
    best: dict[str, tuple[int, dict]] = {}  # scene_id -> (q_num, entry)
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("split") != split:
                continue
            if qtype not in e.get("question_types", []):
                continue
            sid = e.get("scene_id") or e["id"]
            m = re.search(r"_q(\d+)$", e["id"])
            q = int(m.group(1)) if m else 999
            if sid not in best or q < best[sid][0]:
                best[sid] = (q, e)
    return {v[1]["id"]: v[1] for v in best.values()}


def main():
    p = argparse.ArgumentParser(description="Fetch images for a specific question type")
    p.add_argument("--type", required=True, metavar="QTYPE",
                   help="question type key (e.g. grasp_phase_next_IP)")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--hf-token", metavar="TOKEN")
    p.add_argument("--index", default="data/index.jsonl", metavar="PATH",
                   help="index JSONL path (default: data/index.jsonl)")
    args = p.parse_args()

    try:
        from datasets import load_dataset as hf_load
        from PIL import Image
    except ImportError:
        print("Missing: pip install datasets pillow", file=sys.stderr)
        sys.exit(1)

    project_root = Path(__file__).parent.parent
    index_path = (project_root / args.index) if not Path(args.index).is_absolute() else Path(args.index)

    print(f"Loading {args.type!r} entries from index...", flush=True)
    by_id = load_matching(index_path, args.type, args.split)
    if not by_id:
        print(f"No entries found for type={args.type!r} split={args.split!r}")
        sys.exit(0)
    print(f"Found {len(by_id)} entries in index")

    out_dir = project_root / "data" / "by_type" / args.type
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    out_jsonl = out_dir / "entries.jsonl"

    # Resume: skip already-fetched IDs
    fetched_ids: set[str] = set()
    if out_jsonl.exists():
        with open(out_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    fetched_ids.add(json.loads(line)["id"])
    remaining = {eid: e for eid, e in by_id.items() if eid not in fetched_ids}
    if not remaining:
        print(f"All entries already fetched. View with:\n  python scripts/view_scene.py {out_jsonl}")
        return
    if fetched_ids:
        print(f"Resuming — {len(fetched_ids)} already fetched, {len(remaining)} to go")

    load_kwargs = {}
    if args.hf_token:
        load_kwargs["token"] = args.hf_token

    streaming = args.split != "test"
    ds = hf_load(HF_DATASET, split=args.split, streaming=streaming, **load_kwargs)

    written = 0
    scanned = 0
    t0 = time.time()

    with open(out_jsonl, "a", encoding="utf-8") as f:
        for row in ds:
            if not remaining:
                break

            scanned += 1
            row_id = row["id"]

            if row_id not in remaining:
                if scanned % 2000 == 0:
                    elapsed = time.time() - t0
                    rate = written / elapsed if elapsed > 0 else 0
                    print(f"\r  scanned {scanned}, {written}/{len(by_id)} fetched — {rate:.1f}/s", end="", flush=True)
                continue

            entry = remaining.pop(row_id)

            image_path = None
            try:
                img: Image.Image = row["image"].convert("RGB")
                safe_id = row_id.replace("/", "_")
                img_file = images_dir / f"{safe_id}.jpg"
                img.save(img_file, "JPEG", quality=90)
                image_path = str(img_file)
            except Exception as e:
                print(f"\n  [WARN] image save failed for {row_id}: {e}")

            out_entry = {**entry, "image_path": image_path}
            f.write(json.dumps(out_entry) + "\n")
            f.flush()
            written += 1

            elapsed = time.time() - t0
            rate = written / elapsed if elapsed > 0 else 0
            print(f"\r  scanned {scanned}, {written}/{len(by_id)} fetched — {rate:.1f}/s", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\rDone. {written} images saved in {elapsed / 60:.1f}min  ({scanned} rows scanned)")
    if remaining:
        print(f"  WARNING: {len(remaining)} IDs not found in dataset")
    print(f"View with:\n  python scripts/view_scene.py {out_jsonl}")


if __name__ == "__main__":
    main()
