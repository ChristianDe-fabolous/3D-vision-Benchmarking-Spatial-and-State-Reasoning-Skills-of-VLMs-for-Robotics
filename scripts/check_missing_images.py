#!/usr/bin/env python3
"""
Check which dataset entries have missing image files.

Usage
-----
  python scripts/check_missing_images.py
  python scripts/check_missing_images.py --dataset data/action_phase_dataset.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/action_phase_dataset.jsonl")
    args = p.parse_args()

    root = Path(__file__).parent.parent
    path = Path(args.dataset) if Path(args.dataset).is_absolute() else root / args.dataset

    entries = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    missing_by_type: dict[str, list[dict]] = defaultdict(list)
    total_missing = 0

    for e in entries:
        bad_imgs = [img for img in e.get("images", []) if not (root / img).exists()]
        if bad_imgs:
            missing_by_type[e.get("question_type", "?")].append({
                "id":    e.get("id"),
                "scene": e.get("scene_id", "")[:60],
                "imgs":  bad_imgs,
            })
            total_missing += 1

    if not total_missing:
        print(f"All {len(entries)} entries have valid image paths.")
        return

    print(f"{total_missing} / {len(entries)} entries have missing images:\n")
    for qt, items in sorted(missing_by_type.items()):
        print(f"  {qt}: {len(items)} entries")
        for item in items:
            print(f"    id={item['id']}  scene={item['scene']}")
            for img in item["imgs"]:
                print(f"      missing: {img}")


if __name__ == "__main__":
    main()
