#!/usr/bin/env python3
"""
Cap "No" answers in action_phase_dataset.jsonl per (scene_id, question_type).

"Yes" and "Cannot be determined" entries are always kept.
"No" entries are randomly downsampled to --no-cap per group.

Usage:
  python scripts/cap_dataset.py
  python scripts/cap_dataset.py --no-cap 5
  python scripts/cap_dataset.py --input data/action_phase_dataset.jsonl \
                                 --output data/action_phase_dataset_capped.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",   default="data/action_phase_dataset.jsonl")
    p.add_argument("--output",  default="data/action_phase_dataset_capped.jsonl")
    p.add_argument("--no-cap",  type=int, default=10,
                   help="Max 'No' answers per (scene_id, question_type) group (default: 10)")
    p.add_argument("--seed",    type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    root     = Path(__file__).parent.parent
    in_path  = root / args.input
    out_path = root / args.output

    entries = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    # Group "No" entries by (scene_id, question_type)
    no_groups: dict[tuple, list[int]] = defaultdict(list)
    keep = []

    for i, e in enumerate(entries):
        if e.get("answer_text") == "No":
            key = (e["scene_id"], e["question_type"])
            no_groups[key].append(i)
        else:
            keep.append(i)

    # Sample "No" entries down to no-cap per group
    kept_no = 0
    dropped  = 0
    for key, indices in no_groups.items():
        if len(indices) <= args.no_cap:
            keep.extend(indices)
            kept_no += len(indices)
        else:
            sampled = random.sample(indices, args.no_cap)
            keep.extend(sampled)
            kept_no  += args.no_cap
            dropped  += len(indices) - args.no_cap

    keep.sort()
    kept_entries = [entries[i] for i in keep]

    # Re-assign ids
    for i, e in enumerate(kept_entries):
        e["id"] = i

    with open(out_path, "w", encoding="utf-8") as f:
        for e in kept_entries:
            f.write(json.dumps(e) + "\n")

    by_type:   dict[str, int] = defaultdict(int)
    by_answer: dict[str, int] = defaultdict(int)
    for e in kept_entries:
        by_type[e["question_type"]] += 1
        by_answer[e.get("answer_text", "?")] += 1

    print(f"Input:   {len(entries)} entries")
    print(f"Dropped: {dropped} 'No' entries (cap={args.no_cap} per group)")
    print(f"Output:  {len(kept_entries)} entries → {out_path}")
    print()
    print("By question type:")
    for qt, cnt in sorted(by_type.items()):
        print(f"  {qt}: {cnt}")
    print("\nBy answer:")
    for ans, cnt in sorted(by_answer.items()):
        print(f"  {ans}: {cnt}")


if __name__ == "__main__":
    main()
