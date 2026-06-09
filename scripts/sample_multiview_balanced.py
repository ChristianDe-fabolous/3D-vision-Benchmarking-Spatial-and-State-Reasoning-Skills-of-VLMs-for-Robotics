#!/usr/bin/env python3
"""
Down-sample the multiview consistency dataset to a target size while keeping
scene representation as uniform as possible.

Approach: per-scene cap sampling. We search for the smallest per-scene cap C
such that sum(min(count(scene), C) for scene in scenes) >= target. Scenes with
fewer than C entries keep all of them; scenes with more are randomly
sub-sampled down to C. This naturally pulls small scenes up to "full
representation" and large scenes down — flattening the distribution without
dropping any scene entirely.

Usage:
  python scripts/sample_multiview_balanced.py
  python scripts/sample_multiview_balanced.py --target 6500 --seed 0 \
      --input data/multiview_consistency_dataset.jsonl \
      --output data/multiview_consistency_dataset_balanced.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent

DEFAULT_INPUT  = "data/multiview_consistency_dataset.jsonl"
DEFAULT_OUTPUT = "data/multiview_consistency_dataset_balanced.jsonl"
DEFAULT_TARGET = 6500
DEFAULT_SEED   = 0


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def find_cap(counts: list[int], target: int) -> int:
    """Smallest per-scene cap whose sum(min(count, cap)) reaches `target`."""
    for cap in range(1, max(counts) + 1):
        if sum(min(c, cap) for c in counts) >= target:
            return cap
    return max(counts)


def main() -> None:
    p = argparse.ArgumentParser(description="Down-sample multiview dataset with uniform scene representation")
    p.add_argument("--input",  default=DEFAULT_INPUT,  metavar="PATH")
    p.add_argument("--output", default=DEFAULT_OUTPUT, metavar="PATH")
    p.add_argument("--target", type=int, default=DEFAULT_TARGET,
                   help="approximate target sample count (default: %(default)s)")
    p.add_argument("--seed",   type=int, default=DEFAULT_SEED,
                   help="random seed for reproducible sampling (default: %(default)s)")
    args = p.parse_args()

    in_path  = ROOT / args.input
    out_path = ROOT / args.output
    rng = random.Random(args.seed)

    print(f"Loading {in_path}…")
    entries = load_jsonl(in_path)

    by_scene: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_scene[e["scene_id"]].append(e)

    counts = sorted(len(v) for v in by_scene.values())
    cap = find_cap(counts, args.target)

    print(f"  {len(entries)} entries across {len(by_scene)} scenes")
    print(f"  per-scene counts: min={counts[0]} median={counts[len(counts)//2]} max={counts[-1]}")
    print(f"  chosen per-scene cap: {cap}")

    sampled: list[dict] = []
    kept_counts: Counter[str] = Counter()
    for scene_id, scene_entries in by_scene.items():
        if len(scene_entries) <= cap:
            chosen = scene_entries
        else:
            chosen = rng.sample(scene_entries, cap)
        kept_counts[scene_id] = len(chosen)
        sampled.extend(chosen)

    # Keep a stable, reproducible order: by original id.
    sampled.sort(key=lambda e: e["id"])
    for e in sampled:
        e["source_dataset_id"] = e["id"]
    for i, e in enumerate(sampled):
        e["id"] = i

    kept = sorted(kept_counts.values())
    print()
    print(f"  sampled {len(sampled)} / {len(entries)} entries")
    print(f"  kept-per-scene: min={kept[0]} median={kept[len(kept)//2]} max={kept[-1]}")
    print(f"  scenes fully kept (count <= cap): {sum(1 for c in counts if c <= cap)} / {len(counts)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in sampled:
            f.write(json.dumps(e) + "\n")

    print(f"\nWritten {len(sampled)} entries → {out_path}")


if __name__ == "__main__":
    main()
