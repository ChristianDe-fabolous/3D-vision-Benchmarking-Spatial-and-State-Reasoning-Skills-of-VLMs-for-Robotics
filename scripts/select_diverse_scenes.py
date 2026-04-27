#!/usr/bin/env python3
"""
Select 50 diverse DROID scenes for annotation.

Scoring per scene (higher = more action phases = better for annotation):
  +3 per explicit phase marker ("then", "and_then", "and_finally", "twice")
  +1 per unique verb found in the scene name
  +0.1 per word in the scene name (length proxy)

Selection: iterates verb categories in round-robin order, always picking
the highest-scoring unused scene from the next least-represented category.
This guarantees verb diversity while still preferring complex scenes.

Output:
  scenes.txt              — plain list of 50 scene IDs (paste into fetch_scenes.py)
  scenes_ranked.jsonl     — full scored list for inspection

Usage:
  python scripts/select_diverse_scenes.py
  python scripts/select_diverse_scenes.py --count 50 --index data/index.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Verbs that appear in task names — used to count distinct actions per scene
KNOWN_VERBS = {
    "put", "place", "pick", "remove", "take", "move", "close", "open",
    "fold", "pour", "stack", "cover", "uncover", "hang", "unhang",
    "wipe", "scrub", "push", "pull", "slide", "turn", "flip", "press",
    "switch", "use", "get", "bring", "lift", "drop", "insert", "attach",
    "detach", "rotate", "align", "arrange", "separate", "gather", "sort",
    "stack", "unstack", "scoop", "stir", "spread", "stretch", "unfold",
    "straighten", "wrap", "unwrap", "roll", "rip", "spell", "set",
}

# Phase markers in scene names (underscored form)
PHASE_MARKERS = {"then", "and_then", "and_finally", "twice", "and_put",
                 "and_place", "and_stack", "and_close", "and_open"}


def parse_scene(scene_id: str) -> dict:
    name = scene_id.removeprefix("droid_")
    # strip trailing numeric ID
    name_clean = re.sub(r"_\d+$", "", name)
    words = name_clean.split("_")

    primary_verb = words[0] if words else "unknown"

    # count distinct known verbs
    found_verbs = {w for w in words if w in KNOWN_VERBS}

    # count explicit phase markers
    text = name_clean
    phase_count = sum(1 for m in PHASE_MARKERS if m in text)

    score = phase_count * 3 + len(found_verbs) * 1 + len(words) * 0.1

    return {
        "scene_id": scene_id,
        "primary_verb": primary_verb,
        "verbs": sorted(found_verbs),
        "verb_count": len(found_verbs),
        "phase_markers": phase_count,
        "word_count": len(words),
        "score": round(score, 2),
        "name": name_clean,
    }


def load_droid_scenes(index_path: Path) -> list[dict]:
    seen: set[str] = set()
    scenes = []
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = d.get("scene_id", "")
            if sid.startswith("droid_") and sid not in seen:
                seen.add(sid)
                scenes.append(parse_scene(sid))
    return scenes


def select_diverse(scenes: list[dict], n: int) -> list[dict]:
    """Round-robin across primary verb groups, picking highest-score each turn."""
    by_verb: dict[str, list[dict]] = defaultdict(list)
    for s in scenes:
        by_verb[s["primary_verb"]].append(s)

    # Sort each group descending by score
    for verb in by_verb:
        by_verb[verb].sort(key=lambda x: x["score"], reverse=True)

    # Order verb groups by descending group size (most common first, for balance)
    verb_order = sorted(by_verb.keys(), key=lambda v: -len(by_verb[v]))

    selected = []
    idx = 0
    while len(selected) < n:
        made_progress = False
        for verb in verb_order:
            if len(selected) >= n:
                break
            if by_verb[verb]:
                selected.append(by_verb[verb].pop(0))
                made_progress = True
        if not made_progress:
            break  # exhausted all categories

    return selected


def select_first(scenes: list[dict], n: int) -> list[dict]:
    """Return the first N scenes in index order (no diversity selection)."""
    return scenes[:n]


def select_combined(scenes: list[dict], n_first: int, n_diverse: int) -> tuple[list[dict], list[dict], list[dict]]:
    """
    First n_first scenes from index + diverse extras for verb groups not already covered.
    Returns (first_scenes, diverse_extras, combined).
    """
    first = select_first(scenes, n_first)
    covered_verbs = {s["primary_verb"] for s in first}
    covered_ids   = {s["scene_id"]     for s in first}

    # Diverse selection from full set, then keep only new verbs not in first
    diverse_all = select_diverse(scenes, n_diverse)
    extras = [s for s in diverse_all
              if s["primary_verb"] not in covered_verbs and s["scene_id"] not in covered_ids]

    combined = first + extras
    return first, extras, combined


def _write(path: Path, scenes: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for s in scenes:
            f.write(s["scene_id"] + "\n")


def _write_ranked(path: Path, scenes: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for s in scenes:
            f.write(json.dumps(s) + "\n")


def _print_table(scenes: list[dict], label: str):
    print(f"\n{label}  ({len(scenes)} scenes)")
    print(f"{'Verb':<20} {'Scene':<70} {'Score':>5}")
    print("-" * 98)
    for s in scenes:
        print(f"{s['primary_verb']:<20} {s['name'][:68]:<70} {s['score']:>5}")


def main():
    p = argparse.ArgumentParser(description="Select DROID scenes for annotation")
    p.add_argument("--count",         type=int, default=70,  metavar="N",
                   help="number of scenes for --first or --diverse (default: 70)")
    p.add_argument("--first-count",   type=int, default=70,  metavar="N",
                   help="first-N count for --combined (default: 70)")
    p.add_argument("--diverse-count", type=int, default=70,  metavar="N",
                   help="diverse count for --combined (default: 70)")
    p.add_argument("--index",  default="data/index.jsonl", metavar="PATH")
    p.add_argument("--output", default=None,               metavar="PATH",
                   help="override output .txt path")
    p.add_argument("--ranked", default=None,               metavar="PATH",
                   help="override ranked .jsonl path")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--first",    action="store_true", help="first N scenes in index order")
    mode.add_argument("--diverse",  action="store_true", help="diverse N scenes (default mode)")
    mode.add_argument("--combined", action="store_true",
                      help="first N + diverse extras for uncovered verbs → scenes_combined.txt")
    args = p.parse_args()

    root       = Path(__file__).parent.parent
    index_path = Path(args.index) if Path(args.index).is_absolute() else root / args.index

    print(f"Loading DROID scenes from {index_path}...")
    scenes = load_droid_scenes(index_path)
    print(f"  {len(scenes)} unique DROID scenes found")

    if args.first:
        selected = select_first(scenes, args.count)
        selected_sorted = selected
        out_txt   = root / (args.output or "scenes_first.txt")
        out_jsonl = root / (args.ranked or "scenes_first_ranked.jsonl")
        print(f"  Mode: first {args.count}")
        _print_table(selected_sorted, "First scenes")
        _write(out_txt, selected_sorted)
        _write_ranked(out_jsonl, selected_sorted)
        print(f"\n{len(selected_sorted)} scenes → {out_txt}")

    elif args.combined:
        first, extras, combined = select_combined(scenes, args.first_count, args.diverse_count)
        out_txt      = root / (args.output or "scenes_combined.txt")
        out_jsonl    = root / (args.ranked or "scenes_combined_ranked.jsonl")
        out_first    = root / "scenes_first.txt"
        out_diverse  = root / "scenes_diverse.txt"
        print(f"  Mode: combined  (first={args.first_count}, diverse pool={args.diverse_count})")
        _print_table(first,   "First scenes")
        _print_table(extras,  "Diverse extras (new verbs only)")
        _write(out_first,   first)
        _write(out_diverse, sorted(extras, key=lambda x: (x["primary_verb"], -x["score"])))
        _write(out_txt,     combined)
        _write_ranked(out_jsonl, combined)
        print(f"\n{len(first)} first + {len(extras)} diverse extras = {len(combined)} total → {out_txt}")
        print(f"Also wrote {out_first} and {out_diverse} separately")

    else:  # default: diverse
        selected = select_diverse(scenes, args.count)
        selected_sorted = sorted(selected, key=lambda x: (x["primary_verb"], -x["score"]))
        out_txt   = root / (args.output or "scenes_diverse.txt")
        out_jsonl = root / (args.ranked or "scenes_diverse_ranked.jsonl")
        print(f"  Mode: diverse {args.count}")
        _print_table(selected_sorted, "Diverse scenes")
        _write(out_txt, selected_sorted)
        _write_ranked(out_jsonl, selected_sorted)
        print(f"\n{len(selected_sorted)} scenes → {out_txt}")


if __name__ == "__main__":
    main()
