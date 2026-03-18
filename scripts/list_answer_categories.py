"""
List all distinct answer categories in the dataset.

An answer category is the exact set of choices for a question,
e.g. ('Yes', 'No') or ('Yes', 'No', 'Cannot be determined').

For each category, shows:
  - How many questions have this category
  - Up to 100 example questions from different scenes

Usage:
    python scripts/list_answer_categories.py
    python scripts/list_answer_categories.py --limit 5000
    python scripts/list_answer_categories.py --examples 50
    python scripts/list_answer_categories.py --no-examples
    python scripts/list_answer_categories.py --local-data /path/to/data
"""

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import _is_invalid


def _parse_choices(raw: str):
    parsed = ast.literal_eval(raw)
    return tuple(str(c) for c in parsed)


def _parse_scene_id(entry_id: str):
    m = re.search(r'_(\d+)_q\d+$', entry_id)
    return m.group(1) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5000,
                        help="Max samples to scan (default 5000)")
    parser.add_argument("--examples", type=int, default=100,
                        help="Max example questions per category from different scenes (default 100)")
    parser.add_argument("--no-examples", action="store_true",
                        help="Only show counts, skip example questions")
    parser.add_argument("--local-data", default=None,
                        help="Local dataset directory (skips HuggingFace)")
    parser.add_argument("--output", default=None,
                        help="Save output to file (default: outputs/answer_categories.txt)")
    args = parser.parse_args()

    from datasets import load_dataset as hf_load

    if args.local_data:
        ds = hf_load("parquet", data_dir=args.local_data, split="test", streaming=True)
    else:
        ds = hf_load("keplerccc/Robo2VLM-1", split="test", streaming=True)

    # category -> count
    counts: dict[tuple, int] = defaultdict(int)
    # category -> {scene_id -> example question text}
    scene_examples: dict[tuple, dict[str, str]] = defaultdict(dict)

    scanned = 0
    for row in ds:
        if scanned >= args.limit:
            break
        scanned += 1

        if _is_invalid(row["id"]):
            continue

        try:
            choices = _parse_choices(row["choices"])
        except Exception:
            continue

        key = tuple(sorted(choices))
        counts[key] += 1

        if not args.no_examples:
            scene_id = _parse_scene_id(row["id"]) or row["id"]
            examples = scene_examples[key]
            if len(examples) < args.examples and scene_id not in examples:
                examples[scene_id] = row["question"]

    lines = [f"Scanned {scanned} samples — {len(counts)} distinct answer categories\n"]
    lines.append("=" * 60)

    for choices, count in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"\nCategory: {list(choices)}  —  {count} questions")
        if not args.no_examples:
            for scene_id, question in scene_examples[choices].items():
                lines.append(f"  [scene {scene_id}] {question[:120]}")

    output = "\n".join(lines)
    print(output)

    out_path = Path(args.output) if args.output else Path(__file__).parent.parent / "outputs" / "answer_categories.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
    import os
    os._exit(0)
