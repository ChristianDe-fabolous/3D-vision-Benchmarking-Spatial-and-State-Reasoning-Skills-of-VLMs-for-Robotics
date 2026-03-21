"""
List all distinct question templates with their occurrence counts, and below
each question all distinct answer choice sets that appear with it (also counted).

Answer choice sets are sorted alphabetically within each set (tuples of sorted
choices). Choice sets below each question are sorted by descending count.
Questions are sorted by descending count.

Usage:
    python scripts/list_questions_with_answers.py
    python scripts/list_questions_with_answers.py --limit 2000
    python scripts/list_questions_with_answers.py --local-data /path/to/data
"""

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import _is_invalid


def _parse_choices(raw: str):
    parsed = ast.literal_eval(raw)
    return tuple(sorted(str(c) for c in parsed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples to scan (default: all)")
    parser.add_argument("--split", default="train",
                        help="Dataset split to use (default: train)")
    parser.add_argument("--local-data", default=None,
                        help="Local dataset directory (skips HuggingFace)")
    parser.add_argument("--output", default=None,
                        help="Save results to a file (default: analysis/questions_with_answers_<split>.txt)")
    args = parser.parse_args()

    from datasets import load_dataset as hf_load

    if args.local_data:
        ds = hf_load("parquet", data_dir=args.local_data, split=args.split, streaming=True).select_columns(["id", "question", "choices"])
    else:
        ds = hf_load("keplerccc/Robo2VLM-1", split=args.split, streaming=True).select_columns(["id", "question", "choices"])

    # template -> total count
    q_counts: dict[str, int] = defaultdict(int)
    # template -> choices_tuple -> count
    q_choices: dict[str, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))

    scanned = 0
    for row in ds:
        if args.limit is not None and scanned >= args.limit:
            break
        scanned += 1

        if scanned % 1000 == 0:
            print(f"  processed {scanned}...", flush=True)

        if _is_invalid(row["id"]):
            continue

        try:
            choices = _parse_choices(row["choices"])
        except Exception:
            continue

        question = row["question"]
        q_counts[question] += 1
        q_choices[question][choices] += 1

    lines = [f"Scanned {scanned} samples — {len(q_counts)} distinct question templates\n"]
    lines.append("=" * 60)

    for question, total in sorted(q_counts.items(), key=lambda x: -x[1]):
        lines.append(f"\n[{total:>4}x]  {question}")
        sorted_choices = sorted(q_choices[question].items(), key=lambda x: -x[1])
        if len(sorted_choices) > 5:
            for choices, count in sorted_choices[:4]:
                lines.append(f"        [{count:>4}x]  {list(choices)}")
            others = sum(c for _, c in sorted_choices[4:])
            lines.append(f"        [{others:>4}x]  (others)")
        else:
            for choices, count in sorted_choices:
                lines.append(f"        [{count:>4}x]  {list(choices)}")

    output = "\n".join(lines)
    print(output)

    out_path = Path(args.output) if args.output else Path(__file__).parent.parent / "outputs" / f"questions_with_answers_{args.split}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
    import os
    os._exit(0)
