"""
Scan the dataset and report two problem categories:

1. MULTI-TYPE OVERLAP — questions that matched more than one question type
   within their task (per QUESTION_TYPES in src/config.py).

2. UNTYPED BUT INCLUDED — questions that matched a task via the TASK_KEYWORDS
   fallback but matched no question type, meaning they appear in runs without
   a type assignment.

Each entry is listed with its id, question text, answer choices, and correct answer.

Usage:
    python scripts/list_question_type_overlaps.py
    python scripts/list_question_type_overlaps.py --split test
    python scripts/list_question_type_overlaps.py --limit 10000
    python scripts/list_question_type_overlaps.py --local-data /path/to/data
"""

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import _classify_task_and_types, _is_invalid


def _parse_choices(raw: str):
    return [str(c) for c in ast.literal_eval(raw)]


def _fmt_entry(entry_id, question, choices, correct_answer):
    correct_label = choices[correct_answer] if 0 <= correct_answer < len(choices) else "?"
    lines = [
        f"  id:      {entry_id}",
        f"  q:       {question}",
        f"  choices: {choices}",
        f"  answer:  [{correct_answer}] {correct_label}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples to scan (default: all)")
    parser.add_argument("--split", default="train",
                        help="Dataset split to use (default: train)")
    parser.add_argument("--local-data", default=None,
                        help="Local dataset directory (skips HuggingFace)")
    parser.add_argument("--output", default=None,
                        help="Output file (default: analysis/question_type_overlaps_<split>.txt)")
    args = parser.parse_args()

    from datasets import load_dataset as hf_load

    if args.local_data:
        ds = hf_load("parquet", data_dir=args.local_data, split=args.split, streaming=True).select_columns(["id", "question", "choices", "correct_answer"])
    else:
        ds = hf_load("keplerccc/Robo2VLM-1", split=args.split, streaming=True).select_columns(["id", "question", "choices", "correct_answer"])

    # types_tuple -> list of (id, question, choices, correct_answer)
    overlaps: dict[tuple, list] = defaultdict(list)
    # task -> list of (id, question, choices, correct_answer)
    untyped: dict[str, list] = defaultdict(list)

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

        task, types = _classify_task_and_types(row["question"])
        if task is None:
            continue

        entry = (row["id"], row["question"], choices, int(row["correct_answer"]))

        if len(types) > 1:
            overlaps[tuple(sorted(types))].append(entry)
        elif len(types) == 0:
            untyped[task].append(entry)

    lines = [f"Scanned {scanned} samples\n"]
    lines.append("=" * 60)

    # --- Section 1: multi-type overlaps ---
    total_overlap_entries = sum(len(v) for v in overlaps.values())
    lines.append(f"\n### SECTION 1: MULTI-TYPE OVERLAP ({len(overlaps)} type combinations, {total_overlap_entries} entries)\n")

    if overlaps:
        for types_combo, entries in sorted(overlaps.items(), key=lambda x: -len(x[1])):
            lines.append(f"\n[{len(entries):>4}x]  types: {list(types_combo)}")
            for entry in entries:
                lines.append("")
                lines.append(_fmt_entry(*entry))
    else:
        lines.append("  (none)")

    # --- Section 2: untyped but included ---
    lines.append("\n\n" + "=" * 60)
    total_untyped = sum(len(v) for v in untyped.values())
    lines.append(f"\n### SECTION 2: UNTYPED BUT INCLUDED IN RUNS ({total_untyped} entries across {len(untyped)} tasks)\n")

    if untyped:
        for task, entries in sorted(untyped.items()):
            lines.append(f"\n-- task: {task} ({len(entries)} entries) --")
            for entry in entries:
                lines.append("")
                lines.append(_fmt_entry(*entry))
    else:
        lines.append("  (none)")

    output = "\n".join(lines)
    print(output)

    out_path = Path(args.output) if args.output else Path(__file__).parent.parent / "dataset_analysis" / f"question_type_overlaps_{args.split}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
    import os
    os._exit(0)
