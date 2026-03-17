"""
List all distinct question templates in the dataset, grouped by task.

Streams the dataset so no full download is needed. Use --limit to cap how
many samples are scanned (more samples = more complete template list).

Usage:
    python scripts/list_questions.py
    python scripts/list_questions.py --limit 2000
    python scripts/list_questions.py --local-data /path/to/data
    python scripts/list_questions.py --task multiview

The output shows each unique question template and how many times it appears.
Copy the relevant substrings into ALLOWED_QUESTION_PATTERNS in src/config.py.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import _classify_task, _is_invalid, _parse_choices


def _to_template(question: str) -> str:
    """
    Normalise a question to a template by replacing task-specific nouns
    (object names, task descriptions) with placeholders.
    Keeps the structural wording so similar questions group together.
    """
    # Strip the task/object prefix that appears before common question phrases
    q = re.sub(r"^[^?]+(has the robot|did the robot|which configuration|in the (left|right|ext\d) image)", r"\1", question, flags=re.IGNORECASE)
    return q.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5000,
                        help="Max samples to scan (default 5000)")
    parser.add_argument("--task", default=None,
                        help="Filter to a specific task (failure_mode or multiview)")
    parser.add_argument("--local-data", default=None,
                        help="Local dataset directory (skips HuggingFace)")
    args = parser.parse_args()

    from datasets import load_dataset as hf_load

    if args.local_data:
        ds = hf_load("parquet", data_dir=args.local_data, split="test", streaming=True)
    else:
        ds = hf_load("keplerccc/Robo2VLM-1", split="test", streaming=True)

    # task -> template -> count
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # template -> one example full question
    examples: dict[str, str] = {}

    scanned = 0
    for row in ds:
        if scanned >= args.limit:
            break
        scanned += 1

        if _is_invalid(row["id"]):
            continue

        task = _classify_task(row["question"])
        if task is None:
            continue
        if args.task and task != args.task:
            continue

        template = _to_template(row["question"])
        counts[task][template] += 1
        if template not in examples:
            examples[template] = row["question"]

    print(f"\nScanned {scanned} samples\n")

    for task, templates in sorted(counts.items()):
        print(f"{'='*60}")
        print(f"TASK: {task}  ({len(templates)} distinct templates)")
        print(f"{'='*60}")
        for template, count in sorted(templates.items(), key=lambda x: -x[1]):
            print(f"\n  [{count:>4}x]  {template}")
            print(f"          e.g. {examples[template][:120]}")
        print()


if __name__ == "__main__":
    main()
