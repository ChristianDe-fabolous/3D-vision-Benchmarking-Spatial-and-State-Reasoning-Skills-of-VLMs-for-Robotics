"""
List all distinct question templates in the dataset.

Streams the dataset so no full download is needed. Use --limit to cap how
many samples are scanned (more samples = more complete template list).

Usage:
    python scripts/list_questions.py
    python scripts/list_questions.py --limit 2000
    python scripts/list_questions.py --local-data /path/to/data

The output shows each unique question template and how many times it appears.
Use this to decide which questions belong to each task, then fill in
ALLOWED_QUESTION_PATTERNS in src/config.py.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import _is_invalid


def _to_template(question: str) -> str:
    """Normalise a question by stripping task/object-specific prefixes."""
    q = re.sub(r"^[^?]+(has the robot|did the robot|which configuration|in the (left|right|ext\d) image)", r"\1", question, flags=re.IGNORECASE)
    return q.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5000,
                        help="Max samples to scan (default 5000)")
    parser.add_argument("--local-data", default=None,
                        help="Local dataset directory (skips HuggingFace)")
    parser.add_argument("--output", default=None,
                        help="Save results to a file (default: outputs/question_templates.txt)")
    args = parser.parse_args()

    from datasets import load_dataset as hf_load

    if args.local_data:
        ds = hf_load("parquet", data_dir=args.local_data, split="test", streaming=True)
    else:
        ds = hf_load("keplerccc/Robo2VLM-1", split="test", streaming=True)

    counts: dict[str, int] = defaultdict(int)
    examples: dict[str, str] = {}

    scanned = 0
    for row in ds:
        if scanned >= args.limit:
            break
        scanned += 1

        if _is_invalid(row["id"]):
            continue

        template = _to_template(row["question"])
        counts[template] += 1
        if template not in examples:
            examples[template] = row["question"]

    lines = [f"Scanned {scanned} samples — {len(counts)} distinct templates\n"]
    lines.append("=" * 60)

    for template, count in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"\n  [{count:>4}x]  {template}")
        lines.append(f"          e.g. {examples[template][:120]}")

    output = "\n".join(lines)
    print(output)

    out_path = Path(args.output) if args.output else Path(__file__).parent.parent / "outputs" / "question_templates.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
    # Force exit to avoid PyArrow GIL cleanup crash on streaming dataset teardown
    import os
    os._exit(0)
