"""
Merge all JSONL chunk files produced by list_questions_with_answers_chunked.py into a
single formatted text file (same format as list_questions_with_answers.py).

Reads:  analysis/chunks/<split>/chunk_*.jsonl
Writes: analysis/questions_with_answers_<split>.txt

Usage:
    python scripts/merge_questions_with_answers_chunks.py
    python scripts/merge_questions_with_answers_chunks.py --split test
    python scripts/merge_questions_with_answers_chunks.py --output my_output.txt
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train",
                        help="Dataset split whose chunks to merge (default: train)")
    parser.add_argument("--output", default=None,
                        help="Output text file (default: analysis/questions_with_answers_<split>.txt)")
    args = parser.parse_args()

    chunk_dir = Path(__file__).parent.parent / "dataset_analysis" / "chunks" / args.split
    if not chunk_dir.exists():
        print(f"No chunk directory found at {chunk_dir}")
        return

    chunk_files = sorted(chunk_dir.glob("chunk_*.jsonl"))
    if not chunk_files:
        print(f"No chunk files found in {chunk_dir}")
        return

    print(f"Merging {len(chunk_files)} chunk file(s) from {chunk_dir}...")

    q_counts: dict[str, int] = defaultdict(int)
    q_choices: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for chunk_file in chunk_files:
        print(f"  reading {chunk_file.name}", flush=True)
        with chunk_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                question = entry["question"]
                q_counts[question] += entry["count"]
                for choices_key, count in entry["choices"].items():
                    q_choices[question][choices_key] += count

    print(f"Aggregated {len(q_counts)} distinct questions. Formatting output...")

    state_path = chunk_dir / "state.json"
    total_scanned = "unknown"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        total_scanned = state["scanned"]

    lines = [f"Scanned {total_scanned} samples — {len(q_counts)} distinct questions\n"]
    lines.append("=" * 60)

    for question, total in sorted(q_counts.items(), key=lambda x: -x[1]):
        lines.append(f"\n[{total:>4}x]  {question}")
        sorted_choices = sorted(q_choices[question].items(), key=lambda x: -x[1])
        if len(sorted_choices) > 5:
            for choices_key, count in sorted_choices[:4]:
                lines.append(f"        [{count:>4}x]  {choices_key}")
            others = sum(c for _, c in sorted_choices[4:])
            lines.append(f"        [{others:>4}x]  (others)")
        else:
            for choices_key, count in sorted_choices:
                lines.append(f"        [{count:>4}x]  {choices_key}")

    output = "\n".join(lines)
    print(output)

    out_path = Path(args.output) if args.output else Path(__file__).parent.parent / "dataset_analysis" / f"questions_with_answers_{args.split}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
