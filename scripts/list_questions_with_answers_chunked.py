"""
Chunked version of list_questions_with_answers.py.

Writes a JSONL delta file every --checkpoint rows so that a crash does not
lose all progress. Each chunk file contains only the data seen in that chunk.
A state file tracks how many rows have been scanned so that the script can
resume automatically by skipping already-processed rows on the next run.

Chunk files:  analysis/chunks/<split>/chunk_<start>_<end>.jsonl
State file:   analysis/chunks/<split>/state.json

Each JSONL line:
    {"question": "...", "choices": {"[\"No\", \"Yes\"]": 42, ...}, "last_id": "..."}

Run merge_questions_with_answers_chunks.py afterwards to combine chunks into a final text file.

Usage:
    python scripts/list_questions_with_answers_chunked.py
    python scripts/list_questions_with_answers_chunked.py --checkpoint 50000
    python scripts/list_questions_with_answers_chunked.py --split test
    python scripts/list_questions_with_answers_chunked.py --local-data /path/to/data
    python scripts/list_questions_with_answers_chunked.py --fresh   # ignore existing state, start over
"""

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import _is_invalid


def _parse_choices(raw: str):
    parsed = ast.literal_eval(raw)
    return tuple(sorted(str(c) for c in parsed))


def _write_chunk(chunk_dir: Path, start: int, end: int, q_counts: dict, q_choices: dict, last_id: str):
    chunk_path = chunk_dir / f"chunk_{start:08d}_{end:08d}.jsonl"
    with chunk_path.open("w") as f:
        for question, count in q_counts.items():
            choices = {str(list(k)): v for k, v in q_choices[question].items()}
            f.write(json.dumps({"question": question, "count": count, "choices": choices, "last_id": last_id}) + "\n")
    return chunk_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=int, default=100000,
                        help="Write a chunk file every N rows (default: 100000)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples to scan (default: all)")
    parser.add_argument("--split", default="train",
                        help="Dataset split to use (default: train)")
    parser.add_argument("--local-data", default=None,
                        help="Local dataset directory (skips HuggingFace)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing state and start from scratch")
    args = parser.parse_args()

    chunk_dir = Path(__file__).parent.parent / "outputs" / "chunks" / args.split
    chunk_dir.mkdir(parents=True, exist_ok=True)
    state_path = chunk_dir / "state.json"

    # Resume from existing state unless --fresh
    skip = 0
    if not args.fresh and state_path.exists():
        state = json.loads(state_path.read_text())
        skip = state["scanned"]
        print(f"Resuming from row {skip} (last id: {state['last_id']})")

    from datasets import load_dataset as hf_load

    if args.local_data:
        ds = hf_load("parquet", data_dir=args.local_data, split=args.split, streaming=True).select_columns(["id", "question", "choices"])
    else:
        ds = hf_load("keplerccc/Robo2VLM-1", split=args.split, streaming=True).select_columns(["id", "question", "choices"])

    if skip:
        print(f"Skipping {skip} rows...", flush=True)
        ds = ds.skip(skip)

    # Accumulators for the current chunk
    q_counts: dict[str, int] = defaultdict(int)
    q_choices: dict[str, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))

    scanned = skip
    chunk_start = skip
    last_id = ""

    for row in ds:
        if args.limit is not None and scanned >= args.limit:
            break
        scanned += 1

        if scanned % 1000 == 0:
            print(f"  processed {scanned}...", flush=True)

        last_id = row["id"]

        if _is_invalid(last_id):
            continue

        try:
            choices = _parse_choices(row["choices"])
        except Exception:
            continue

        question = row["question"]
        q_counts[question] += 1
        q_choices[question][choices] += 1

        if scanned % args.checkpoint == 0:
            path = _write_chunk(chunk_dir, chunk_start, scanned, q_counts, q_choices, last_id)
            print(f"  -> chunk saved: {path.name}", flush=True)
            state_path.write_text(json.dumps({"scanned": scanned, "last_id": last_id}))
            # Reset accumulators for next chunk
            q_counts = defaultdict(int)
            q_choices = defaultdict(lambda: defaultdict(int))
            chunk_start = scanned

    # Write remaining rows as final chunk
    if q_counts:
        path = _write_chunk(chunk_dir, chunk_start, scanned, q_counts, q_choices, last_id)
        print(f"  -> final chunk saved: {path.name}", flush=True)

    state_path.write_text(json.dumps({"scanned": scanned, "last_id": last_id}))
    print(f"\nDone. Scanned {scanned} rows total. Chunks in: {chunk_dir}")
    print("Run merge_questions_chunks.py to produce the final text file.")


if __name__ == "__main__":
    main()
    import os
    os._exit(0)
