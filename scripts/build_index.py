#!/usr/bin/env python3
"""
Download all dataset rows (no images) into a local JSONL index.

Each line is a JSON object with parsed metadata — use this to find and filter
scenes before fetching them with their images via fetch_scenes.py.

Usage:
  python scripts/build_index.py                        # train split
  python scripts/build_index.py --split test
  python scripts/build_index.py --split train test     # both
  python scripts/build_index.py --output data/index.jsonl

Resume: already-seen IDs are skipped, so re-running appends only new rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import QUESTION_TYPE_GROUPS
from data.dataset import (
    _classify_task_and_types_template,
    _extract_question_parts,
    _is_invalid,
    _parse_choices,
    _parse_scene_id,
)

HF_DATASET = "keplerccc/Robo2VLM-1"


def _load_seen_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["id"])
            except Exception:
                pass
    return seen


def stream_split(split: str, seen_ids: set[str], out_path: Path) -> int:
    from datasets import load_dataset

    streaming = split != "test"
    ds = load_dataset(
        HF_DATASET,
        split=split,
        streaming=streaming,
        columns=["id", "question", "choices", "correct_answer"],
    )

    written = 0
    skipped_seen = 0
    skipped_invalid = 0
    t0 = time.time()

    with open(out_path, "a", encoding="utf-8") as f:
        for i, row in enumerate(ds, 1):
            row_id = row["id"]

            if row_id in seen_ids:
                skipped_seen += 1
                continue

            if _is_invalid(row_id):
                skipped_invalid += 1
                continue

            task, question_types = _classify_task_and_types_template(row["question"])
            if task is None:
                print(f"\n  [UNMATCHED] {row['id']}: {row['question'][:120]!r}")
                scene_id = _parse_scene_id(row_id)
                entry = {
                    "id": row_id,
                    "split": split,
                    "task": None,
                    "question": row["question"],
                    "choices": [],
                    "correct_answer": None,
                    "correct_choice": None,
                    "scene_id": scene_id,
                    "question_types": [],
                    "question_group": None,
                }
                f.write(json.dumps(entry) + "\n")
                written += 1
                seen_ids.add(row_id)
                continue

            try:
                choices = _parse_choices(row["choices"])
            except Exception:
                continue

            scene_id = _parse_scene_id(row_id)
            parts = _extract_question_parts(row["question"], question_types)

            entry = {
                "id": row_id,
                "split": split,
                "task": task,
                "question": row["question"],
                "choices": choices,
                "correct_answer": int(row["correct_answer"]),
                "correct_choice": choices[int(row["correct_answer"])],
                "scene_id": scene_id,
                "question_types": question_types,
                "question_group": QUESTION_TYPE_GROUPS.get(question_types[0]),
                **({"question_parts": parts} if parts else {}),
            }

            f.write(json.dumps(entry) + "\n")
            written += 1
            seen_ids.add(row_id)

            if i % 500 == 0:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(
                    f"\r  [{split}] {i:>7} scanned, {written:>6} written"
                    f" — {rate:5.0f}/s — {elapsed/60:.1f}min",
                    end="",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(
        f"\r  [{split}] done: {written} written, {skipped_seen} already seen,"
        f" {skipped_invalid} invalid — {elapsed/60:.1f}min"
    )
    return written


def main():
    p = argparse.ArgumentParser(description="Build local metadata index (no images)")
    p.add_argument("--split", nargs="+", default=["train"],
                   choices=["train", "test"],
                   help="splits to download (default: train)")
    p.add_argument("--output", default="data/index.jsonl", metavar="PATH",
                   help="output JSONL path (default: data/index.jsonl)")
    args = p.parse_args()

    project_root = Path(__file__).parent.parent
    out_path = (project_root / args.output) if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Index: {out_path}")
    seen_ids = _load_seen_ids(out_path)
    if seen_ids:
        print(f"Resuming — {len(seen_ids)} IDs already in index")

    total = 0
    for split in args.split:
        total += stream_split(split, seen_ids, out_path)

    print(f"\nTotal written: {total} — index at {out_path}")


if __name__ == "__main__":
    main()
