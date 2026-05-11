"""
Compute random-guess baselines per question type over the HF test split.

For each question type (from QUESTION_TYPE_TEMPLATES in config.py):
  - Random accuracy: expected score when guessing uniformly at random
                     = mean(1 / n_choices) over all questions of that type
  - Always-Yes / Always-No accuracy: for question types where choices contain
    both 'Yes' and 'No', what score you'd get by always answering one of them

Images are never decoded — only text columns are fetched.

Usage:
    cd src
    python ../scripts/random_baseline.py
    python ../scripts/random_baseline.py --task failure_mode
    python ../scripts/random_baseline.py --source keplerccc/Robo2VLM-1
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import QUESTION_TYPE_GROUPS, QUESTION_TYPE_TEMPLATES


# ---------------------------------------------------------------------------
# Classification (mirrors dataset.py but runs on text only)
# ---------------------------------------------------------------------------

def _classify(question: str) -> tuple[Optional[str], list[str]]:
    for task, types in QUESTION_TYPE_TEMPLATES.items():
        matched = [
            t for t, pats in types.items()
            if any(re.search(p, question, re.IGNORECASE) for p in pats)
        ]
        if matched:
            return task, matched
    return None, []


def _has_yn(choices: list[str]) -> bool:
    lower = {c.lower() for c in choices}
    return "yes" in lower and "no" in lower


def _yn_indices(choices: list[str]) -> tuple[int, int]:
    yes_idx = next(i for i, c in enumerate(choices) if c.lower() == "yes")
    no_idx  = next(i for i, c in enumerate(choices) if c.lower() == "no")
    return yes_idx, no_idx


# ---------------------------------------------------------------------------
# Stats accumulator
# ---------------------------------------------------------------------------

class TypeStats:
    def __init__(self) -> None:
        self.total: int = 0
        self.random_sum: float = 0.0   # sum of 1/n_choices
        self.yn_total: int = 0
        self.yes_correct: int = 0
        self.no_correct: int = 0

    def update(self, choices: list[str], correct_idx: int) -> None:
        self.total += 1
        self.random_sum += 1.0 / len(choices)
        if _has_yn(choices):
            yes_i, no_i = _yn_indices(choices)
            self.yn_total += 1
            if correct_idx == yes_i:
                self.yes_correct += 1
            if correct_idx == no_i:
                self.no_correct += 1

    @property
    def random_acc(self) -> float:
        return self.random_sum / self.total if self.total else 0.0

    @property
    def always_yes_acc(self) -> Optional[float]:
        return self.yes_correct / self.yn_total if self.yn_total else None

    @property
    def always_no_acc(self) -> Optional[float]:
        return self.no_correct / self.yn_total if self.yn_total else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="keplerccc/Robo2VLM-1")
    p.add_argument("--split", default="test")
    p.add_argument("--task", choices=list(QUESTION_TYPE_TEMPLATES), default=None,
                   help="Restrict to one task (default: all tasks)")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N matched rows (for quick tests)")
    return p.parse_args()


def _load_text_rows(source: str, split: str):
    """Stream text-only columns from HF dataset; never decode images."""
    from datasets import load_dataset as hf_load
    ds = hf_load(source, split=split, streaming=True)
    text_cols = [c for c in ds.column_names if c != "image"]
    ds = ds.select_columns(text_cols)
    yield from ds


def main() -> None:
    args = parse_args()

    stats: dict[str, TypeStats] = defaultdict(TypeStats)
    group_stats: dict[str, TypeStats] = defaultdict(TypeStats)

    processed = matched = 0
    print(f"Streaming {args.split} split from {args.source} …", flush=True)

    for row in _load_text_rows(args.source, args.split):
        processed += 1
        if processed % 5000 == 0:
            print(f"  {processed:,} rows scanned, {matched:,} matched …", flush=True)
        if args.limit and matched >= args.limit:
            break

        task, qtypes = _classify(row["question"])
        if task is None:
            continue
        if args.task and task != args.task:
            continue

        try:
            choices = ast.literal_eval(row["choices"])
            choices = [str(c) for c in choices]
            correct = int(row["correct_answer"])
        except Exception:
            continue

        for qt in qtypes:
            stats[qt].update(choices, correct)
            group = QUESTION_TYPE_GROUPS.get(qt)
            if group:
                group_stats[group].update(choices, correct)

        matched += 1

    print(f"\nDone. {processed:,} rows scanned, {matched:,} matched.\n")

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    COL = 32
    print(f"{'Question type':<{COL}}  {'N':>6}  {'Random%':>8}  {'AlwaysYes%':>11}  {'AlwaysNo%':>10}")
    print("-" * (COL + 42))

    # Group by task for readability
    for task in QUESTION_TYPE_TEMPLATES:
        if args.task and task != args.task:
            continue
        task_types = [t for t in QUESTION_TYPE_TEMPLATES[task] if t in stats]
        if not task_types:
            continue
        print(f"\n[{task}]")
        for qt in task_types:
            s = stats[qt]
            yn_yes = f"{s.always_yes_acc * 100:>10.1f}%" if s.always_yes_acc is not None else f"{'—':>11}"
            yn_no  = f"{s.always_no_acc  * 100:>9.1f}%"  if s.always_no_acc  is not None else f"{'—':>10}"
            print(f"  {qt:<{COL}}  {s.total:>6,}  {s.random_acc * 100:>7.1f}%  {yn_yes}  {yn_no}")

    print(f"\n{'Group':<{COL}}  {'N':>6}  {'Random%':>8}  {'AlwaysYes%':>11}  {'AlwaysNo%':>10}")
    print("-" * (COL + 42))
    for group, s in sorted(group_stats.items()):
        yn_yes = f"{s.always_yes_acc * 100:>10.1f}%" if s.always_yes_acc is not None else f"{'—':>11}"
        yn_no  = f"{s.always_no_acc  * 100:>9.1f}%"  if s.always_no_acc  is not None else f"{'—':>10}"
        print(f"  {group:<{COL}}  {s.total:>6,}  {s.random_acc * 100:>7.1f}%  {yn_yes}  {yn_no}")

    print()


if __name__ == "__main__":
    main()
