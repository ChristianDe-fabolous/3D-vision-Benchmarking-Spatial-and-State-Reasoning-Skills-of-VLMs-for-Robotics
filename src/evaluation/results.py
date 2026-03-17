"""
Saves run outputs to disk.

outputs/<run_id>/
  config.json       — exact run configuration
  results.jsonl     — one JSON line per sample (written live by SampleLogger)
  summary.json      — final aggregated metrics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from evaluation.metrics import accuracy_by_field, summarize


def save_config(output_dir: Path, config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def save_summary(output_dir: Path, results: List[dict]) -> None:
    summary = summarize(results)
    summary["by_task"] = accuracy_by_field(results, "task")
    summary["by_augmented"] = accuracy_by_field(results, "augmented_reverse")
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"\nResults: {summary['correct']}/{summary['total']} correct "
        f"({summary['accuracy']:.1%}) — "
        f"{summary['wrong']} wrong, {summary['unparseable']} unparseable"
    )
