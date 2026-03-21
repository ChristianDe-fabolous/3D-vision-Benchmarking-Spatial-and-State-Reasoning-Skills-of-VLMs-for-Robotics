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

from config import SCENE_MIN_QUESTIONS, SCENE_OUTLIER_STD
from evaluation.metrics import (
    accuracy_by_field,
    answer_category_analysis,
    answer_distribution_analysis,
    question_type_analysis,
    scene_analysis,
    summarize,
)


def save_config(output_dir: Path, config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def save_type_issues(output_dir: Path, results: List[dict]) -> None:
    """
    Write a report of two problem categories to question_type_issues.txt:
    1. Questions that matched multiple question types.
    2. Questions that matched no question type but were still included in the run.
    """
    multi: list[dict] = [r for r in results if len(r.get("question_types", [])) > 1]
    untyped: list[dict] = [r for r in results if not r.get("question_type")]

    def fmt(r: dict) -> str:
        return (
            f"  id:      {r['entry_id']}\n"
            f"  q:       {r['question']}\n"
            f"  choices: {r['choices']}\n"
            f"  answer:  [{r['ground_truth_index']}] {r['ground_truth_label']}"
        )

    lines = ["### SECTION 1: MULTI-TYPE OVERLAP\n"]
    if multi:
        for r in multi:
            lines.append(f"\n  types: {r['question_types']}")
            lines.append(fmt(r))
    else:
        lines.append("  (none)")

    lines.append("\n\n### SECTION 2: UNTYPED BUT INCLUDED\n")
    if untyped:
        for r in untyped:
            lines.append("")
            lines.append(fmt(r))
    else:
        lines.append("  (none)")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "question_type_issues.txt").write_text("\n".join(lines))

    if multi or untyped:
        print(f"  Type issues: {len(multi)} multi-type, {len(untyped)} untyped — see question_type_issues.txt")


def save_summary(output_dir: Path, results: List[dict], analyse_categories: bool = False) -> None:
    summary = summarize(results)
    summary["by_task"] = accuracy_by_field(results, "task")
    summary["scene_analysis"] = scene_analysis(results, SCENE_MIN_QUESTIONS, SCENE_OUTLIER_STD)

    # Question-type breakdowns — only populated when QUESTION_TYPES is configured
    if any(r.get("question_type") for r in results):
        summary["by_question_type"] = accuracy_by_field(results, "question_type")
        summary["question_type_analysis"] = question_type_analysis(results, SCENE_OUTLIER_STD)

    summary["answer_distribution"] = answer_distribution_analysis(results)

    if analyse_categories:
        summary["answer_category_analysis"] = answer_category_analysis(results)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    save_type_issues(output_dir, results)

    print(
        f"\nResults: {summary['correct']}/{summary['total']} correct "
        f"({summary['accuracy']:.1%}) — "
        f"{summary['wrong']} wrong, {summary['unparseable']} unparseable"
    )
    sa = summary["scene_analysis"]
    print(
        f"Scenes: {sa['included_scenes']} analysed, "
        f"{sa['excluded_scenes']} excluded (<{SCENE_MIN_QUESTIONS} questions, "
        f"{sa['excluded_questions']} questions dropped)"
    )
