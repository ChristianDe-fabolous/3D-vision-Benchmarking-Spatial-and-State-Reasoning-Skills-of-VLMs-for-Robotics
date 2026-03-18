from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple


def accuracy(results: List[dict]) -> float:
    if not results:
        return 0.0
    return sum(r["correct"] for r in results) / len(results)


def accuracy_by_field(results: List[dict], field: str) -> Dict[str, float]:
    """
    Compute accuracy broken down by an arbitrary metadata field.
    E.g. field="task" or field="augmented_reverse".
    """
    buckets: Dict[str, List[bool]] = defaultdict(list)
    for r in results:
        key = str(r.get(field, "unknown"))
        buckets[key].append(r["correct"])
    return {k: sum(v) / len(v) for k, v in buckets.items()}


def random_baseline(results: List[dict]) -> float:
    """Expected accuracy of random guessing, accounting for variable choice counts."""
    if not results:
        return 0.0
    return sum(1 / len(r["choices"]) for r in results) / len(results)


def scene_analysis(results: List[dict], min_questions: int, outlier_std: float) -> dict:
    """Per-scene accuracy analysis with outlier flagging."""
    buckets: Dict[str, List[bool]] = defaultdict(list)
    for r in results:
        scene_id = r.get("scene_id")
        if scene_id:
            buckets[scene_id].append(r["correct"])

    total_scenes = len(buckets)
    excluded_scenes = {s for s, v in buckets.items() if len(v) < min_questions}
    excluded_questions = sum(len(buckets[s]) for s in excluded_scenes)
    included = {s: v for s, v in buckets.items() if s not in excluded_scenes}

    if not included:
        return {
            "total_scenes": total_scenes,
            "excluded_scenes": len(excluded_scenes),
            "excluded_questions": excluded_questions,
            "included_scenes": 0,
            "outliers_above": [],
            "outliers_below": [],
        }

    accs = {s: sum(v) / len(v) for s, v in included.items()}
    mean = sum(accs.values()) / len(accs)
    std = math.sqrt(sum((a - mean) ** 2 for a in accs.values()) / len(accs))

    outliers_above = sorted(
        [{"scene_id": s, "accuracy": round(a, 3), "n": len(included[s])}
         for s, a in accs.items() if a > mean + outlier_std * std],
        key=lambda x: -x["accuracy"],
    )
    outliers_below = sorted(
        [{"scene_id": s, "accuracy": round(a, 3), "n": len(included[s])}
         for s, a in accs.items() if a < mean - outlier_std * std],
        key=lambda x: x["accuracy"],
    )

    return {
        "total_scenes": total_scenes,
        "excluded_scenes": len(excluded_scenes),
        "excluded_questions": excluded_questions,
        "included_scenes": len(included),
        "mean_accuracy": round(mean, 3),
        "std": round(std, 3),
        "outlier_std_threshold": outlier_std,
        "outliers_above": outliers_above,
        "outliers_below": outliers_below,
    }


def answer_category_analysis(results: List[dict]) -> List[dict]:
    """Per-answer-category accuracy (grouped by the set of choices, order-independent)."""
    buckets: Dict[Tuple, List[bool]] = defaultdict(list)
    for r in results:
        key = tuple(sorted(r["choices"]))
        buckets[key].append(r["correct"])

    return sorted(
        [
            {
                "choices": list(key),
                "total": len(vals),
                "correct": sum(vals),
                "accuracy": round(sum(vals) / len(vals), 3),
                "random_baseline": round(1 / len(key), 3),
            }
            for key, vals in buckets.items()
        ],
        key=lambda x: -x["total"],
    )


def summarize(results: List[dict]) -> dict:
    total = len(results)
    correct = sum(r["correct"] for r in results)
    unparseable = sum(r["predicted_index"] is None for r in results)
    wrong = total - correct - unparseable
    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "unparseable": unparseable,
        "accuracy": accuracy(results),
        "random_baseline": random_baseline(results),
    }
