from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


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
    }
