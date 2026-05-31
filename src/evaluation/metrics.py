from __future__ import annotations

import math
import re
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
        val = r.get(field, "unknown")
        key = str(val[0] if isinstance(val, list) else val)
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


def question_type_analysis(results: List[dict], outlier_std: float) -> dict:
    """Per-question-type accuracy with outlier flagging."""
    buckets: Dict[str, List[bool]] = defaultdict(list)
    for r in results:
        qt = (r.get("question_types") or [None])[0]
        if qt:
            buckets[qt].append(r["correct"])

    if not buckets:
        return {"total_types": 0, "outliers_above": [], "outliers_below": []}

    accs = {qt: sum(v) / len(v) for qt, v in buckets.items()}
    mean = sum(accs.values()) / len(accs)
    std = math.sqrt(sum((a - mean) ** 2 for a in accs.values()) / len(accs))

    all_types = sorted(
        [{"question_type": qt, "accuracy": round(a, 3), "n": len(buckets[qt])}
         for qt, a in accs.items()],
        key=lambda x: -x["accuracy"],
    )
    outliers_above = [t for t in all_types if t["accuracy"] > mean + outlier_std * std]
    outliers_below = sorted(
        [t for t in all_types if t["accuracy"] < mean - outlier_std * std],
        key=lambda x: x["accuracy"],
    )

    return {
        "total_types": len(buckets),
        "mean_accuracy": round(mean, 3),
        "std": round(std, 3),
        "outlier_std_threshold": outlier_std,
        "all_types": all_types,
        "outliers_above": outliers_above,
        "outliers_below": outliers_below,
    }


def scene_analysis_by_question_type(results: List[dict], min_questions: int, outlier_std: float) -> Dict[str, dict]:
    """For each question type, run scene_analysis on that subset of results. Currently unused."""
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        qt = (r.get("question_types") or [None])[0]
        if qt:
            by_type[qt].append(r)
    return {qt: scene_analysis(subset, min_questions, outlier_std) for qt, subset in by_type.items()}


def cross_bucket_scene_analysis(results: List[dict]) -> Dict[str, Dict[str, dict]]:
    """Currently unused."""
    """For each scene, accuracy broken down by question type across all buckets."""
    data: Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        scene_id = r.get("scene_id")
        qt = (r.get("question_types") or [None])[0]
        if scene_id and qt:
            data[scene_id][qt].append(r["correct"])
    return {
        scene_id: {
            qt: {"accuracy": round(sum(v) / len(v), 3), "n": len(v)}
            for qt, v in buckets.items()
        }
        for scene_id, buckets in sorted(data.items())
    }


_GRASP_PHASE_PATTERNS: list[Tuple[re.Pattern, str]] = [
    (re.compile(r"^Approaching the .+ with open gripper$", re.IGNORECASE),
     "Approaching the <object> with open gripper"),
    (re.compile(r"^Closing gripper to grasp the .+$", re.IGNORECASE),
     "Closing gripper to grasp the <object>"),
    (re.compile(r"^Firmly grasping the .+$", re.IGNORECASE),
     "Firmly grasping the <object>"),
    (re.compile(r"^Moving away with open gripper after releasing the .+$", re.IGNORECASE),
     "Moving away with open gripper after releasing the <object>"),
    (re.compile(r"^Releasing the .+ by opening gripper$", re.IGNORECASE),
     "Releasing the <object> by opening gripper"),
]


def _normalize_choice(choice: str) -> str:
    """Replace object-specific substrings so structurally identical choice sets merge."""
    for pattern, replacement in _GRASP_PHASE_PATTERNS:
        if pattern.match(choice):
            return replacement
    return choice


def answer_category_analysis(results: List[dict]) -> List[dict]:
    """
    Per-answer-category accuracy grouped by the *normalized* set of choices.

    Choices that differ only in the object name (e.g. grasp phase questions for
    'drawer' vs 'can') are merged into a single category using <object> placeholders.
    Each category also reports the random baseline (1 / n_choices).
    """
    buckets: Dict[Tuple, List[bool]] = defaultdict(list)
    # Map normalized key → one representative (human-readable) choice list
    representative: Dict[Tuple, list] = {}

    for r in results:
        norm_key = tuple(sorted(_normalize_choice(c) for c in r["choices"]))
        buckets[norm_key].append(r["correct"])
        if norm_key not in representative:
            representative[norm_key] = sorted(_normalize_choice(c) for c in r["choices"])

    return sorted(
        [
            {
                "choices": representative[key],
                "total": len(vals),
                "correct": sum(vals),
                "accuracy": round(sum(vals) / len(vals), 3),
                "random_baseline": round(1 / len(key), 3),
            }
            for key, vals in buckets.items()
        ],
        key=lambda x: -x["total"],
    )


def answer_distribution_analysis(results: List[dict]) -> List[dict]:
    """
    For each distinct answer label, show how often it is the ground truth,
    how often the model predicted it, and the accuracy when it is the ground truth.
    """
    gt_counts: Dict[str, int] = defaultdict(int)
    pred_counts: Dict[str, int] = defaultdict(int)
    correct_when_gt: Dict[str, int] = defaultdict(int)

    for r in results:
        gt = r["ground_truth_label"]
        pred = r.get("predicted_label")
        gt_counts[gt] += 1
        if pred is not None:
            pred_counts[pred] += 1
        if r["correct"]:
            correct_when_gt[gt] += 1

    all_labels = sorted(set(gt_counts) | set(pred_counts))
    return [
        {
            "label": label,
            "ground_truth_count": gt_counts[label],
            "predicted_count": pred_counts[label],
            "accuracy_when_gt": round(correct_when_gt[label] / gt_counts[label], 3)
            if gt_counts[label] else None,
        }
        for label in sorted(all_labels, key=lambda l: -gt_counts[l])
    ]


def yes_no_random_baseline_analysis(results: List[dict]) -> dict:
    """
    For all questions where both 'Yes' and 'No' appear among the choices,
    compute the random baseline of a strategy that randomly picks Yes or No
    (ignoring all other options).

    P(correct) = (n_gt_yes + n_gt_no) / (2 * n_yn_questions)
    because:
      - if ground truth is Yes or No  → P(correct) = 1/2
      - if ground truth is anything else → P(correct) = 0
    """
    yn_results = [
        r for r in results
        if "Yes" in r["choices"] and "No" in r["choices"]
    ]
    if not yn_results:
        return {"total": 0, "baseline": None}

    n_gt_yes_or_no = sum(r["ground_truth_label"] in ("Yes", "No") for r in yn_results)
    baseline = n_gt_yes_or_no / (2 * len(yn_results))

    gt_counts: Dict[str, int] = defaultdict(int)
    for r in yn_results:
        gt_counts[r["ground_truth_label"]] += 1

    return {
        "total_yn_questions": len(yn_results),
        "gt_yes": gt_counts.get("Yes", 0),
        "gt_no": gt_counts.get("No", 0),
        "gt_other": len(yn_results) - gt_counts.get("Yes", 0) - gt_counts.get("No", 0),
        "yn_random_baseline": round(baseline, 4),
    }


def cntbd_always_correct_baseline(results: List[dict]) -> dict:
    """
    Baseline: always answer 'Cannot be determined' correctly, then for the
    remaining questions always guess only 'Yes' or only 'No'.

    Strategy A (always-yes): score = n_cntbd_gt + n_yes_gt
    Strategy B (always-no):  score = n_cntbd_gt + n_no_gt
    Both divided by total questions in the subset (those with CNTBD as a choice).
    """
    subset = [r for r in results if "Cannot be determined" in r.get("choices", [])]
    if not subset:
        return {"total": 0, "baseline_always_yes": None, "baseline_always_no": None}

    total = len(subset)
    n_cntbd = sum(r["ground_truth_label"] == "Cannot be determined" for r in subset)
    n_yes = sum(r["ground_truth_label"] == "Yes" for r in subset)
    n_no = sum(r["ground_truth_label"] == "No" for r in subset)

    return {
        "total_with_cntbd_choice": total,
        "gt_cntbd": n_cntbd,
        "gt_yes": n_yes,
        "gt_no": n_no,
        "gt_cntbd_frac": round(n_cntbd / total, 4),
        "gt_yes_frac": round(n_yes / total, 4),
        "gt_no_frac": round(n_no / total, 4),
        "baseline_always_yes": round((n_cntbd + n_yes) / total, 4),
        "baseline_always_no": round((n_cntbd + n_no) / total, 4),
    }


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
