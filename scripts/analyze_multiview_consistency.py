#!/usr/bin/env python3
"""
Cross-view consistency analysis for multiview runs.

Reads a completed run's results.jsonl and groups entries that represent the
same question asked from different camera views. Groups are formed by
(scene_id, question_type, question) — entries sharing those three fields came
from the same source timestep and differ only in camera view.

Consistency metrics (only counted when both views of a pair are present):

  tops_agree       : top_left prediction == top_right prediction
  one_top_wrist    : top_left==wrist OR top_right==wrist  (>=1 top agrees with wrist)
  both_tops_wrist  : top_left==wrist AND top_right==wrist (requires all 3 views)
  all_agree        : top_left == top_right == wrist       (requires all 3 views)

Broken down by: overall / question_type / correctness / scene.

Usage:
  python scripts/analyze_multiview_consistency.py outputs/<run_id>/results.jsonl
  python scripts/analyze_multiview_consistency.py outputs/<run_id>/
  python scripts/analyze_multiview_consistency.py --done
  python scripts/analyze_multiview_consistency.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
W = 84

METRICS = ["tops_agree", "one_top_wrist", "both_tops_wrist", "all_agree"]
METRIC_LABELS = {
    "tops_agree":      "top_left ↔ top_right",
    "one_top_wrist":   ">=1 top  ↔ wrist",
    "both_tops_wrist": "both tops ↔ wrist",
    "all_agree":       "all 3 agree",
}


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    decoder = json.JSONDecoder()
    text = path.read_text(encoding="utf-8")
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx] in " \t\n\r":
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        records.append(obj)
        idx = end
    return records


# ── grouping ──────────────────────────────────────────────────────────────────

def group_by_source(records: list[dict]) -> dict[tuple, dict]:
    """
    Group records by (scene_id, question_type, question).
    Only records with a real camera view (not 'combined') are included.
    """
    groups: dict[tuple, dict] = {}
    for r in records:
        view = r.get("view", "")
        if not view or view == "combined":
            continue
        key = (
            r.get("scene_id", ""),
            r.get("question_type") or r.get("task", ""),
            r.get("question", ""),
        )
        if key not in groups:
            groups[key] = {
                "scene_id":     key[0],
                "question_type": key[1],
                "question":     key[2],
                "views": {},
            }
        groups[key]["views"][view] = {
            "correct":   bool(r.get("correct", False)),
            "predicted": r.get("predicted_label") or "",
        }
    return groups


# ── per-group flags ───────────────────────────────────────────────────────────

def consistency_flags(group: dict) -> dict[str, bool]:
    v = group["views"]
    has_tl    = "top_left"    in v
    has_tr    = "top_right"   in v
    has_wrist = "bottom_left" in v

    flags: dict[str, bool] = {}

    if has_tl and has_tr:
        flags["tops_agree"] = v["top_left"]["predicted"] == v["top_right"]["predicted"]

    if has_wrist and (has_tl or has_tr):
        tl_ok = has_tl and v["top_left"]["predicted"]  == v["bottom_left"]["predicted"]
        tr_ok = has_tr and v["top_right"]["predicted"] == v["bottom_left"]["predicted"]
        flags["one_top_wrist"] = tl_ok or tr_ok
        if has_tl and has_tr:
            flags["both_tops_wrist"] = tl_ok and tr_ok

    if has_tl and has_tr and has_wrist:
        flags["all_agree"] = (
            v["top_left"]["predicted"]
            == v["top_right"]["predicted"]
            == v["bottom_left"]["predicted"]
        )

    return flags


def correctness_label(group: dict) -> str:
    views = group["views"]
    n_ok  = sum(v["correct"] for v in views.values())
    n     = len(views)
    if n_ok == n: return "all_correct"
    if n_ok == 0: return "none_correct"
    return "some_correct"


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate(groups: dict, filter_fn=None) -> dict[str, dict]:
    counts = {m: {"eligible": 0, "agree": 0} for m in METRICS}
    for g in groups.values():
        if filter_fn and not filter_fn(g):
            continue
        for m, val in consistency_flags(g).items():
            counts[m]["eligible"] += 1
            if val:
                counts[m]["agree"] += 1
    result = {}
    for m, c in counts.items():
        result[m] = {
            "eligible": c["eligible"],
            "agree":    c["agree"],
            "rate":     round(c["agree"] / c["eligible"], 4) if c["eligible"] else None,
        }
    return result


# ── printing ──────────────────────────────────────────────────────────────────

def hdr(title: str):
    print(f"\n{'='*W}\n  {title}\n{'='*W}")


def print_table(agg: dict, indent: int = 2):
    pad = " " * indent
    print(f"{pad}{'Metric':<28} {'Rate':>6}   Agree / Eligible")
    print(f"{pad}{'-'*52}")
    for m in METRICS:
        c = agg[m]
        if c["eligible"] == 0:
            print(f"{pad}{METRIC_LABELS[m]:<28}    —")
        else:
            print(f"{pad}{METRIC_LABELS[m]:<28} {c['rate']:>6.1%}   {c['agree']}/{c['eligible']}")


def print_scene_table(rows: list[tuple[str, dict]], limit: int):
    print(f"  {'Scene':<50} {'tops':>6}  {'>=1↔wrist':>9}  {'all':>6}  {'n':>4}")
    print(f"  {'-'*(W-2)}")
    for scene, agg in rows[:limit]:
        t  = agg["tops_agree"]
        o  = agg["one_top_wrist"]
        a  = agg["all_agree"]
        tr  = f"{t['rate']:.0%}" if t["rate"] is not None else "—"
        or_ = f"{o['rate']:.0%}" if o["rate"] is not None else "—"
        ar  = f"{a['rate']:.0%}" if a["rate"] is not None else "—"
        n   = t["eligible"] or o["eligible"] or a["eligible"]
        print(f"  {scene[:49]:<50} {tr:>6}  {or_:>9}  {ar:>6}  {n:>4}")


# ── main ──────────────────────────────────────────────────────────────────────

def build_summary(groups: dict, top_n: int) -> dict:
    """Build the full consistency summary as a serialisable dict."""
    qtypes = sorted({g["question_type"] for g in groups.values() if g["question_type"]})
    scenes = sorted({g["scene_id"] for g in groups.values() if g["scene_id"]})

    scene_rows = sorted(
        [
            (scene, aggregate(groups, filter_fn=lambda g, s=scene: g["scene_id"] == s))
            for scene in scenes
        ],
        key=lambda x: x[1]["tops_agree"]["rate"] if x[1]["tops_agree"]["rate"] is not None else 1.0,
    )

    return {
        "total_source_groups": len(groups),
        "overall": aggregate(groups),
        "by_question_type": {
            qt: aggregate(groups, filter_fn=lambda g, q=qt: g["question_type"] == q)
            for qt in qtypes
        },
        "by_correctness": {
            label: aggregate(groups, filter_fn=lambda g, l=label: correctness_label(g) == l)
            for label in ["all_correct", "some_correct", "none_correct"]
        },
        "most_inconsistent_scenes": [
            {"scene_id": s, **a} for s, a in scene_rows[:top_n]
        ],
        "most_consistent_scenes": [
            {"scene_id": s, **a} for s, a in reversed(scene_rows[-top_n:])
        ],
        "all_scenes": {s: a for s, a in scene_rows},
    }


def analyze(results_path: Path, top_n: int):
    print(f"\n{'─'*W}")
    print(f"  Run : {results_path.parent.name}")

    records = load_jsonl(results_path)
    groups  = group_by_source(records)

    mv_count = sum(1 for r in records if r.get("view") and r.get("view") != "combined")
    if not groups:
        print("  No multiview records found — skipping.")
        return

    print(f"  {len(records)} records total  |  {mv_count} multiview  |  {len(groups)} source groups")

    summary = build_summary(groups, top_n)

    # ── write summary_multiview.json ──────────────────────────────────────────
    out_path = results_path.parent / "summary_multiview.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Written → {out_path}")

    # ── print to stdout ───────────────────────────────────────────────────────
    hdr("OVERALL CROSS-VIEW CONSISTENCY")
    print_table(summary["overall"])

    qtypes = sorted(summary["by_question_type"])
    if len(qtypes) > 1:
        hdr("CONSISTENCY BY QUESTION TYPE")
        for qt in qtypes:
            n = sum(1 for g in groups.values() if g["question_type"] == qt)
            print(f"\n  {qt}  ({n} groups)")
            print_table(summary["by_question_type"][qt], indent=4)

    hdr("CONSISTENCY  x  CORRECTNESS")
    for label in ["all_correct", "some_correct", "none_correct"]:
        n = sum(1 for g in groups.values() if correctness_label(g) == label)
        if n == 0:
            continue
        print(f"\n  {label}  ({n} source entries)")
        print_table(summary["by_correctness"][label], indent=4)

    scene_rows = list(summary["all_scenes"].items())
    scene_rows.sort(key=lambda x: x[1]["tops_agree"]["rate"] if x[1]["tops_agree"]["rate"] is not None else 1.0)
    n = min(top_n, len(scene_rows))
    hdr(f"MOST INCONSISTENT {n} SCENES  (tops_agree lowest first)")
    print_scene_table(scene_rows, n)
    hdr(f"MOST CONSISTENT {n} SCENES  (tops_agree highest first)")
    print_scene_table(list(reversed(scene_rows)), n)


def main():
    p = argparse.ArgumentParser(
        description="Cross-view consistency analysis for multiview runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("path", nargs="?", type=Path,
                   help="results.jsonl path or run output directory")
    p.add_argument("--done", action="store_true",
                   help="auto-discover all *_done runs under outputs/")
    p.add_argument("--all", dest="use_all", action="store_true",
                   help="auto-discover ALL runs under outputs/")
    p.add_argument("--top", type=int, default=10,
                   help="scenes to show per table (default: 10)")
    args = p.parse_args()

    outputs_dir = PROJECT_ROOT / "outputs"

    if args.path:
        rp = args.path
        paths = [rp / "results.jsonl" if rp.is_dir() else rp]
    elif args.done:
        paths = sorted(
            q for q in outputs_dir.glob("*/results.jsonl")
            if q.parent.name.endswith("done")
        )
    elif args.use_all:
        paths = sorted(outputs_dir.glob("*/results.jsonl"))
    else:
        p.print_help()
        sys.exit(1)

    if not paths:
        print("No results.jsonl files found.", file=sys.stderr)
        sys.exit(1)

    for rp in paths:
        if not rp.exists():
            print(f"Not found: {rp}", file=sys.stderr)
            continue
        analyze(rp, args.top)


if __name__ == "__main__":
    main()
