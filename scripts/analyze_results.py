"""
Analyze results from one or more results.jsonl files produced by pipeline.py.

Usage:
    python scripts/analyze_results.py outputs/my_run/results.jsonl
    python scripts/analyze_results.py outputs/run1/results.jsonl outputs/run2/results.jsonl
    python scripts/analyze_results.py outputs/  # scans for all results.jsonl under dir

Stats printed:
  1. Overall accuracy
  2. Accuracy by image_step (are "further away" steps easier/harder?)
  3. Accuracy by question_type / task
  4. Yes / No / other distribution  (model vs ground truth)
  5. Per-scene difficulty
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def load_results(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return rows


def acc(correct_list: list[bool]) -> float:
    return sum(correct_list) / len(correct_list) if correct_list else float("nan")


def bar(value: float, width: int = 20) -> str:
    if value != value:  # nan
        return " " * width
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def pct(value: float) -> str:
    if value != value:
        return "  n/a"
    return f"{value:5.1%}"


def _header(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ── sections ─────────────────────────────────────────────────────────────────

def print_overall(rows: list[dict]) -> None:
    _header("1. OVERALL")
    total = len(rows)
    correct = sum(r["correct"] for r in rows)
    unparseable = sum(r.get("predicted_index") is None for r in rows)
    random_bl = sum(1 / len(r["choices"]) for r in rows) / total if total else 0
    print(f"  Samples     : {total}")
    print(f"  Correct     : {correct}  ({pct(correct/total if total else float('nan'))})")
    print(f"  Wrong       : {total - correct - unparseable}")
    print(f"  Unparseable : {unparseable}")
    print(f"  Random BL   : {pct(random_bl)}")


def print_by_image_step(rows: list[dict]) -> None:
    _header('2. ACCURACY BY image_step  ("further away" = higher step)')
    buckets: dict[int, list[bool]] = defaultdict(list)
    for r in rows:
        step = r.get("image_step")
        if step is not None:
            buckets[step].append(r["correct"])

    if not buckets:
        print("  No image_step field in results.")
        return

    print(f"  {'step':>6}  {'n':>6}  {'acc':>7}  bar")
    print(f"  {'-----':>6}  {'------':>6}  {'-------':>7}  ---")
    for step in sorted(buckets):
        vals = buckets[step]
        a = acc(vals)
        print(f"  {step:>6}  {len(vals):>6}  {pct(a)}  {bar(a)}")


def print_by_question_type(rows: list[dict]) -> None:
    _header("3. ACCURACY BY question_type / task")
    buckets: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        qt = r.get("question_type") or r.get("task") or "unknown"
        buckets[qt].append(r["correct"])

    if not buckets:
        print("  No question_type field in results.")
        return

    sorted_types = sorted(buckets.items(), key=lambda x: -acc(x[1]))
    max_len = max(len(k) for k in buckets)
    print(f"  {'type':{max_len}}  {'n':>6}  {'acc':>7}  bar")
    print(f"  {'-'*max_len}  {'------':>6}  {'-------':>7}  ---")
    for qt, vals in sorted_types:
        a = acc(vals)
        print(f"  {qt:{max_len}}  {len(vals):>6}  {pct(a)}  {bar(a)}")


def print_yn_distribution(rows: list[dict]) -> None:
    _header("4. YES / NO / OTHER DISTRIBUTION")

    def classify(label: str | None) -> str:
        if label is None:
            return "unparseable"
        l = label.strip().lower()
        if l == "yes":
            return "yes"
        if l == "no":
            return "no"
        return "other"

    gt_counts: dict[str, int] = defaultdict(int)
    pred_counts: dict[str, int] = defaultdict(int)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in rows:
        gt_cat   = classify(r.get("ground_truth_label"))
        pred_cat = classify(r.get("predicted_label"))
        gt_counts[gt_cat] += 1
        pred_counts[pred_cat] += 1
        confusion[gt_cat][pred_cat] += 1

    cats = ["yes", "no", "other", "unparseable"]
    total = len(rows)

    print(f"\n  {'category':<14}  {'GT count':>9}  {'GT %':>7}  {'Pred count':>11}  {'Pred %':>7}")
    print(f"  {'-'*14}  {'-'*9}  {'-'*7}  {'-'*11}  {'-'*7}")
    for cat in cats:
        gt_n   = gt_counts[cat]
        pred_n = pred_counts[cat]
        print(
            f"  {cat:<14}  {gt_n:>9}  {pct(gt_n/total if total else 0)}  "
            f"{pred_n:>11}  {pct(pred_n/total if total else 0)}"
        )

    print("\n  Confusion matrix (rows = GT, cols = predicted):")
    header = f"  {'GT \\ pred':<14}  " + "  ".join(f"{'→'+c:>14}" for c in cats)
    print(header)
    for gt_cat in cats:
        row_vals = [confusion[gt_cat][pc] for pc in cats]
        row_str = "  ".join(f"{v:>14}" for v in row_vals)
        # accuracy for this gt category
        total_gt = gt_counts[gt_cat]
        correct_this = confusion[gt_cat][gt_cat]
        acc_str = f"  acc={pct(correct_this/total_gt if total_gt else float('nan'))}" if total_gt else ""
        print(f"  {gt_cat:<14}  {row_str}{acc_str}")


def print_scene_difficulty(rows: list[dict], min_q: int = 3) -> None:
    _header("5. PER-SCENE DIFFICULTY")
    buckets: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        sid = r.get("scene_id")
        if sid:
            buckets[sid].append(r["correct"])

    if not buckets:
        print("  No scene_id field in results.")
        return

    total_scenes = len(buckets)
    excluded = {s for s, v in buckets.items() if len(v) < min_q}
    included = {s: v for s, v in buckets.items() if s not in excluded}

    if not included:
        print(f"  All {total_scenes} scenes have fewer than {min_q} questions — lower --min-q threshold.")
        return

    accs = {s: acc(v) for s, v in included.items()}
    mean_acc = sum(accs.values()) / len(accs)

    sorted_scenes = sorted(accs.items(), key=lambda x: x[1])

    print(f"  Total scenes : {total_scenes}")
    print(f"  Excluded     : {len(excluded)}  (< {min_q} questions)")
    print(f"  Included     : {len(included)}")
    print(f"  Mean accuracy: {pct(mean_acc)}")

    n_show = 10

    print(f"\n  --- Hardest {n_show} scenes ---")
    print(f"  {'scene_id':<60}  {'n':>5}  {'acc':>7}  bar")
    print(f"  {'-'*60}  {'-----':>5}  {'-------':>7}  ---")
    for sid, a in sorted_scenes[:n_show]:
        n = len(included[sid])
        trunc = sid[:60] if len(sid) <= 60 else sid[:57] + "..."
        print(f"  {trunc:<60}  {n:>5}  {pct(a)}  {bar(a)}")

    print(f"\n  --- Easiest {n_show} scenes ---")
    print(f"  {'scene_id':<60}  {'n':>5}  {'acc':>7}  bar")
    print(f"  {'-'*60}  {'-----':>5}  {'-------':>7}  ---")
    for sid, a in sorted_scenes[-n_show:]:
        n = len(included[sid])
        trunc = sid[:60] if len(sid) <= 60 else sid[:57] + "..."
        print(f"  {trunc:<60}  {n:>5}  {pct(a)}  {bar(a)}")


# ── entry point ───────────────────────────────────────────────────────────────

def resolve_paths(args: list[str]) -> list[Path]:
    paths = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            found = sorted(p.rglob("results.jsonl"))
            if not found:
                print(f"Warning: no results.jsonl found under {p}", file=sys.stderr)
            paths.extend(found)
        elif p.exists():
            paths.append(p)
        else:
            print(f"Warning: path not found: {p}", file=sys.stderr)
    return paths


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Analyze VLM benchmark results")
    parser.add_argument("paths", nargs="+", help="results.jsonl file(s) or directories")
    parser.add_argument("--min-q", type=int, default=3, help="Min questions per scene to include (default: 3)")
    args = parser.parse_args()

    paths = resolve_paths(args.paths)
    if not paths:
        print("No results.jsonl files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(paths)} file(s)...")
    for p in paths:
        print(f"  {p}")

    rows = load_results(paths)
    print(f"Loaded {len(rows)} result rows.")

    print_overall(rows)
    print_by_image_step(rows)
    print_by_question_type(rows)
    print_yn_distribution(rows)
    print_scene_difficulty(rows, min_q=args.min_q)
    print()


if __name__ == "__main__":
    main()
