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
import os
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


def _normalize_steps(rows: list[dict]) -> dict[str, dict[int, int]]:
    """
    Per scene, rank raw image_step values → 0-based normalized index.

    Returns {scene_id: {raw_step: norm_step}}.
    Rows without scene_id or image_step are ignored.
    """
    scene_steps: dict[str, set[int]] = defaultdict(set)
    for r in rows:
        sid  = r.get("scene_id")
        step = r.get("image_step")
        if sid is not None and step is not None:
            scene_steps[sid].add(step)
    return {
        sid: {raw: norm for norm, raw in enumerate(sorted(steps))}
        for sid, steps in scene_steps.items()
    }


def print_by_image_step(rows: list[dict]) -> None:
    _header('2. ACCURACY BY image_step  (normalized per scene: 0 = earliest, n = latest)')

    norm_map = _normalize_steps(rows)
    buckets: dict[int, list[bool]] = defaultdict(list)
    skipped = 0
    for r in rows:
        sid  = r.get("scene_id")
        step = r.get("image_step")
        if sid is None or step is None:
            skipped += 1
            continue
        scene_norm = norm_map.get(sid, {})
        norm_step = scene_norm.get(step)
        if norm_step is None:
            skipped += 1
            continue
        buckets[norm_step].append(r["correct"])

    if not buckets:
        print("  No image_step field in results.")
        return

    if skipped:
        print(f"  (skipped {skipped} rows missing scene_id or image_step)")

    max_norm = max(buckets)
    print(f"  Normalized steps 0..{max_norm}  (0 = first frame in scene, {max_norm} = last)")
    print()
    print(f"  {'norm_step':>10}  {'n':>6}  {'acc':>7}  bar")
    print(f"  {'----------':>10}  {'------':>6}  {'-------':>7}  ---")
    for step in range(max_norm + 1):
        vals = buckets.get(step, [])
        a = acc(vals) if vals else float("nan")
        n_str = str(len(vals)) if vals else "-"
        print(f"  {step:>10}  {n_str:>6}  {pct(a)}  {bar(a)}")


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


def _classify_yn(label: str | None) -> str:
    if label is None:
        return "unparseable"
    l = label.strip().lower()
    if l == "yes":
        return "yes"
    if l == "no":
        return "no"
    return "other"


def _image_type(r: dict) -> str:
    """Classify entry as 'black_tile', 'random_scene', or 'normal'."""
    special = r.get("special_image")
    if special == "random_scene":
        return "random_scene"
    if special == "black_image":
        return "black_tile"
    # Fallback for datasets built before special_image was populated:
    # the black tile image is stored as tiles/_black.jpg (no scene prefix).
    paths = r.get("image_paths") or []
    if any(p.endswith("/_black.jpg") or p.endswith(os.sep + "_black.jpg") for p in paths):
        return "black_tile"
    return "normal"


def _print_yn_table(rows: list[dict], label: str = "") -> None:
    cats = ["yes", "no", "other", "unparseable"]
    gt_counts: dict[str, int] = defaultdict(int)
    pred_counts: dict[str, int] = defaultdict(int)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total = len(rows)

    for r in rows:
        gt_cat   = _classify_yn(r.get("ground_truth_label"))
        pred_cat = _classify_yn(r.get("predicted_label"))
        gt_counts[gt_cat] += 1
        pred_counts[pred_cat] += 1
        confusion[gt_cat][pred_cat] += 1

    if label:
        print(f"\n  [{label}]  n={total}")
    print(f"  {'category':<14}  {'GT count':>9}  {'GT %':>7}  {'Pred count':>11}  {'Pred %':>7}")
    print(f"  {'-'*14}  {'-'*9}  {'-'*7}  {'-'*11}  {'-'*7}")
    for cat in cats:
        gt_n   = gt_counts[cat]
        pred_n = pred_counts[cat]
        print(
            f"  {cat:<14}  {gt_n:>9}  {pct(gt_n/total if total else 0)}  "
            f"{pred_n:>11}  {pct(pred_n/total if total else 0)}"
        )
    print("  Confusion (rows=GT, cols=pred):")
    header = f"  {'GT \\ pred':<14}  " + "  ".join(f"{'→'+c:>14}" for c in cats)
    print(header)
    for gt_cat in cats:
        row_vals = [confusion[gt_cat][pc] for pc in cats]
        row_str  = "  ".join(f"{v:>14}" for v in row_vals)
        total_gt = gt_counts[gt_cat]
        correct_this = confusion[gt_cat][gt_cat]
        acc_str = f"  acc={pct(correct_this/total_gt if total_gt else float('nan'))}" if total_gt else ""
        print(f"  {gt_cat:<14}  {row_str}{acc_str}")


def print_yn_distribution(rows: list[dict]) -> None:
    _header("4. YES / NO / OTHER DISTRIBUTION")

    _print_yn_table(rows, label="ALL")

    # Breakdown by image type
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[_image_type(r)].append(r)

    type_order = ["normal", "random_scene", "black_tile"]
    type_labels = {
        "normal":       "normal  (real scene tile)",
        "random_scene": "random  (tile from different scene → CBD expected)",
        "black_tile":   "black   (black tile → CBD expected)",
    }
    present = [t for t in type_order if by_type[t]]
    if len(present) > 1:
        for itype in present:
            _print_yn_table(by_type[itype], label=type_labels[itype])


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

    # Count unique normalized steps per scene
    norm_map = _normalize_steps(rows)
    scene_n_steps = {sid: len(norm_map.get(sid, {})) for sid in included}

    accs = {s: acc(v) for s, v in included.items()}
    mean_acc = sum(accs.values()) / len(accs)

    sorted_scenes = sorted(accs.items(), key=lambda x: x[1])

    print(f"  Total scenes : {total_scenes}")
    print(f"  Excluded     : {len(excluded)}  (< {min_q} questions)")
    print(f"  Included     : {len(included)}")
    print(f"  Mean accuracy: {pct(mean_acc)}")

    n_show = 10

    print(f"\n  --- Hardest {n_show} scenes ---")
    print(f"  {'scene_id':<60}  {'steps':>6}  {'n_q':>5}  {'acc':>7}  bar")
    print(f"  {'-'*60}  {'------':>6}  {'-----':>5}  {'-------':>7}  ---")
    for sid, a in sorted_scenes[:n_show]:
        nq    = len(included[sid])
        nstep = scene_n_steps.get(sid, 0)
        trunc = sid[:60] if len(sid) <= 60 else sid[:57] + "..."
        print(f"  {trunc:<60}  {nstep:>6}  {nq:>5}  {pct(a)}  {bar(a)}")

    print(f"\n  --- Easiest {n_show} scenes ---")
    print(f"  {'scene_id':<60}  {'steps':>6}  {'n_q':>5}  {'acc':>7}  bar")
    print(f"  {'-'*60}  {'------':>6}  {'-----':>5}  {'-------':>7}  ---")
    for sid, a in sorted_scenes[-n_show:]:
        nq    = len(included[sid])
        nstep = scene_n_steps.get(sid, 0)
        trunc = sid[:60] if len(sid) <= 60 else sid[:57] + "..."
        print(f"  {trunc:<60}  {nstep:>6}  {nq:>5}  {pct(a)}  {bar(a)}")

    # Accuracy grouped by number of steps
    step_buckets: dict[int, list[float]] = defaultdict(list)
    for sid, a in accs.items():
        step_buckets[scene_n_steps.get(sid, 0)].append(a)

    print(f"\n  --- Accuracy by scene length (number of distinct steps) ---")
    print(f"  {'n_steps':>8}  {'scenes':>7}  {'mean_acc':>9}  bar")
    print(f"  {'--------':>8}  {'-------':>7}  {'-'*9}  ---")
    for ns in sorted(step_buckets):
        scene_accs = step_buckets[ns]
        mean_a = sum(scene_accs) / len(scene_accs)
        print(f"  {ns:>8}  {len(scene_accs):>7}  {pct(mean_a)}  {bar(mean_a)}")


def print_variant_comparison(rows: list[dict]) -> None:
    """
    Compare variant A (no action sequence) vs variant B (sequence in context).

    Pairs are matched by original_id when available, otherwise by
    (scene_id, image_step, question_type, label_phase) — the fields that
    identify the same underlying question across variants.
    """
    _header("6. VARIANT COMPARISON  (A = no sequence context  vs  B = sequence in context)")

    variants = sorted({r.get("variant") for r in rows if r.get("variant")})
    if not variants:
        print("  No variant field in results.")
        return
    if len(variants) == 1:
        print(f"  Only one variant present: {variants[0]}  — nothing to compare.")
        return

    # ── per-variant overall ───────────────────────────────────────────────────
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        v = r.get("variant")
        if v:
            by_variant[v].append(r)

    max_v = max(len(v) for v in variants)
    print(f"  {'variant':{max_v}}  {'n':>6}  {'acc':>7}  bar")
    print(f"  {'-'*max_v}  {'------':>6}  {'-------':>7}  ---")
    for v in variants:
        vals = [r["correct"] for r in by_variant[v]]
        a = acc(vals)
        print(f"  {v:{max_v}}  {len(vals):>6}  {pct(a)}  {bar(a)}")

    # ── per-question-type breakdown ───────────────────────────────────────────
    print()
    print("  Per question_type:")
    qt_variants: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v  = r.get("variant")
        qt = r.get("question_type") or r.get("task") or "unknown"
        if v:
            qt_variants[qt][v].append(r["correct"])

    all_qts = sorted(qt_variants)
    max_qt = max(len(q) for q in all_qts) if all_qts else 4
    variant_cols = "  ".join(f"{'acc_' + v:>9}" for v in variants)
    print(f"  {'question_type':{max_qt}}  {variant_cols}  delta")
    print(f"  {'-'*max_qt}  {'  '.join(['─'*9]*len(variants))}  -----")
    for qt in all_qts:
        accs = [acc(qt_variants[qt].get(v, [])) for v in variants]
        acc_cols = "  ".join(f"{pct(a):>9}" for a in accs)
        # delta = last variant minus first (B - A if only two)
        delta = accs[-1] - accs[0] if all(a == a for a in accs) else float("nan")
        sign  = "+" if delta >= 0 else ""
        delta_str = f"{sign}{delta:+.1%}" if delta == delta else "  n/a"
        print(f"  {qt:{max_qt}}  {acc_cols}  {delta_str}")

    # ── matched-pair analysis ─────────────────────────────────────────────────
    # Only meaningful when we have exactly A and B; skip for 3+ variants.
    if set(variants) != {"A", "B"}:
        return

    print()
    print("  Matched-pair analysis (same question, A vs B):")

    def _pair_key(r: dict) -> str:
        oid = r.get("original_id")
        if oid:
            return oid
        return "|".join(str(r.get(k, "")) for k in
                        ("scene_id", "image_step", "question_type", "label_phase"))

    keyed: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        v = r.get("variant")
        if v in ("A", "B"):
            keyed[_pair_key(r)][v] = r

    pairs = [(d["A"], d["B"]) for d in keyed.values() if "A" in d and "B" in d]
    unpaired_a = sum(1 for d in keyed.values() if "A" in d and "B" not in d)
    unpaired_b = sum(1 for d in keyed.values() if "B" in d and "A" not in d)

    if not pairs:
        print(f"  No matched pairs found. Unpaired: A={unpaired_a}, B={unpaired_b}")
        return

    both_correct   = sum(a["correct"] and b["correct"]     for a, b in pairs)
    a_only_correct = sum(a["correct"] and not b["correct"] for a, b in pairs)
    b_only_correct = sum(not a["correct"] and b["correct"] for a, b in pairs)
    both_wrong     = sum(not a["correct"] and not b["correct"] for a, b in pairs)

    print(f"  Matched pairs   : {len(pairs)}")
    if unpaired_a or unpaired_b:
        print(f"  Unpaired        : A={unpaired_a}, B={unpaired_b}")
    print()
    print(f"  {'outcome':<28}  {'count':>6}  {'%':>7}")
    print(f"  {'-'*28}  {'------':>6}  {'-------':>7}")
    n = len(pairs)
    print(f"  {'both correct':<28}  {both_correct:>6}  {pct(both_correct/n)}")
    print(f"  {'A correct, B wrong':<28}  {a_only_correct:>6}  {pct(a_only_correct/n)}")
    print(f"  {'B correct, A wrong':<28}  {b_only_correct:>6}  {pct(b_only_correct/n)}")
    print(f"  {'both wrong':<28}  {both_wrong:>6}  {pct(both_wrong/n)}")
    net = b_only_correct - a_only_correct
    sign = "+" if net >= 0 else ""
    print(f"\n  Net gain B over A: {sign}{net} questions  ({sign}{pct(net/n)})")


def print_progress_yn_baseline(rows: list[dict]) -> None:
    """
    For the 'progress' question type: what if the model always said Yes or always No?
    Choices are Yes / No / Cannot be determined.
    Also shows the actual model bias (how often it picks each option).
    """
    prog = [r for r in rows if r.get("question_type") == "progress" or r.get("task") == "progress"]
    if not prog:
        return

    _header("7. PROGRESS — YES/NO CONSTANT BASELINE")

    total = len(prog)
    gt_yes = sum(r["ground_truth_label"] == "Yes" for r in prog)
    gt_no  = sum(r["ground_truth_label"] == "No"  for r in prog)
    gt_cbd = total - gt_yes - gt_no

    print(f"  Progress samples : {total}")
    print(f"  GT distribution  : Yes={gt_yes} ({pct(gt_yes/total)})  "
          f"No={gt_no} ({pct(gt_no/total)})  CBD={gt_cbd} ({pct(gt_cbd/total)})")

    # Constant baselines
    acc_always_yes = gt_yes / total
    acc_always_no  = gt_no  / total
    acc_always_cbd = gt_cbd / total
    acc_random     = sum(1 / len(r["choices"]) for r in prog) / total

    print()
    print(f"  {'strategy':<28}  {'acc':>7}  bar")
    print(f"  {'-'*28}  {'-------':>7}  ---")
    print(f"  {'always Yes':<28}  {pct(acc_always_yes)}  {bar(acc_always_yes)}")
    print(f"  {'always No':<28}  {pct(acc_always_no)}  {bar(acc_always_no)}")
    print(f"  {'always CBD':<28}  {pct(acc_always_cbd)}  {bar(acc_always_cbd)}")
    print(f"  {'random (uniform)':<28}  {pct(acc_random)}  {bar(acc_random)}")

    # Actual model accuracy and prediction distribution
    actual_acc  = acc([r["correct"] for r in prog])
    pred_yes    = sum(r.get("predicted_label") == "Yes" for r in prog)
    pred_no     = sum(r.get("predicted_label") == "No"  for r in prog)
    pred_cbd    = sum(r.get("predicted_label") not in ("Yes", "No") and r.get("predicted_label") is not None for r in prog)
    pred_unp    = sum(r.get("predicted_label") is None for r in prog)

    print(f"  {'model (actual)':<28}  {pct(actual_acc)}  {bar(actual_acc)}")
    print()
    print(f"  Model prediction distribution:")
    print(f"    Yes={pred_yes} ({pct(pred_yes/total)})  "
          f"No={pred_no} ({pct(pred_no/total)})  "
          f"CBD={pred_cbd} ({pct(pred_cbd/total)})  "
          f"unparseable={pred_unp} ({pct(pred_unp/total)})")

    # Yes/No only: if we only count questions where GT is Yes or No,
    # what's the model's accuracy on those vs random?
    yn_only = [r for r in prog if r["ground_truth_label"] in ("Yes", "No")]
    if yn_only:
        yn_total = len(yn_only)
        yn_acc   = acc([r["correct"] for r in yn_only])
        yn_always_yes = sum(r["ground_truth_label"] == "Yes" for r in yn_only) / yn_total
        yn_always_no  = sum(r["ground_truth_label"] == "No"  for r in yn_only) / yn_total
        print()
        print(f"  Yes/No-only subset (n={yn_total}, excludes CBD ground truths):")
        print(f"    always Yes    : {pct(yn_always_yes)}")
        print(f"    always No     : {pct(yn_always_no)}")
        print(f"    random (50/50): {pct(0.5)}")
        print(f"    model actual  : {pct(yn_acc)}")


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
    print_variant_comparison(rows)
    print_progress_yn_baseline(rows)
    print()


if __name__ == "__main__":
    main()
