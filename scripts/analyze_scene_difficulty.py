#!/usr/bin/env python3
"""Analyze scene/question difficulty across result files.

Result selection (auto-discovery, no positional args given):
  Default : outputs/*_done/details.jsonl   (completed runs only)
  --all   : outputs/*/details.jsonl        (all runs including partial)

Filters applied after loading:
  --task  <type>   keep only records with that question_type
  --model <id>     keep only records from that model

Usage examples:
  python scripts/analyze_scene_difficulty.py
  python scripts/analyze_scene_difficulty.py --all --model qwen3-8b
  python scripts/analyze_scene_difficulty.py /abs/path/details.jsonl ...
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
W = 80  # column width


def iter_json_objects(text: str):
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx] in " \t\n\r":
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        yield obj
        idx = end


EXPECTED_RECORDS = 6224

def load_records(paths: list[Path]) -> list[dict]:
    records = []
    for p in paths:
        recs = list(iter_json_objects(p.read_text()))
        if len(recs) != EXPECTED_RECORDS:
            print(f"  SKIP {p.parent.name}  ({len(recs)} records, expected {EXPECTED_RECORDS})")
            continue
        is_cot = "cot" in p.parent.name.lower()
        if is_cot:
            for r in recs:
                r["model_id"] = r.get("model_id", "unknown") + "-cot"
        print(f"  OK   {p.parent.name}  model={recs[0].get('model_id', 'unknown')!r}")
        records.extend(recs)
    return records


# ── aggregation ──────────────────────────────────────────────────────────────

def aggregate_scenes(records: list[dict]) -> dict[str, dict]:
    scenes: dict = defaultdict(lambda: {
        "total": 0, "correct": 0,
        "task_desc": "",
        "models": defaultdict(lambda: {"total": 0, "correct": 0}),
        "qtypes": defaultdict(lambda: {"total": 0, "correct": 0}),
    })
    for r in records:
        sid   = r.get("scene_id") or "unknown"
        ok    = int(r.get("correct", False))
        model = r.get("model_id", "unknown")
        qtype = r.get("question_type") or r.get("task") or "unknown"
        s = scenes[sid]
        s["total"]  += 1
        s["correct"] += ok
        if not s["task_desc"] and r.get("task_desc"):
            s["task_desc"] = r["task_desc"]
        s["models"][model]["total"]   += 1
        s["models"][model]["correct"] += ok
        s["qtypes"][qtype]["total"]   += 1
        s["qtypes"][qtype]["correct"] += ok
    return dict(scenes)


def aggregate_questions(records: list[dict]) -> dict[str, dict]:
    """Group by entry_id — unique per question (0-6223), consistent across runs/models."""
    qs: dict = defaultdict(lambda: {
        "total": 0, "correct": 0,
        "scene_id": "", "qtype": "", "question": "",
        "models": defaultdict(lambda: {"total": 0, "correct": 0}),
    })
    for r in records:
        qid   = str(r.get("entry_id", "unknown"))
        ok    = int(r.get("correct", False))
        model = r.get("model_id", "unknown")
        q = qs[qid]
        q["total"]   += 1
        q["correct"] += ok
        if not q["scene_id"]:
            q["scene_id"] = r.get("scene_id", "")
        if not q["qtype"]:
            q["qtype"] = r.get("question_type") or r.get("task") or ""
        if not q["question"]:
            q["question"] = r.get("question", "")[:120]
        q["models"][model]["total"]   += 1
        q["models"][model]["correct"] += ok
    return dict(qs)


# ── print helpers ─────────────────────────────────────────────────────────────

def hdr(title: str):
    print(f"\n{'='*W}\n  {title}\n{'='*W}")


def print_model_breakdown(records: list[dict]):
    data: dict = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in records:
        m = r.get("model_id", "unknown")
        data[m]["total"]   += 1
        data[m]["correct"] += int(r.get("correct", False))
    hdr("Per-model accuracy (all scenes, all tasks)")
    print(f"  {'Model':<35} {'Acc':>6}   Correct / Total")
    print(f"  {'-'*60}")
    for model, d in sorted(data.items()):
        acc = d["correct"] / d["total"]
        print(f"  {model:<35} {acc:>6.1%}   {d['correct']}/{d['total']}")


def compute_baselines(records: list[dict]) -> dict[str, dict]:
    """Per question type: uniform-random, yes/no-random, majority-class baselines."""
    from collections import Counter
    qt_choices: dict = defaultdict(list)
    qt_gt:      dict = defaultdict(list)
    for r in records:
        t  = r.get("question_type") or r.get("task") or "unknown"
        qt_choices[t].append(len(r.get("choices") or []))
        qt_gt[t].append(r.get("ground_truth_label", ""))

    baselines: dict = {}
    all_choices, all_gt = [], []
    for t in qt_choices:
        n_choices = round(sum(qt_choices[t]) / len(qt_choices[t]))  # should be constant
        gt_counts = Counter(qt_gt[t])
        majority_acc = gt_counts.most_common(1)[0][1] / len(qt_gt[t])
        # yes/no random: fraction of qs with binary (yes/no) answer × 0.5
        yn_count = gt_counts.get("Yes", 0) + gt_counts.get("No", 0)
        yn_acc = (yn_count / len(qt_gt[t])) * 0.5
        baselines[t] = {
            "uniform":  1.0 / n_choices if n_choices else 0,
            "yn":       yn_acc,
            "majority": majority_acc,
            "n_choices": n_choices,
            "majority_label": gt_counts.most_common(1)[0][0],
        }
        all_choices += qt_choices[t]
        all_gt      += qt_gt[t]

    overall_n = round(sum(all_choices) / len(all_choices))
    gt_all = Counter(all_gt)
    yn_all = gt_all.get("Yes", 0) + gt_all.get("No", 0)
    baselines["__overall__"] = {
        "uniform":  1.0 / overall_n,
        "yn":       (yn_all / len(all_gt)) * 0.5,
        "majority": gt_all.most_common(1)[0][1] / len(all_gt),
        "n_choices": overall_n,
        "majority_label": gt_all.most_common(1)[0][0],
    }
    return baselines


def print_task_breakdown(records: list[dict]):
    # aggregate: model -> qtype -> {total, correct}
    models: list[str] = sorted({r.get("model_id", "unknown") for r in records})
    qtypes: list[str] = sorted({r.get("question_type") or r.get("task") or "unknown" for r in records})
    cell: dict = defaultdict(lambda: {"total": 0, "correct": 0})
    overall_qt: dict = defaultdict(lambda: {"total": 0, "correct": 0})
    overall_m:  dict = defaultdict(lambda: {"total": 0, "correct": 0})
    grand = {"total": 0, "correct": 0}
    for r in records:
        m = r.get("model_id", "unknown")
        t = r.get("question_type") or r.get("task") or "unknown"
        ok = int(r.get("correct", False))
        cell[(m, t)]["total"]   += 1
        cell[(m, t)]["correct"] += ok
        overall_qt[t]["total"]   += 1
        overall_qt[t]["correct"] += ok
        overall_m[m]["total"]    += 1
        overall_m[m]["correct"]  += ok
        grand["total"]   += 1
        grand["correct"] += ok

    baselines = compute_baselines(records)

    def fmt(d: dict) -> str:
        if d["total"] == 0:
            return f"{'—':>8}"
        return f"{d['correct']/d['total']:>6.1%}"

    def fmtb(v: float) -> str:
        return f"{v:>6.1%}"

    col  = 22
    qt_w = 18
    hdr("Per-model × per-question-type accuracy")
    header = f"  {'Model':<{col}}" + "".join(f"  {qt[:qt_w]:<{qt_w}}" for qt in qtypes) + f"  {'Overall':>8}"
    sep = f"  {'-'*(len(header)-2)}"
    print(header)
    print(sep)
    for m in models:
        row = f"  {m:<{col}}"
        for t in qtypes:
            row += f"  {fmt(cell[(m,t)]):<{qt_w}}"
        row += f"  {fmt(overall_m[m]):>8}"
        print(row)
    print(sep)
    total_row = f"  {'Overall':<{col}}"
    for t in qtypes:
        total_row += f"  {fmt(overall_qt[t]):<{qt_w}}"
    total_row += f"  {fmt(grand):>8}"
    print(total_row)
    print(sep)

    # baselines
    for label, key in [
        ("random uniform (1/N)",  "uniform"),
        ("random yes/no (50/50)", "yn"),
        ("majority class",        "majority"),
    ]:
        row = f"  {label:<{col}}"
        for t in qtypes:
            b = baselines.get(t, {})
            row += f"  {fmtb(b.get(key, 0)):<{qt_w}}"
        b_all = baselines["__overall__"]
        row += f"  {fmtb(b_all[key]):>8}"
        print(row)

    # footnote: choices count and majority label per qtype
    print()
    for t in qtypes:
        b = baselines.get(t, {})
        print(f"  {t}: {b.get('n_choices','?')} choices, majority='{b.get('majority_label','?')}'")
    b_all = baselines["__overall__"]
    print(f"  overall: avg {b_all['n_choices']} choices, majority='{b_all['majority_label']}'")


def print_scene_table(scenes: dict, title: str, ranked_slice: list):
    hdr(title)
    print(f"  {'Scene ID':<52} {'Acc':>6}  Correct/Total")
    print(f"  {'-'*(W-2)}")
    for sid, acc, total, correct in ranked_slice:
        print(f"  {sid[:51]:<52} {acc:>6.1%}  {correct}/{total}")
        if scenes[sid]["task_desc"]:
            print(f"    desc : {scenes[sid]['task_desc'][:68]}")
        # per-model
        for model, s in sorted(scenes[sid]["models"].items()):
            m_acc = s["correct"] / s["total"] if s["total"] else 0
            print(f"    [{model:<22}] {m_acc:>6.1%}  {s['correct']}/{s['total']}")
        # per-question-type within scene
        for qt, s in sorted(scenes[sid]["qtypes"].items()):
            qt_acc = s["correct"] / s["total"] if s["total"] else 0
            print(f"    {qt:<28}   {qt_acc:>6.1%}  {s['correct']}/{s['total']}")


def print_question_table(questions: dict, title: str, ranked_slice: list):
    hdr(title)
    print(f"  {'original_id':<55} {'Type':<18} {'Acc':>6}  C/T")
    print(f"  {'-'*(W-2)}")
    for qid, acc, total, correct in ranked_slice:
        q = questions[qid]
        print(f"  {qid[:54]:<55} {q['qtype'][:17]:<18} {acc:>6.1%}  {correct}/{total}")
        # condensed question text
        qtext = q["question"].replace("\n", " ")[:75]
        print(f"    Q: {qtext}")
        # per-model
        for model, s in sorted(q["models"].items()):
            m_acc = s["correct"] / s["total"] if s["total"] else 0
            print(f"    [{model:<22}] {m_acc:>6.1%}  {s['correct']}/{s['total']}")


# ── main ──────────────────────────────────────────────────────────────────────

def rank(data: dict, key_total="total", key_correct="correct") -> list:
    out = []
    for k, v in data.items():
        if v[key_total] == 0:
            continue
        acc = v[key_correct] / v[key_total]
        out.append((k, acc, v[key_total], v[key_correct]))
    out.sort(key=lambda x: x[1])
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Analyze scene/question difficulty across result files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths", nargs="*", type=Path,
        help="Explicit details.jsonl paths. Auto-discovers if omitted.",
    )
    parser.add_argument(
        "--all", dest="use_all", action="store_true",
        help="Auto-discover ALL runs (default: only *_done runs)",
    )
    parser.add_argument("--top",     type=int, default=10, help="N scenes/questions to show (default 10)")
    parser.add_argument("--top-q",   type=int, default=None, help="N questions to show (default: same as --top)")
    parser.add_argument("--task",    help="Filter records by question_type")
    parser.add_argument("--model",   help="Filter records by model_id")
    args = parser.parse_args()

    # ── file selection ────────────────────────────────────────────────────────
    if args.paths:
        paths = list(args.paths)
        print(f"\nUsing {len(paths)} explicitly provided file(s).")
    else:
        outputs_dir = PROJECT_ROOT / "outputs"
        if args.use_all:
            paths = sorted(outputs_dir.rglob("details.jsonl"))
            print(f"\nAuto-discovered ALL runs (--all flag set).")
        else:
            paths = sorted(
                p for p in outputs_dir.rglob("details.jsonl")
                if p.parent.name.endswith("done")
            )
            print(f"\nAuto-discovered completed runs only (dirs ending in '_done').")
            print("  Pass --all to include partial/in-progress runs.")
        if not paths:
            print("No details.jsonl files found.", file=sys.stderr)
            sys.exit(1)

    print(f"\nLoading {len(paths)} file(s):")
    records = load_records(paths)
    print(f"\nTotal records loaded: {len(records)}")

    # ── filters ───────────────────────────────────────────────────────────────
    if args.task:
        records = [r for r in records if (r.get("question_type") or r.get("task")) == args.task]
        print(f"Filtered → task={args.task!r}: {len(records)} records")
    if args.model:
        records = [r for r in records if r.get("model_id") == args.model]
        print(f"Filtered → model={args.model!r}: {len(records)} records")
    if not records:
        print("No records after filtering.", file=sys.stderr)
        sys.exit(1)

    # ── per-model / per-task summaries ────────────────────────────────────────
    print_model_breakdown(records)
    print_task_breakdown(records)

    # ── scene-level analysis ──────────────────────────────────────────────────
    scenes  = aggregate_scenes(records)
    ranked_scenes = rank(scenes)
    n = args.top

    print_scene_table(scenes, f"HARDEST {n} SCENES (lowest accuracy)", ranked_scenes[:n])
    print_scene_table(scenes, f"EASIEST {n} SCENES (highest accuracy)", ranked_scenes[-n:][::-1])

    # ── question-level analysis ───────────────────────────────────────────────
    questions = aggregate_questions(records)
    ranked_qs = rank(questions)
    nq = args.top_q if args.top_q is not None else n

    print_question_table(questions, f"HARDEST {nq} QUESTIONS (lowest accuracy)", ranked_qs[:nq])
    print_question_table(questions, f"EASIEST {nq} QUESTIONS (highest accuracy)", ranked_qs[-nq:][::-1])

    # ── overall summary ───────────────────────────────────────────────────────
    overall_correct = sum(int(r.get("correct", False)) for r in records)
    hdr("OVERALL SUMMARY")
    print(f"  Records  : {len(records)}")
    print(f"  Correct  : {overall_correct}  ({overall_correct/len(records):.1%})")
    print(f"  Scenes   : {len(scenes)}  |  acc range [{ranked_scenes[0][1]:.1%} – {ranked_scenes[-1][1]:.1%}]")
    print(f"  Questions: {len(questions)}  |  acc range [{ranked_qs[0][1]:.1%} – {ranked_qs[-1][1]:.1%}]")
    print()


if __name__ == "__main__":
    main()
