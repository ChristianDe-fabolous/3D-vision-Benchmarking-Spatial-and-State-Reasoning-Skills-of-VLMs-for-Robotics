"""
Convert eval_results.jsonl produced by scripts/train_lora.py (run_eval) into
the results.jsonl row format produced by src/main.py, so it can be analyzed
with scripts/analyze_results.py.

eval_results.jsonl only stores {id, scene_id, question_type, ground_truth,
response_raw, predicted, correct} — everything else (choices, image_step,
variant, ...) is looked up from the original dataset jsonl by `id`.

Also writes summary.json + question_type_issues.txt next to the output file,
using the same src/evaluation/results.py code path as a normal pipeline run
(test-set only, since eval_results.jsonl only contains the held-out scenes).

Usage:
    python scripts/convert_lora_eval_results.py \
        outputs/lora-qwen3-vl-4b/eval_results.jsonl \
        data/action_phase_dataset.jsonl \
        outputs/lora-qwen3-vl-4b/results.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluation.results import save_summary  # noqa: E402


def strip_choice_prefix(choice: str) -> str:
    # "A. Cannot be determined" -> "Cannot be determined"
    return choice.split(". ", 1)[1] if ". " in choice[:4] else choice


def main() -> None:
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    eval_path, dataset_path, out_path = (Path(p) for p in sys.argv[1:4])

    dataset = {d["id"]: d for d in (json.loads(l) for l in open(dataset_path))}

    rows = []
    for line in open(eval_path):
        e = json.loads(line)
        d = dataset[e["id"]]

        choices = [strip_choice_prefix(c) for c in d["choices"]]
        gt_index = ord(d["answer"]) - ord("A")

        pred_letter = e["predicted"]
        pred_index = ord(pred_letter) - ord("A") if pred_letter in "ABCD" and (ord(pred_letter) - ord("A")) < len(choices) else None
        pred_label = choices[pred_index] if pred_index is not None else None

        rows.append({
            "entry_id": str(e["id"]),
            "task": d["question_type"],
            "question": d["question"],
            "choices": choices,
            "ground_truth_index": gt_index,
            "ground_truth_label": choices[gt_index],
            "response_raw": e["response_raw"],
            "predicted_index": pred_index,
            "predicted_label": pred_label,
            "correct": e["correct"],
            "image_paths": d["images"],
            "scene_id": d["scene_id"],
            "question_type": d["question_type"],
            "variant": d.get("variant"),
            "task_desc": d.get("task"),
            "answer_text": d.get("answer_text"),
            "view": d.get("view"),
            "image_step": d.get("image_step"),
            "image_phase": d.get("image_phase"),
            "tile_ids": d.get("tile_ids"),
            "original_id": d.get("original_id"),
            "special_image": d.get("special_image"),
            "label_phase": d.get("label_phase"),
            "label_step": d.get("label_step"),
        })

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(rows)} rows to {out_path}")

    save_summary(out_path.parent, rows)


if __name__ == "__main__":
    main()
