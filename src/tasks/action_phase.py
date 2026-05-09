"""
Task loader for the locally built action-phase dataset (action_phase_dataset.jsonl).

Each entry has:
  id, scene_id, question_type, task, question, images (list of paths),
  choices (labelled strings), answer (letter), answer_text.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image

from data.dataset import Sample
from tasks.base import BaseTask, CHOICE_LABELS


class ActionPhaseTask(BaseTask):
    def __init__(
        self,
        dataset_path: str,
        question_type: Optional[str] = None,
        limit: Optional[int] = None,
        prompt_id: str = "default",
        image_root: Optional[str] = None,
        describe: bool = False,
    ):
        self.dataset_path  = Path(dataset_path)
        self.question_type = question_type
        self.limit         = limit
        self.prompt_id     = prompt_id
        self.describe      = describe
        # Root for resolving relative image paths.
        # Defaults to the project root (two levels above this file).
        self.image_root = Path(image_root) if image_root else Path(__file__).parent.parent.parent

    def get_samples(self, skip: int = 0) -> Iterator[Sample]:
        yielded = 0
        with open(self.dataset_path, encoding="utf-8") as f:
            for raw_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if raw_idx < skip:
                    continue
                if self.limit is not None and yielded >= self.limit:
                    break

                e = json.loads(line)

                if self.question_type and e.get("question_type") != self.question_type:
                    continue

                # Parse choices — stored as ["A. Yes", "B. No", ...]
                choices = [c[3:] if len(c) > 2 and c[1] == "." else c
                           for c in e.get("choices", [])]

                # Correct answer: letter → index
                answer_letter = e.get("answer", "A")
                correct_idx   = ord(answer_letter) - ord("A")

                # Load images — resolve relative paths from image_root
                images = []
                for img_path in e.get("images", []):
                    p = Path(img_path)
                    if not p.is_absolute():
                        p = self.image_root / p
                    try:
                        images.append(Image.open(p).convert("RGB"))
                    except Exception:
                        pass
                if not images:
                    continue

                yield Sample(
                    id=str(e["id"]),
                    task=e.get("question_type", "action_phase"),
                    image=images[0],
                    images=images,
                    question=e.get("question", ""),
                    choices=choices,
                    correct_answer=correct_idx,
                    metadata={
                        "raw_row":       raw_idx,
                        # identity
                        "scene_id":      e.get("scene_id", ""),
                        "question_type": e.get("question_type", ""),
                        "variant":       e.get("variant"),
                        "task_desc":     e.get("task", ""),
                        # ground truth
                        "answer_text":   e.get("answer_text", ""),
                        # image provenance
                        "image_step":    e.get("image_step"),
                        "image_phase":   e.get("image_phase"),
                        "tile_ids":      e.get("tile_ids"),
                        "original_id":   e.get("original_id"),
                        "special_image": e.get("special_image"),
                        # Q1 / Q5 — claimed phase being tested
                        "label_phase":   e.get("label_phase"),
                        "label_step":    e.get("label_step"),
                        # Q3 — claimed current phase (variant C)
                        "claimed_phase": e.get("claimed_phase"),
                        # Q2 — both image steps
                        "step_a":        e.get("step_a"),
                        "step_b":        e.get("step_b"),
                        "phase_a":       e.get("phase_a"),
                        "phase_b":       e.get("phase_b"),
                        "special_a":     e.get("special_a"),
                        "special_b":     e.get("special_b"),
                    },
                )
                yielded += 1

    def build_prompt(self, sample: Sample) -> str:
        lines = [sample.question, ""]
        for i, choice in enumerate(sample.choices):
            lines.append(f"{CHOICE_LABELS[i]}. {choice}")
        if self.describe:
            lines += ["", "First briefly describe what you observe in the image(s). Then answer with the letter of the correct choice only."]
        else:
            lines += ["", "Answer with the letter of the correct choice only."]
        return "\n".join(lines)
