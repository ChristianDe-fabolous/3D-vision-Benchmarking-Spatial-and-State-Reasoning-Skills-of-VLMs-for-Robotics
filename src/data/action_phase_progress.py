"""
Dataset loader for the progress-detection subset of action_phase_dataset.jsonl.

Split logic:
  Train  — entries where step_a AND step_b are both in {0, 1, 2}
  Test   — entries where at least one of step_a / step_b is > 2
  Ignored — entries where step_b is None (cross-scene "random scene" distractors)

Scenes that only have steps 0-2 appear in train only (no test entries).

Usage:
    from data.action_phase_progress import ProgressDataset

    ds = ProgressDataset(dataset_path="data/action_phase_dataset.jsonl")
    train = ds.train()   # list[Sample]
    test  = ds.test()    # list[Sample]
    print(ds.summary())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from PIL import Image

from data.dataset import Sample

CHOICE_LABELS = ["A", "B", "C", "D", "E"]

# Steps 0, 1, 2 (0-indexed) = "first three images" used for training
TRAIN_MAX_STEP = 2


def _parse_choices(raw: list[str]) -> list[str]:
    """Strip leading 'A. ' labels from stored choice strings."""
    result = []
    for c in raw:
        if len(c) > 2 and c[1] == "." and c[2] == " ":
            result.append(c[3:])
        else:
            result.append(c)
    return result


def _answer_index(answer_letter: str, choices: list[str]) -> int:
    idx = ord(answer_letter.strip().upper()) - ord("A")
    return max(0, min(idx, len(choices) - 1))


def _load_image(path: str, image_root: Path) -> Optional[Image.Image]:
    p = Path(path)
    if not p.is_absolute():
        p = image_root / p
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


def _entry_to_sample(entry: dict, image_root: Path) -> Optional[Sample]:
    """Convert one jsonl progress entry to a Sample with two images."""
    images = []
    image_paths = []
    for img_path in entry.get("images", []):
        img = _load_image(img_path, image_root)
        if img is None:
            return None
        images.append(img)
        image_paths.append(img_path)

    if len(images) < 2:
        return None

    raw_choices = entry.get("choices", [])
    choices = _parse_choices(raw_choices)
    if not choices:
        return None

    answer_letter = entry.get("answer", "A")
    correct_idx = _answer_index(answer_letter, choices)

    return Sample(
        id=str(entry["id"]),
        task="progress",
        image=images[0],
        images=images,
        question=entry.get("question", ""),
        choices=choices,
        correct_answer=correct_idx,
        metadata={
            "scene_id":    entry.get("scene_id", ""),
            "task_desc":   entry.get("task", ""),
            "step_a":      entry.get("step_a"),
            "step_b":      entry.get("step_b"),
            "phase_a":     entry.get("phase_a"),
            "phase_b":     entry.get("phase_b"),
            "variant":     entry.get("variant"),
            "answer_text": entry.get("answer_text", ""),
            "image_paths": image_paths,
            "view":        entry.get("view", "combined"),
        },
    )


@dataclass
class ProgressDataset:
    """
    Loads and splits progress-detection entries from action_phase_dataset.jsonl.

    Args:
        dataset_path: Path to action_phase_dataset.jsonl (absolute or relative to cwd).
        image_root:   Root directory for resolving relative image paths.
                      Defaults to the project root (two levels above src/).
    """

    dataset_path: str
    image_root: Optional[str] = None

    _train: List[Sample] = field(default_factory=list, init=False, repr=False)
    _test:  List[Sample] = field(default_factory=list, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    def _resolve_image_root(self) -> Path:
        if self.image_root:
            return Path(self.image_root)
        # Assume this file lives in src/data/ — project root is two levels up
        return Path(__file__).parent.parent.parent

    def _load(self) -> None:
        if self._loaded:
            return

        image_root = self._resolve_image_root()
        path = Path(self.dataset_path)
        if not path.is_absolute():
            path = Path.cwd() / path

        train_raw, test_raw = [], []

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)

                if e.get("question_type") != "progress":
                    continue

                step_a = e.get("step_a")
                step_b = e.get("step_b")

                # Skip cross-scene distractors (step_b=None)
                if not isinstance(step_a, int) or not isinstance(step_b, int):
                    continue

                sample = _entry_to_sample(e, image_root)
                if sample is None:
                    continue

                if step_a <= TRAIN_MAX_STEP and step_b <= TRAIN_MAX_STEP:
                    train_raw.append(sample)
                else:
                    test_raw.append(sample)

        self._train = train_raw
        self._test = test_raw
        self._loaded = True

    def train(self) -> List[Sample]:
        """Samples whose step_a and step_b are both in {0, 1, 2}."""
        self._load()
        return self._train

    def test(self) -> List[Sample]:
        """Samples where at least one of step_a/step_b is > 2."""
        self._load()
        return self._test

    def all(self) -> List[Sample]:
        """All valid same-scene progress samples (train + test)."""
        self._load()
        return self._train + self._test

    def train_scenes(self) -> set[str]:
        self._load()
        return {s.metadata["scene_id"] for s in self._train}

    def test_scenes(self) -> set[str]:
        self._load()
        return {s.metadata["scene_id"] for s in self._test}

    def train_only_scenes(self) -> set[str]:
        """Scenes that appear in train but have no test entries (≤3 total steps)."""
        self._load()
        return self.train_scenes() - self.test_scenes()

    def summary(self) -> str:
        self._load()
        from collections import Counter
        train_answers = Counter(s.metadata["answer_text"] for s in self._train)
        test_answers  = Counter(s.metadata["answer_text"] for s in self._test)
        train_pairs   = Counter(
            (s.metadata["step_a"], s.metadata["step_b"]) for s in self._train
        )
        return (
            f"ProgressDataset\n"
            f"  Train : {len(self._train):4d} samples  answers={dict(train_answers)}\n"
            f"  Test  : {len(self._test):4d} samples  answers={dict(test_answers)}\n"
            f"  Train-only scenes (≤3 steps): {len(self.train_only_scenes())}\n"
            f"  Train step pairs: {dict(sorted(train_pairs.items()))}\n"
        )


def build_prompt(sample: Sample) -> str:
    """
    Prompt for a two-image progress question.
    The question text is already stored verbatim from the dataset generator,
    so this just wraps it with the image context header used by ActionPhaseTask.
    """
    labels = CHOICE_LABELS[: len(sample.choices)]
    label_list = ", ".join(labels[:-1]) + f", or {labels[-1]}"
    label_eg   = ", ".join(labels)

    instruction = (
        f"Answer the following multiple choice question by selecting the letter "
        f"({label_list}). ONLY output the correct option letter, i.e., {label_eg}."
    )
    image_ctx = "You are given 2 images taken at different timesteps of a robot task."

    inline_choices = "".join(
        f" {labels[i]}. {c}" for i, c in enumerate(sample.choices)
    )
    q = f"{sample.question}{inline_choices}"

    return f"{image_ctx} {instruction} {q}"
