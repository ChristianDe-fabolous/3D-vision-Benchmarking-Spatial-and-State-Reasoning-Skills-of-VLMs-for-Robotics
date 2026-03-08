"""
Dataset loading for Robo2VLM-1 (HuggingFace) or a local equivalent.

HuggingFace schema:
  id             str   — unique question identifier
  question       str   — question text
  choices        str   — serialized Python list, e.g. "['Yes', 'No', ...]"
  correct_answer int   — 0-indexed position in choices
  image          PIL   — single image (may be a composite for multiview questions)

We assign tasks by matching question text patterns since the dataset has no
explicit task column.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from PIL import Image

from config import INVALID_ENTRIES, TASK_FAILURE_MODE, TASK_MULTIVIEW

# Keywords used to classify samples into tasks.
# A sample matches a task if ANY of its keywords appear in the question (case-insensitive).
TASK_KEYWORDS = {
    TASK_FAILURE_MODE: [
        "successfully completed",
        "goal state",
        "task was",
        "has the robot",
    ],
    TASK_MULTIVIEW: [
        "left image",
        "right image",
        "ext1",
        "ext2",
        "3d location",
        "same 3d",
        "corresponding",
    ],
}


@dataclass
class Sample:
    id: str
    task: str
    image: Image.Image
    question: str
    choices: List[str]          # parsed list, e.g. ["Yes", "No", "Cannot be determined"]
    correct_answer: int         # 0-indexed
    metadata: dict = field(default_factory=dict)

    @property
    def correct_choice(self) -> str:
        return self.choices[self.correct_answer]


def _classify_task(question: str) -> Optional[str]:
    """Return the task name based on question content, or None if unrecognised."""
    q = question.lower()
    for task, keywords in TASK_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return task
    return None


def _parse_choices(raw: str) -> List[str]:
    parsed = ast.literal_eval(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"Unexpected choices format: {raw!r}")
    return [str(c) for c in parsed]


def _is_invalid(entry_id: str) -> bool:
    """Check against INVALID_ENTRIES (stored as ints for numeric IDs)."""
    try:
        return int(entry_id) in INVALID_ENTRIES
    except (ValueError, TypeError):
        return False


def load_dataset(
    source: str = "keplerccc/Robo2VLM-1",
    split: str = "test",
    task_filter: Optional[str] = None,
    limit: Optional[int] = None,
    local_path: Optional[str] = None,
) -> Iterator[Sample]:
    """
    Load samples from Robo2VLM-1, lazily via a generator to avoid
    pulling 107 GB into memory at once.

    Args:
        source:      HuggingFace dataset repo id (ignored if local_path given).
        split:       "train" or "test".
        task_filter: If set, yield only samples matching this task.
        limit:       Stop after this many yielded samples.
        local_path:  If given, load from a local directory instead of HF.

    Yields:
        Sample objects.
    """
    from datasets import load_dataset as hf_load

    if local_path:
        ds = hf_load("parquet", data_dir=local_path, split=split, streaming=True)
    else:
        ds = hf_load(source, split=split, streaming=True)

    yielded = 0
    for row in ds:
        if limit is not None and yielded >= limit:
            break

        if _is_invalid(row["id"]):
            continue

        task = _classify_task(row["question"])
        if task is None:
            continue  # skip unrecognised question types

        if task_filter and task != task_filter:
            continue

        try:
            choices = _parse_choices(row["choices"])
        except Exception as e:
            continue  # skip malformed entries

        yield Sample(
            id=row["id"],
            task=task,
            image=row["image"],
            question=row["question"],
            choices=choices,
            correct_answer=int(row["correct_answer"]),
        )
        yielded += 1
