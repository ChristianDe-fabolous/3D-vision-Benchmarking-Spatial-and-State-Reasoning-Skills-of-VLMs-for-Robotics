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
import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from PIL import Image

from config import ALLOWED_QUESTION_PATTERNS, INVALID_ENTRIES, QUESTION_TYPES, TASK_FAILURE_MODE, TASK_MULTIVIEW

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


def _classify_question_type(question: str) -> Optional[str]:
    """Return the question type name from QUESTION_TYPES, or None if unmatched/unconfigured."""
    if not QUESTION_TYPES:
        return None
    q = question.lower()
    for type_name, patterns in QUESTION_TYPES.items():
        if any(p.lower() in q for p in patterns):
            return type_name
    return None


def _classify_question_type(question: str, task: str) -> Optional[str]:
    """Return the question type within its task using QUESTION_TYPES, or None if unmatched/unconfigured."""
    task_types = QUESTION_TYPES.get(task, {})
    if not task_types:
        return None
    q = question.lower()
    for type_name, patterns in task_types.items():
        if any(p.lower() in q for p in patterns):
            return type_name
    return None


def _is_allowed_question(question: str, task: str) -> bool:
    """Return True if the question matches the allowlist for its task (or list is empty)."""
    patterns = ALLOWED_QUESTION_PATTERNS.get(task, [])
    if not patterns:
        return True
    q = question.lower()
    return any(p.lower() in q for p in patterns)


def _parse_scene_id(entry_id: str) -> Optional[str]:
    """Extract scene ID from entry ID, e.g. '14346' from 'droid_..._14346_q9'."""
    m = re.search(r'_(\d+)_q\d+$', entry_id)
    return m.group(1) if m else None


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

        if not _is_allowed_question(row["question"], task):
            continue

        try:
            choices = _parse_choices(row["choices"])
        except Exception as e:
            continue  # skip malformed entries

        scene_id = _parse_scene_id(row["id"])
        question_type = _classify_question_type(row["question"], task)
        metadata = {}
        if scene_id:
            metadata["scene_id"] = scene_id
        if question_type is not None:
            metadata["question_type"] = question_type
        yield Sample(
            id=row["id"],
            task=task,
            image=row["image"],
            question=row["question"],
            choices=choices,
            correct_answer=int(row["correct_answer"]),
            metadata=metadata,
        )
        yielded += 1
