"""
Dataset loading for Robo2VLM-1 (HuggingFace) or a local equivalent.

HuggingFace schema:
  id             str   — unique question identifier
  question       str   — question text
  choices        str   — serialized Python list, e.g. "['Yes', 'No', ...]"
  correct_answer int   — 0-indexed position in choices
  image          PIL   — single image (may be a composite for multiview questions)

Task and question type are assigned together by matching question text against
QUESTION_TYPES in config.py (task -> type_name -> keyword patterns). This is
the single source of truth for filtering: questions that match no pattern are
skipped entirely.

When QUESTION_TYPES is not yet configured (empty), TASK_KEYWORDS below is used
as a fallback for task assignment only (no type is assigned). Once QUESTION_TYPES
is fully populated, TASK_KEYWORDS becomes unused and can be removed.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

from PIL import Image

from config import INVALID_ENTRIES, QUESTION_TYPES, TASK_FAILURE_MODE, TASK_MULTIVIEW

# Fallback used for task assignment when QUESTION_TYPES is not yet configured.
# Once QUESTION_TYPES is filled in config.py, this is no longer needed.
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


def _classify_task_and_type(question: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (task, question_type) for a question.

    Primary: match against QUESTION_TYPES (config.py) — gives both task and type
    in one step. A question that matches no pattern is skipped (returns None, None).

    Fallback: when QUESTION_TYPES is empty/unconfigured, match against TASK_KEYWORDS
    for task assignment only (question_type will be None).
    """
    q = question.lower()

    for task, types in QUESTION_TYPES.items():
        for type_name, patterns in types.items():
            if any(p.lower() in q for p in patterns):
                return task, type_name

    # Fallback — remove once QUESTION_TYPES is fully configured
    for task, keywords in TASK_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return task, None

    return None, None


def _parse_choices(raw: str) -> List[str]:
    parsed = ast.literal_eval(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"Unexpected choices format: {raw!r}")
    return [str(c) for c in parsed]


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

        task, question_type = _classify_task_and_type(row["question"])
        if task is None:
            continue  # skip unrecognised question types

        if task_filter and task != task_filter:
            continue

        try:
            choices = _parse_choices(row["choices"])
        except Exception as e:
            continue  # skip malformed entries

        scene_id = _parse_scene_id(row["id"])
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
