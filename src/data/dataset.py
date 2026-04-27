"""
Dataset loading for Robo2VLM-1 (HuggingFace) or a local equivalent.

HuggingFace schema:
  id             str   — unique question identifier
  question       str   — question text
  choices        str   — serialized Python list, e.g. "['Yes', 'No', ...]"
  correct_answer int   — 0-indexed position in choices
  image          PIL   — single image (may be a composite for multiview questions)

Task, question type, and question group are assigned by matching question text
against QUESTION_TYPE_TEMPLATES in config.py. This is the single source of
truth for filtering: questions that match no pattern are skipped entirely.
Types use the format <descriptive_name>_<paper_abbrev> (e.g. gripper_state_RS).
Groups correspond to the three paper categories: spatial_reasoning,
goal_reasoning, interaction_reasoning.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

from PIL import Image

from config import INVALID_ENTRIES, QUESTION_TYPE_EXTRACT, QUESTION_TYPE_GROUPS, QUESTION_TYPE_TEMPLATES, TASK_FAILURE_MODE, TASK_MULTIVIEW


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


def _classify_task_and_types_template(question: str) -> Tuple[Optional[str], List[str]]:
    """
    Alternative to _classify_task_and_types using regex patterns that reproduce
    the exact question templates from robo2vlm's vqa.py generator functions.
    Variable slots (object names, camera names, step indices, language instructions)
    are matched with regex wildcards — see QUESTION_TYPE_TEMPLATES in config.py.

    Advantages over substring matching: more precise, no false positives from
    shared keywords, and directly traceable to the original generator function.
    """
    for task, types in QUESTION_TYPE_TEMPLATES.items():
        matched = [
            type_name
            for type_name, patterns in types.items()
            if any(re.search(p, question, re.IGNORECASE) for p in patterns)
        ]
        if matched:
            return task, matched
    return None, []


def _parse_choices(raw: str) -> List[str]:
    parsed = ast.literal_eval(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"Unexpected choices format: {raw!r}")
    return [str(c) for c in parsed]


def _parse_scene_id(entry_id: str) -> Optional[str]:
    """Extract scene ID (full prefix) from entry ID, e.g. 'droid_..._14346' from 'droid_..._14346_q9'."""
    m = re.match(r'^(.+)_q\d+$', entry_id)
    return m.group(1) if m else None


def _extract_question_parts(question: str, question_types: List[str]) -> dict:
    """Extract named variable slots from a question using QUESTION_TYPE_EXTRACT patterns."""
    parts = {}
    for type_key in question_types:
        for pat in QUESTION_TYPE_EXTRACT.get(type_key, []):
            m = re.search(pat, question, re.IGNORECASE)
            if m:
                parts.update(m.groupdict())
                break
    return parts


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
    skip: int = 0,
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
        skip:        Skip this many rows at the parquet level before iterating
                     (fast-forward for resume; avoids decoding skipped images).

    Yields:
        Sample objects.
    """
    from datasets import load_dataset as hf_load

    # test split: download parquet files — ds.skip() is an efficient seek.
    # train split: stream row-by-row — ds.skip() iterates through rows (O(n)),
    # so for large resume offsets on train, fall back to completed_ids filtering
    # in the pipeline rather than relying on skip for speed.
    streaming = split != "test"

    if local_path:
        ds = hf_load("parquet", data_dir=local_path, split=split, streaming=streaming)
    else:
        ds = hf_load(source, split=split, streaming=streaming)

    # Only use ds.skip() when it's efficient (non-streaming / parquet seek).
    # For streaming datasets skip() is O(n); the pipeline's completed_ids check
    # handles deduplication instead.
    if skip > 0 and not streaming:
        ds = ds.skip(skip)
        raw_offset = skip
    else:
        raw_offset = 0

    yielded = 0
    for raw_idx, row in enumerate(ds):
        if limit is not None and yielded >= limit:
            break

        if _is_invalid(row["id"]):
            continue

        task, question_types = _classify_task_and_types_template(row["question"])
        if task is None:
            continue  # skip unrecognised question types

        if task_filter and task != task_filter:
            continue

        try:
            choices = _parse_choices(row["choices"])
        except Exception as e:
            continue  # skip malformed entries

        scene_id = _parse_scene_id(row["id"])
        metadata = {
            "raw_row": raw_offset + raw_idx,  # absolute raw parquet row index
        }
        if scene_id:
            metadata["scene_id"] = scene_id
        if question_types:
            metadata["question_types"] = question_types
            metadata["question_group"] = QUESTION_TYPE_GROUPS.get(question_types[0])
            parts = _extract_question_parts(row["question"], question_types)
            if parts:
                metadata["question_parts"] = parts
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
