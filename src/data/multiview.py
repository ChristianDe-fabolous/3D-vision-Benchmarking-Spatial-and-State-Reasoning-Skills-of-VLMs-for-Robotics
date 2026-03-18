"""
Multiview task — prompt building.

Relevant question types from Robo2VLM-1:
  - 3D Spatial Correspondence: "In the left image (ext1 camera), a red dot is marked.
    Which point in the right image (ext2 camera) corresponds to the same 3D location?"
  - Spatial depth reasoning: "Which colored point is CLOSEST to the camera?"

The image column for these samples is a composite (both views side by side).
"""

from __future__ import annotations

from typing import Optional

from data.dataset import Sample

CHOICE_LABELS = ["A", "B", "C", "D", "E"]


def _build_prompt_default(sample: Sample) -> str:
    choices_text = "\n".join(
        f"  {CHOICE_LABELS[i]}: {choice}"
        for i, choice in enumerate(sample.choices)
    )
    return (
        "You are given an image showing one or more camera perspectives of a robot scene. "
        "Use the spatial information in the image to answer the question.\n\n"
        f"Question: {sample.question}\n\n"
        f"Choices:\n{choices_text}\n\n"
        "Reply with only the letter of the correct answer "
        f"({', '.join(CHOICE_LABELS[:len(sample.choices)])})."
    )


_PROMPT_BUILDERS = {
    "default": _build_prompt_default,
}


def build_prompt(sample: Sample, prompt_id: str = "default") -> str:
    builder = _PROMPT_BUILDERS.get(prompt_id)
    if builder is None:
        raise ValueError(f"Unknown prompt_id for multiview task: {prompt_id!r}")
    return builder(sample)


def response_to_index(response: str, num_choices: int) -> Optional[int]:
    letter = response.strip().upper()[:1]
    if letter in CHOICE_LABELS[:num_choices]:
        return CHOICE_LABELS.index(letter)
    return None
