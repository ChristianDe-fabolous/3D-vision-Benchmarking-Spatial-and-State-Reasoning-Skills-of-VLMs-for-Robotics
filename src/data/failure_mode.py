"""
Failure mode task — prompt building.

Relevant question types from Robo2VLM-1:
  - Goal-conditioned: "Has the robot successfully completed the task?"
  - Goal state identification: "Which configuration shows the goal state?"
"""

from __future__ import annotations

from typing import Optional

from data.dataset import Sample

# Letters used to label choices in prompts (dataset has up to 5 options)
CHOICE_LABELS = ["A", "B", "C", "D", "E"]


def build_prompt(sample: Sample) -> str:
    choices_text = "\n".join(
        f"  {CHOICE_LABELS[i]}: {choice}"
        for i, choice in enumerate(sample.choices)
    )
    return (
        "You are evaluating a robot performing a manipulation task. "
        "Analyse the image carefully.\n\n"
        f"Question: {sample.question}\n\n"
        f"Choices:\n{choices_text}\n\n"
        "Reply with only the letter of the correct answer "
        f"({', '.join(CHOICE_LABELS[:len(sample.choices)])})."
    )


def response_to_index(response: str, num_choices: int) -> Optional[int]:
    """Parse a letter response (e.g. 'B') back to a 0-based index."""
    letter = response.strip().upper()[:1]
    if letter in CHOICE_LABELS[:num_choices]:
        return CHOICE_LABELS.index(letter)
    return None
