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


def _build_prompt_default(sample: Sample) -> str:
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


def _build_prompt_paper(sample: Sample) -> str:
    """Replicates the exact prompt format from the Robo2VLM-1 paper evaluation."""
    choice_labels = ["A", "B", "C", "D", "E"]
    inline_choices = "".join(
        f" {choice_labels[i]}. {choice}"
        for i, choice in enumerate(sample.choices)
    )
    formatted_question = f"{sample.question}{inline_choices}"
    return (
        f"Answer the following multiple choice question by selecting the letter "
        f"(A, B, C, D, or E). ONLY output the correct option letter, i.e., A, B, C, D, E. "
        f"{formatted_question}"
    )


def _build_prompt_test(sample: Sample) -> str:
    """Dummy prompt for quick experiments — change freely."""
    choices_text = "\n".join(
        f"  {CHOICE_LABELS[i]}: {choice}"
        for i, choice in enumerate(sample.choices)
    )
    return (
        "Look at the image.\n\n"
        f"Question: {sample.question}\n\n"
        f"Choices:\n{choices_text}\n\n"
        "Describe what is in the image and what you can read. "
        "Also answer: What did I give you as an exercise before? "
        f"({', '.join(CHOICE_LABELS[:len(sample.choices)])})."
    )


_PROMPT_BUILDERS = {
    "default": _build_prompt_default,
    "paper": _build_prompt_paper,
    "test": _build_prompt_test,
}


def build_prompt(sample: Sample, prompt_id: str = "default") -> str:
    builder = _PROMPT_BUILDERS.get(prompt_id)
    if builder is None:
        raise ValueError(f"Unknown prompt_id for failure_mode task: {prompt_id!r}")
    return builder(sample)


def response_to_index(response: str, num_choices: int) -> Optional[int]:
    """Parse a letter response (e.g. 'B') back to a 0-based index."""
    letter = response.strip().upper()[:1]
    if letter in CHOICE_LABELS[:num_choices]:
        return CHOICE_LABELS.index(letter)
    return None
