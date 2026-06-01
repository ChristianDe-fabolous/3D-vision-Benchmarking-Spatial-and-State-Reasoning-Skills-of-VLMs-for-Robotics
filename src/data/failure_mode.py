"""
Failure mode task — prompt building.

Relevant question types from Robo2VLM-1:
  - Goal-conditioned: "Has the robot successfully completed the task?"
  - Goal state identification: "Which configuration shows the goal state?"
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image

from data.dataset import Sample


_WIDE_ASPECT_THRESHOLD = 3.0  # matches tile_multiview.py


def split_tiled_image(img: Image.Image) -> List[Image.Image]:
    w, h = img.size
    if w / h >= _WIDE_ASPECT_THRESHOLD:
        mw = w // 2
        return [img.crop((0, 0, mw, h)), img.crop((mw, 0, w, h))]
    mw, mh = w // 2, h // 2
    return [
        img.crop((0,   0,   mw, mh)),
        img.crop((mw,  0,   w,  mh)),
        img.crop((0,   mh,  mw, h)),
        img.crop((mw,  mh,  w,  h)),
    ]

# Letters used to label choices in prompts (dataset has up to 5 options)
CHOICE_LABELS = ["A", "B", "C", "D", "E"]


def _image_context(sample: Sample) -> str:
    n = len(sample.all_images)
    if n > 1:
        return f"You are given {n} camera views of the robot scene as separate images."
    return "The image shows multiple camera views of the robot scene tiled into a single composite."


def _build_prompt_default(sample: Sample) -> str:
    choices_text = "\n".join(
        f"  {CHOICE_LABELS[i]}: {choice}"
        for i, choice in enumerate(sample.choices)
    )
    n = len(sample.all_images)
    image_word = "images" if n > 1 else "image"
    return (
        f"{_image_context(sample)} "
        f"You are evaluating a robot performing a manipulation task. "
        f"Analyse the {image_word} carefully.\n\n"
        f"Question: {sample.question}\n\n"
        f"Choices:\n{choices_text}\n\n"
        "Reply with only the letter of the correct answer "
        f"({', '.join(CHOICE_LABELS[:len(sample.choices)])})."
    )


def _build_prompt_paper(sample: Sample) -> str:
    """Replicates the exact prompt format from the Robo2VLM-1 paper evaluation."""
    inline_choices = "".join(
        f" {CHOICE_LABELS[i]}. {choice}"
        for i, choice in enumerate(sample.choices)
    )
    formatted_question = f"{sample.question}{inline_choices}"
    return (
        f"{_image_context(sample)} "
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


def _build_prompt_paper_cot(sample: Sample) -> str:
    """Paper prompt with Chain-of-Thought reasoning before the final answer."""
    inline_choices = "".join(
        f" {CHOICE_LABELS[i]}. {choice}"
        for i, choice in enumerate(sample.choices)
    )
    formatted_question = f"{sample.question}{inline_choices}"
    return (
        f"{_image_context(sample)} "
        f"Answer the following multiple choice question. "
        f"Think step by step, then output your final answer as a single letter "
        f"(A, B, C, D, or E) on the last line. "
        f"{formatted_question}"
    )


def _build_prompt_smoke(sample: Sample) -> str:
    n = len(sample.all_images)
    if n > 1:
        return f"You are given {n} images. Describe what you observe in each image."
    return "Describe what you observe in the image."


_PROMPT_BUILDERS = {
    "default": _build_prompt_default,
    "paper": _build_prompt_paper,
    "paper_cot": _build_prompt_paper_cot,
    "test": _build_prompt_test,
    "smoke": _build_prompt_smoke,
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
