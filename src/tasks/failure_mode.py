from __future__ import annotations

from typing import Iterator, Optional

import data.failure_mode as fm_data
from config import QUESTION_TYPE_TEMPLATES, TASK_FAILURE_MODE
from data.dataset import Sample, load_dataset
from data.failure_mode import split_tiled_image
from tasks.base import BaseTask

_FAILURE_MODE_QUESTION_TYPES = set(QUESTION_TYPE_TEMPLATES[TASK_FAILURE_MODE].keys())


class FailureModeTask(BaseTask):
    def __init__(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        local_path: Optional[str] = None,
        prompt_id: str = "default",
        individual_images: bool = False,
        smoke: bool = False,
    ):
        self.split = split
        self.limit = limit
        self.local_path = local_path
        self.prompt_id = prompt_id
        self.individual_images = individual_images
        self.smoke = smoke

    def _apply_individual_images(self, sample: Sample) -> Sample:
        if self.individual_images:
            sample.images = split_tiled_image(sample.image)
        return sample

    def get_samples(self, skip: int = 0) -> Iterator[Sample]:
        if self.smoke:
            yield from self._smoke_samples()
            return
        for sample in load_dataset(
            split=self.split,
            task_filter=TASK_FAILURE_MODE,
            limit=self.limit,
            local_path=self.local_path,
            skip=skip,
        ):
            yield self._apply_individual_images(sample)

    def _smoke_samples(self) -> Iterator[Sample]:
        """Stream until one sample per question type is collected (or dataset exhausted)."""
        seen_types: set[str] = set()
        for sample in load_dataset(
            split=self.split,
            task_filter=TASK_FAILURE_MODE,
            limit=None,
            local_path=self.local_path,
            skip=0,
        ):
            qtypes = set(sample.metadata.get("question_types", []))
            new_types = qtypes - seen_types
            if new_types:
                seen_types |= new_types
                yield self._apply_individual_images(sample)
            if seen_types >= _FAILURE_MODE_QUESTION_TYPES:
                break

    def build_prompt(self, sample: Sample) -> str:
        return fm_data.build_prompt(sample, self.prompt_id)
