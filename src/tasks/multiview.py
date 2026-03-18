from __future__ import annotations

from typing import Iterator, Optional

import data.multiview as mv_data
from config import TASK_MULTIVIEW
from data.dataset import Sample, load_dataset
from tasks.base import BaseTask


class MultiviewTask(BaseTask):
    def __init__(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        local_path: Optional[str] = None,
        prompt_id: str = "default",
    ):
        self.split = split
        self.limit = limit
        self.local_path = local_path
        self.prompt_id = prompt_id

    def get_samples(self) -> Iterator[Sample]:
        return load_dataset(
            split=self.split,
            task_filter=TASK_MULTIVIEW,
            limit=self.limit,
            local_path=self.local_path,
        )

    def build_prompt(self, sample: Sample) -> str:
        return mv_data.build_prompt(sample, self.prompt_id)
