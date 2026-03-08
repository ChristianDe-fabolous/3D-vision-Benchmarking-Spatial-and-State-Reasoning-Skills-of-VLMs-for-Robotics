from abc import ABC, abstractmethod
from typing import Iterator, Optional

from data.dataset import Sample

CHOICE_LABELS = ["A", "B", "C", "D", "E"]


class BaseTask(ABC):
    """Abstract interface for evaluation tasks."""

    @abstractmethod
    def get_samples(self) -> Iterator[Sample]:
        """Yield prepared Sample objects for this task."""

    @abstractmethod
    def build_prompt(self, sample: Sample) -> str:
        """Return the text prompt for a given sample."""

    def parse_response(self, response: str, sample: Sample) -> Optional[int]:
        """
        Parse the model's letter response to a 0-based index.
        Returns None if the response cannot be parsed.
        """
        letter = response.strip().upper()[:1]
        labels = CHOICE_LABELS[: len(sample.choices)]
        if letter in labels:
            return labels.index(letter)
        return None

    def evaluate(self, response: str, sample: Sample) -> bool:
        predicted = self.parse_response(response, sample)
        if predicted is None:
            return False
        return predicted == sample.correct_answer
