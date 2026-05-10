from abc import ABC, abstractmethod
from typing import List, Tuple

from PIL import Image


class BaseVLM(ABC):
    """Abstract interface for all VLMs."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""

    @abstractmethod
    def infer(self, image: Image.Image, prompt: str) -> str:
        """
        Run inference on a PIL image + text prompt.

        Args:
            image:  PIL Image (may be a composite for multiview samples).
            prompt: Text prompt with question and labelled choices.

        Returns:
            Raw string response from the model (expected: a single letter).
        """

    def infer_batch(self, batch: List[Tuple[List[Image.Image], str]]) -> List[str]:
        """
        Run inference on a batch of (images, prompt) pairs.
        Default: loop over infer(). Override for true batched GPU inference.
        """
        return [self.infer(images, prompt) for images, prompt in batch]
