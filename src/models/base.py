from abc import ABC, abstractmethod

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
