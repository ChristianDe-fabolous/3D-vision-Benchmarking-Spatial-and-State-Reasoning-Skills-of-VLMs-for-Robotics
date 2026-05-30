from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
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

    def infer_batch_logprobs(
        self,
        batch: List[Tuple[List[Image.Image], str]],
        choice_labels: List[List[str]],
    ) -> List[Tuple[str, dict]]:
        """
        Return (predicted_letter, {label: prob}) for each sample.
        prob is normalised over the given choice labels only.
        Default: falls back to infer_batch with empty prob dicts.
        """
        responses = self.infer_batch(batch)
        return [(r, {}) for r in responses]

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _decode_outputs(output_ids: torch.Tensor, prompt_len: int, processor) -> List[str]:
        """Slice generated tokens past the prompt and decode to strings."""
        return [
            processor.decode(out[prompt_len:], skip_special_tokens=True).strip()
            for out in output_ids
        ]

    @staticmethod
    def _logprobs_from_first_token(
        first_logits: torch.Tensor,          # (batch, vocab)
        choice_labels: List[List[str]],
        tokenizer,
    ) -> List[Tuple[str, dict]]:
        """Score answer choices by the logprob of their single token at generation step 0."""
        results = []
        for i, labels in enumerate(choice_labels):
            token_ids: dict[str, int] = {}
            for label in labels:
                for form in (label, f" {label}"):
                    ids = tokenizer.encode(form, add_special_tokens=False)
                    if len(ids) == 1:
                        token_ids[label] = ids[0]
                        break
            if not token_ids:
                results.append(("", {}))
                continue
            valid = [l for l in labels if l in token_ids]
            ids = [token_ids[l] for l in valid]
            probs = torch.softmax(first_logits[i, ids], dim=-1).tolist()
            prob_dict = dict(zip(valid, probs))
            results.append((max(prob_dict, key=prob_dict.__getitem__), prob_dict))
        return results
