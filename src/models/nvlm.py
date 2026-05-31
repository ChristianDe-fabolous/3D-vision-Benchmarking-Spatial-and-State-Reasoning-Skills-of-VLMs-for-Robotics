"""
NVIDIA Nemotron Nano 12B v2 VL wrapper (LLaMA-3.2-Vision architecture).

HF model: nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16

Install:
    pip install transformers torch accelerate
"""

from __future__ import annotations

import logging

import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

from config import MODEL_NVLM_12B, NVLM_INT8_KEYS, NVLM_MAX_NEW_TOKENS, NVLM_MODEL_IDS
from models.base import BaseVLM

logger = logging.getLogger("vlm_bench")


class NemotronVLM(BaseVLM):
    def __init__(
        self,
        model_key: str = MODEL_NVLM_12B,
        max_new_tokens: int = NVLM_MAX_NEW_TOKENS,
    ):
        self.model_id = NVLM_MODEL_IDS[model_key]
        self._int8 = model_key in NVLM_INT8_KEYS
        self.max_new_tokens = max_new_tokens
        self.system_prompt: str | None = None
        self._model = None
        self._processor = None

    def load(self) -> None:
        logger.info(f"Loading {self.model_id} {'(int8)' if self._int8 else '(bf16)'} ...")
        kwargs: dict = dict(device_map="auto")
        if self._int8:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            # Nemotron VL is designed for bfloat16
            kwargs["torch_dtype"] = torch.bfloat16

        self._model = MllamaForConditionalGeneration.from_pretrained(self.model_id, **kwargs)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        logger.info("Model loaded.")

    def _build_messages(self, images: list[Image.Image], prompt: str) -> list[dict]:
        # LLaMA 3.2 Vision format: image dicts in content list, processor inserts tokens
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def infer(self, image: Image.Image | list[Image.Image], prompt: str) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("Call load() before infer().")

        imgs = image if isinstance(image, list) else [image]
        msgs = self._build_messages(imgs, prompt)
        text = self._processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self._processor(
            images=imgs, text=text, return_tensors="pt"
        ).to(next(self._model.parameters()).device)

        prompt_len = inputs["input_ids"].shape[1]
        max_ctx = getattr(self._model.config, "max_position_embeddings", 131072)
        max_new = min(self.max_new_tokens, max(64, max_ctx - prompt_len - 32))
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=max_new)

        return self._decode_outputs(output_ids, prompt_len, self._processor)[0]

    # infer_batch falls back to the base-class loop over infer() —
    # LLaMA Vision batching with mixed image counts is non-trivial.

    def infer_batch_logprobs(
        self,
        batch: list[tuple[list[Image.Image], str]],
        choice_labels: list[list[str]],
    ) -> list[tuple[str, dict]]:
        if self._model is None or self._processor is None:
            raise RuntimeError("Call load() before infer().")

        results = []
        for (imgs, prompt), labels in zip(batch, choice_labels):
            msgs = self._build_messages(imgs, prompt)
            text = self._processor.apply_chat_template(msgs, add_generation_prompt=True)
            inputs = self._processor(
                images=imgs, text=text, return_tensors="pt"
            ).to(next(self._model.parameters()).device)

            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # _logprobs_from_first_token expects (batch, vocab) — unsqueeze for single sample
            first_logits = out.scores[0]  # (1, vocab) for single sample
            result = self._logprobs_from_first_token(first_logits, [labels], self._processor.tokenizer)
            results.append(result[0])

        return results
