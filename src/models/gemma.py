"""
Gemma 3 multimodal model wrapper (google/gemma-3-*-it).

Install:
    pip install transformers torch accelerate bitsandbytes
"""

from __future__ import annotations

import logging

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

from config import GEMMA_INT8_KEYS, GEMMA_MAX_NEW_TOKENS, GEMMA_MODEL_IDS, MODEL_GEMMA_4B
from models.base import BaseVLM

logger = logging.getLogger("vlm_bench")


class GemmaVLM(BaseVLM):
    def __init__(
        self,
        model_key: str = MODEL_GEMMA_4B,
        max_new_tokens: int = GEMMA_MAX_NEW_TOKENS,
    ):
        self.model_id = GEMMA_MODEL_IDS[model_key]
        self._int8 = model_key in GEMMA_INT8_KEYS
        self.max_new_tokens = max_new_tokens
        self.system_prompt: str | None = None
        self._model = None
        self._processor = None

    def load(self) -> None:
        logger.info(f"Loading {self.model_id} {'(int8)' if self._int8 else ''} ...")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs = dict(device_map="auto")
        if self._int8:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = dtype
        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._processor.tokenizer.padding_side = "left"
        logger.info("Model loaded.")

    def _build_messages(self, images: list[Image.Image], prompt: str) -> list[dict]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def _prepare_inputs(self, batch: list[tuple[list[Image.Image], str]]):
        all_messages = [self._build_messages(imgs, prompt) for imgs, prompt in batch]
        texts = [
            self._processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in all_messages
        ]
        # Collect images in order — one flat list matching text placeholders
        all_images = []
        for imgs, _ in batch:
            all_images.extend(imgs)

        inputs = self._processor(
            text=texts,
            images=all_images if all_images else None,
            padding=True,
            return_tensors="pt",
        ).to(next(self._model.parameters()).device)
        return inputs

    def infer(self, image: Image.Image | list[Image.Image], prompt: str) -> str:
        return self.infer_batch([(image if isinstance(image, list) else [image], prompt)])[0]

    def infer_batch(self, batch: list[tuple[list[Image.Image], str]]) -> list[str]:
        if self._model is None or self._processor is None:
            raise RuntimeError("Call load() before infer().")

        inputs = self._prepare_inputs(batch)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

        return self._decode_outputs(output_ids, prompt_len, self._processor)

    def infer_batch_logprobs(
        self,
        batch: list[tuple[list[Image.Image], str]],
        choice_labels: list[list[str]],
    ) -> list[tuple[str, dict]]:
        if self._model is None or self._processor is None:
            raise RuntimeError("Call load() before infer().")

        inputs = self._prepare_inputs(batch)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
            )

        return self._logprobs_from_first_token(out.scores[0], choice_labels, self._processor.tokenizer)
