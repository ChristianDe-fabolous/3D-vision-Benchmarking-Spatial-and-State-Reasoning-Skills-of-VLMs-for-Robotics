"""
Microsoft Phi-3.5-Vision / Phi-4-Vision wrapper.

Install:
    pip install transformers torch accelerate
    pip install flash-attn  # optional — auto-detected, falls back to eager
"""

from __future__ import annotations

import logging

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from config import MODEL_PHI35_VISION, PHI_INT8_KEYS, PHI_MAX_NEW_TOKENS, PHI_MODEL_IDS
from models.base import BaseVLM

logger = logging.getLogger("vlm_bench")


class PhiVLM(BaseVLM):
    def __init__(
        self,
        model_key: str = MODEL_PHI35_VISION,
        max_new_tokens: int = PHI_MAX_NEW_TOKENS,
    ):
        self.model_id = PHI_MODEL_IDS[model_key]
        self._int8 = model_key in PHI_INT8_KEYS
        self.max_new_tokens = max_new_tokens
        self.system_prompt: str | None = None
        self._model = None
        self._processor = None

    def load(self) -> None:
        logger.info(f"Loading {self.model_id} {'(int8)' if self._int8 else ''} ...")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "eager"

        kwargs: dict = dict(device_map="auto", trust_remote_code=True, _attn_implementation=attn_impl)
        if self._int8:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = dtype

        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._processor.tokenizer.padding_side = "left"
        logger.info("Model loaded.")

    def _build_messages(self, images: list[Image.Image], prompt: str) -> list[dict]:
        # Phi embeds image references as numbered tokens inside the user text
        image_tokens = "".join(f"<|image_{i + 1}|>\n" for i in range(len(images)))
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": image_tokens + prompt})
        return messages

    def _apply_template(self, messages: list[dict]) -> str:
        return self._processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def infer(self, image: Image.Image | list[Image.Image], prompt: str) -> str:
        return self.infer_batch([(image if isinstance(image, list) else [image], prompt)])[0]

    def infer_batch(self, batch: list[tuple[list[Image.Image], str]]) -> list[str]:
        if self._model is None or self._processor is None:
            raise RuntimeError("Call load() before infer().")

        texts = []
        images_per_sample: list[list[Image.Image]] = []
        for imgs, prompt in batch:
            texts.append(self._apply_template(self._build_messages(imgs, prompt)))
            images_per_sample.append(list(imgs))

        inputs = self._processor(
            text=texts,
            images=images_per_sample if any(images_per_sample) else None,
            padding=True,
            return_tensors="pt",
        ).to(next(self._model.parameters()).device)

        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        return self._decode_outputs(output_ids, prompt_len, self._processor)

    def infer_batch_logprobs(
        self,
        batch: list[tuple[list[Image.Image], str]],
        choice_labels: list[list[str]],
    ) -> list[tuple[str, dict]]:
        if self._model is None or self._processor is None:
            raise RuntimeError("Call load() before infer().")

        texts = []
        images_per_sample: list[list[Image.Image]] = []
        for imgs, prompt in batch:
            texts.append(self._apply_template(self._build_messages(imgs, prompt)))
            images_per_sample.append(list(imgs))

        inputs = self._processor(
            text=texts,
            images=images_per_sample if any(images_per_sample) else None,
            padding=True,
            return_tensors="pt",
        ).to(next(self._model.parameters()).device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
            )

        return self._logprobs_from_first_token(out.scores[0], choice_labels, self._processor.tokenizer)
