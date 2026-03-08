"""
Qwen2.5-VL model wrapper.

Install:
    pip install transformers torch accelerate qwen-vl-utils
"""

from __future__ import annotations

import logging

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from config import QWEN_MAX_NEW_TOKENS, QWEN_MODEL_ID
from models.base import BaseVLM

logger = logging.getLogger("vlm_bench")


class QwenVLM(BaseVLM):
    def __init__(
        self,
        model_id: str = QWEN_MODEL_ID,
        device: str | None = None,
        max_new_tokens: int = QWEN_MAX_NEW_TOKENS,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def load(self) -> None:
        logger.info(f"Loading {self.model_id} on {self.device} ...")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        logger.info("Model loaded.")

    def infer(self, image: Image.Image, prompt: str) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("Call load() before infer().")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info
        image_inputs, _ = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

        generated = output_ids[0][inputs.input_ids.shape[1]:]
        return self._processor.decode(generated, skip_special_tokens=True).strip()
