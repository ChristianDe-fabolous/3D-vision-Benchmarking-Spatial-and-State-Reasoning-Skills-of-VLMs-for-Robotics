"""
NVIDIA Nemotron Nano VL 12B V2 wrapper.

HF model: nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16

Install:
    pip install causal_conv1d "transformers>4.53,<4.54" torch timm "mamba-ssm==2.2.5" accelerate open_clip_torch numpy pillow
"""

from __future__ import annotations

import logging

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from config import MODEL_NVLM_12B, NVLM_MAX_NEW_TOKENS, NVLM_MODEL_IDS
from models.base import BaseVLM

logger = logging.getLogger("vlm_bench")

_NO_THINK = "/no_think"


class NemotronVLM(BaseVLM):
    def __init__(
        self,
        model_key: str = MODEL_NVLM_12B,
        max_new_tokens: int = NVLM_MAX_NEW_TOKENS,
    ):
        self.model_id = NVLM_MODEL_IDS[model_key]
        self.max_new_tokens = max_new_tokens
        self.system_prompt: str | None = None
        self._model = None
        self._processor = None
        self._tokenizer = None

    def load(self) -> None:
        logger.info(f"Loading {self.model_id} ...")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        logger.info("Model loaded.")

    def _build_messages(self, images: list[Image.Image], prompt: str) -> list[dict]:
        sys_content = self.system_prompt if self.system_prompt is not None else _NO_THINK
        content = [{"type": "image", "image": ""} for _ in images]
        content.append({"type": "text", "text": prompt})
        return [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": content},
        ]

    def _prepare_inputs(self, images: list[Image.Image], prompt: str) -> tuple[dict, int]:
        msgs = self._build_messages(images, prompt)
        text = self._tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        device = next(self._model.parameters()).device
        inputs = self._processor(text=[text], images=images, return_tensors="pt").to(device)
        return inputs, inputs["input_ids"].shape[1]

    def infer(self, image: Image.Image | list[Image.Image], prompt: str) -> str:
        if self._model is None:
            raise RuntimeError("Call load() before infer().")

        imgs = image if isinstance(image, list) else [image]
        inputs, prompt_len = self._prepare_inputs(imgs, prompt)

        max_ctx = getattr(self._model.config, "max_position_embeddings", 131072)
        max_new = min(self.max_new_tokens, max(64, max_ctx - prompt_len - 32))

        with torch.no_grad():
            output_ids = self._model.generate(
                pixel_values=inputs.pixel_values,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new,
                do_sample=False,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        new_ids = output_ids[:, prompt_len:]
        return self._processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def infer_batch_logprobs(
        self,
        batch: list[tuple[list[Image.Image], str]],
        choice_labels: list[list[str]],
    ) -> list[tuple[str, dict]]:
        if self._model is None:
            raise RuntimeError("Call load() before infer().")

        results = []
        for (imgs, prompt), labels in zip(batch, choice_labels):
            inputs, _ = self._prepare_inputs(imgs, prompt)

            with torch.no_grad():
                out = self._model.generate(
                    pixel_values=inputs.pixel_values,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=1,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            first_logits = out.scores[0]  # (1, vocab)
            result = self._logprobs_from_first_token(first_logits, [labels], self._tokenizer)
            results.append(result[0])

        return results
