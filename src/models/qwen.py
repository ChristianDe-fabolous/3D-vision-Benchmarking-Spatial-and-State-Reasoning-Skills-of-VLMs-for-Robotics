"""
Qwen2.5-VL / Qwen3-VL model wrapper.

Install:
    pip install transformers torch accelerate qwen-vl-utils
"""

from __future__ import annotations

import logging

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from config import QWEN_INT8_KEYS, QWEN_MAX_NEW_TOKENS, QWEN_MODEL_IDS, MODEL_QWEN_3B, MODEL_QWEN3_2B
from models.base import BaseVLM

logger = logging.getLogger("vlm_bench")

_QWEN3_KEYS = {MODEL_QWEN3_2B}


class QwenVLM(BaseVLM):
    def __init__(
        self,
        model_key: str = MODEL_QWEN_3B,
        max_new_tokens: int = QWEN_MAX_NEW_TOKENS,
    ):
        self.model_id = QWEN_MODEL_IDS[model_key]
        self._is_qwen3 = model_key in _QWEN3_KEYS
        self._int8 = model_key in QWEN_INT8_KEYS
        self.max_new_tokens = max_new_tokens
        self.system_prompt: str | None = None
        self._model = None
        self._processor = None

    def load(self) -> None:
        logger.info(f"Loading {self.model_id} {'(int8)' if self._int8 else ''} ...")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
        kwargs = dict(device_map="auto")
        if self._int8:
            kwargs["load_in_8bit"] = True
        else:
            kwargs["torch_dtype"] = dtype
        if self._is_qwen3:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, trust_remote_code=True, **kwargs
            )
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id, **kwargs
            )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._processor.tokenizer.padding_side = "left"
        logger.info("Model loaded.")

    def _answer_token_ids(self, labels: list[str]) -> dict[str, int]:
        """Return {label: token_id} for each answer letter, trying plain and space-prefixed forms."""
        tok = self._processor.tokenizer
        result = {}
        for label in labels:
            for form in (label, f" {label}"):
                ids = tok.encode(form, add_special_tokens=False)
                if len(ids) == 1:
                    result[label] = ids[0]
                    break
        return result

    def _build_messages(self, images: list[Image.Image], prompt: str) -> list[dict]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def infer(self, image: Image.Image | list[Image.Image], prompt: str) -> str:
        return self.infer_batch([(image if isinstance(image, list) else [image], prompt)])[0]

    def infer_batch(self, batch: list[tuple[list[Image.Image], str]]) -> list[str]:
        if self._model is None or self._processor is None:
            raise RuntimeError("Call load() before infer().")

        from qwen_vl_utils import process_vision_info

        all_messages = [self._build_messages(imgs, prompt) for imgs, prompt in batch]

        texts = [
            self._processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in all_messages
        ]
        image_inputs = []
        for msgs in all_messages:
            imgs, _ = process_vision_info(msgs)
            image_inputs.extend(imgs or [])

        inputs = self._processor(
            text=texts,
            images=image_inputs if image_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(next(self._model.parameters()).device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

        prompt_len = inputs.input_ids.shape[1]
        return [
            self._processor.decode(out[prompt_len:], skip_special_tokens=True).strip()
            for out in output_ids
        ]

    def infer_batch_logprobs(
        self,
        batch: list[tuple[list[Image.Image], str]],
        choice_labels: list[list[str]],
    ) -> list[tuple[str, dict]]:
        if self._model is None or self._processor is None:
            raise RuntimeError("Call load() before infer().")

        from qwen_vl_utils import process_vision_info

        all_messages = [self._build_messages(imgs, prompt) for imgs, prompt in batch]
        texts = [
            self._processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in all_messages
        ]
        image_inputs = []
        for msgs in all_messages:
            imgs, _ = process_vision_info(msgs)
            image_inputs.extend(imgs or [])

        inputs = self._processor(
            text=texts,
            images=image_inputs if image_inputs else None,
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

        # out.scores[0]: (batch, vocab_size) — logits at first generated token
        first_logits = out.scores[0]  # (batch, vocab)

        results = []
        for i, labels in enumerate(choice_labels):
            token_ids = self._answer_token_ids(labels)
            if not token_ids:
                results.append(("", {}))
                continue
            ids = [token_ids[l] for l in labels if l in token_ids]
            label_keys = [l for l in labels if l in token_ids]
            logits_for_choices = first_logits[i, ids]
            probs = torch.softmax(logits_for_choices, dim=-1).tolist()
            prob_dict = dict(zip(label_keys, probs))
            predicted = max(prob_dict, key=prob_dict.__getitem__)
            results.append((predicted, prob_dict))

        return results
