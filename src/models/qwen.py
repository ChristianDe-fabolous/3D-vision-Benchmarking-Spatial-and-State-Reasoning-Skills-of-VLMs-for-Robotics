"""
Qwen3-VL model wrapper (Qwen/Qwen3-VL-*).

Install:
    pip install transformers torch accelerate qwen-vl-utils
"""

from __future__ import annotations

import contextlib
import logging
import platform

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from config import MODEL_QWEN3_4B, QWEN_MAX_NEW_TOKENS, QWEN_MODEL_IDS
from models.base import BaseVLM

logger = logging.getLogger("vlm_bench")


class QwenVLM(BaseVLM):
    def __init__(
        self,
        model_key: str = MODEL_QWEN3_4B,
        max_new_tokens: int = QWEN_MAX_NEW_TOKENS,
        lora_adapter: str | None = None,
    ):
        self.model_id = QWEN_MODEL_IDS[model_key]
        self.max_new_tokens = max_new_tokens
        self.lora_adapter = lora_adapter
        self.system_prompt: str | None = None
        self._model = None
        self._processor = None

    def load(self) -> None:
        if platform.machine() == "aarch64":
            torch.backends.cudnn.enabled = False
        print(f"[qwen] Loading {self.model_id} ...", flush=True)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"[qwen] dtype={dtype}, calling from_pretrained ...", flush=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
        )
        if self.lora_adapter:
            print(f"[qwen] Applying LoRA adapter from {self.lora_adapter} ...", flush=True)
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.lora_adapter)
            self._model = self._model.merge_and_unload()
        print("[qwen] from_pretrained done, calling .eval() ...", flush=True)
        self._model.eval()
        print("[qwen] loading processor ...", flush=True)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._processor.tokenizer.padding_side = "left"
        print("[qwen] Model loaded.", flush=True)
        logger.info("Model loaded.")

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

    def _infer_single(self, imgs: list[Image.Image], prompt: str) -> str:
        from qwen_vl_utils import process_vision_info
        msgs = self._build_messages(imgs, prompt)
        text = self._processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(msgs)
        inputs = self._processor(
            text=[text],
            images=image_inputs or None,
            return_tensors="pt",
        ).to(next(self._model.parameters()).device)
        _sdp = (
            torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH])
            if platform.machine() == "aarch64"
            else contextlib.nullcontext()
        )
        with torch.no_grad(), _sdp:
            output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self._decode_outputs(output_ids, inputs.input_ids.shape[1], self._processor)[0]

    def infer_batch(self, batch: list[tuple[list[Image.Image], str]]) -> list[str]:
        if self._model is None or self._processor is None:
            raise RuntimeError("Call load() before infer().")
        groups: dict[int, list[int]] = {}
        for i, (imgs, _) in enumerate(batch):
            groups.setdefault(len(imgs), []).append(i)
        print(f"[Qwen] batch={len(batch)} → {len(groups)} group(s): { {k: len(v) for k, v in groups.items()} }", flush=True)
        results: list[str] = [""] * len(batch)
        for indices in groups.values():
            sub_batch = [batch[i] for i in indices]
            for i, r in zip(indices, self._infer_batch_grouped(sub_batch)):
                results[i] = r
        return results

    def _infer_batch_grouped(self, batch: list[tuple[list[Image.Image], str]]) -> list[str]:
        """True batched inference — only safe when all samples have equal image counts."""
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

        _sdp = (
            torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH])
            if platform.machine() == "aarch64"
            else contextlib.nullcontext()
        )
        with torch.no_grad(), _sdp:
            output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        return self._decode_outputs(output_ids, inputs.input_ids.shape[1], self._processor)

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

        return self._logprobs_from_first_token(out.scores[0], choice_labels, self._processor.tokenizer)
