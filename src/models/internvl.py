"""
OpenGVLab InternVL3 model wrapper (OpenGVLab/InternVL3_5-*).

Uses trust_remote_code=True and the model's .chat() API.

Install:
    pip install transformers torch accelerate torchvision timm einops
"""

from __future__ import annotations

import logging

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from config import INTERNVL_MAX_NEW_TOKENS, INTERNVL_MODEL_IDS, MODEL_INTERNVL3_14B
from models.base import BaseVLM

logger = logging.getLogger("vlm_bench")

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_TILE_SIZE     = 448

_TRANSFORM = T.Compose([
    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    T.Resize((_TILE_SIZE, _TILE_SIZE), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
) -> tuple[int, int]:
    best_diff = float("inf")
    best = (1, 1)
    area = width * height
    for ratio in target_ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff or (diff == best_diff and area > 0.5 * _TILE_SIZE ** 2 * ratio[0] * ratio[1]):
            best_diff = diff
            best = ratio
    return best


def _dynamic_preprocess(image: Image.Image, max_num: int = 12) -> list[Image.Image]:
    w, h = image.size
    aspect_ratio = w / h
    target_ratios = sorted(
        {
            (i, j)
            for n in range(1, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if 1 <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    cols, rows = _find_closest_aspect_ratio(aspect_ratio, target_ratios, w, h)
    tw, th = _TILE_SIZE * cols, _TILE_SIZE * rows
    resized = image.resize((tw, th))
    tiles = [
        resized.crop((
            (i % cols) * _TILE_SIZE,
            (i // cols) * _TILE_SIZE,
            ((i % cols) + 1) * _TILE_SIZE,
            ((i // cols) + 1) * _TILE_SIZE,
        ))
        for i in range(cols * rows)
    ]
    if len(tiles) != 1:
        tiles.append(image.resize((_TILE_SIZE, _TILE_SIZE)))
    return tiles


def _preprocess_image(image: Image.Image, max_num: int = 12) -> torch.Tensor:
    return torch.stack([_TRANSFORM(t) for t in _dynamic_preprocess(image, max_num=max_num)])


class InternVLM(BaseVLM):
    def __init__(
        self,
        model_key: str = MODEL_INTERNVL3_14B,
        max_new_tokens: int = INTERNVL_MAX_NEW_TOKENS,
    ):
        self.model_id = INTERNVL_MODEL_IDS[model_key]
        self.max_new_tokens = max_new_tokens
        self.system_prompt: str | None = None
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        logger.info(f"Loading {self.model_id} ...")
        # cuDNN 9.20's conv engine selection crashes (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH)
        # on the ViT patch_embedding Conv2d for this GPU arch — fall back to non-cuDNN conv.
        torch.backends.cudnn.enabled = False
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self._model = AutoModel.from_pretrained(
            self.model_id,
            dtype=dtype,
            trust_remote_code=True,
        ).eval().to("cuda")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False
        )
        logger.info("Model loaded.")

    def _pixel_values(self, imgs: list[Image.Image]) -> tuple[torch.Tensor, list[int]]:
        pvs = [_preprocess_image(img) for img in imgs]
        num_patches = [pv.shape[0] for pv in pvs]
        device = next(self._model.parameters()).device
        dtype  = next(self._model.parameters()).dtype
        return torch.cat(pvs, dim=0).to(device=device, dtype=dtype), num_patches

    def _build_question(self, imgs: list[Image.Image], prompt: str) -> str:
        if len(imgs) == 1:
            prefix = "<image>\n"
        else:
            prefix = "\n".join(f"Image-{i+1}: <image>" for i in range(len(imgs))) + "\n"
        return prefix + prompt

    def infer(self, image: Image.Image | list[Image.Image], prompt: str) -> str:
        if self._model is None:
            raise RuntimeError("Call load() before infer().")
        imgs = image if isinstance(image, list) else [image]
        pixel_values, num_patches = self._pixel_values(imgs)
        question = self._build_question(imgs, prompt)
        gen_config = dict(max_new_tokens=self.max_new_tokens, do_sample=False)
        if self.system_prompt is not None:
            self._model.system_message = self.system_prompt
        response = self._model.chat(
            self._tokenizer,
            pixel_values,
            question,
            gen_config,
            num_patches_list=num_patches if len(imgs) > 1 else None,
            history=None,
            return_history=False,
        )
        return response.strip()

    def infer_batch(self, batch: list[tuple[list[Image.Image], str]]) -> list[str]:
        return [self.infer(imgs, prompt) for imgs, prompt in batch]
