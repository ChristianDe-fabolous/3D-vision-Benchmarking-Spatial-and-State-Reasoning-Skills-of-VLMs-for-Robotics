"""
LoRA fine-tuning for Qwen2.5-VL-7B-Instruct (int8) on Robo2VLM-1 VQA.

Usage (from src/):
    # Robo2VLM-1 failure_mode / multiview tasks
    python train_lora.py --limit 6000 --epochs 2 --output ../outputs/lora_qwen7b

    # Action-phase progress dataset (local, no HF download)
    python train_lora.py --source progress \
        --dataset_path ../data/action_phase_dataset.jsonl \
        --epochs 5 --output ../outputs/lora_progress

Extra deps (beyond existing project stack):
    pip install peft bitsandbytes
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

sys.path.insert(0, os.path.dirname(__file__))

from config import QWEN_MODEL_IDS, MODEL_QWEN_7B_INT8, TASK_FAILURE_MODE, TASK_MULTIVIEW
from data.dataset import Sample, load_dataset
from data.action_phase_progress import ProgressDataset
import data.action_phase_progress as progress_data
import data.failure_mode as fm_data
import data.multiview as mv_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_lora")

CHOICE_LABELS = ["A", "B", "C", "D", "E"]

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _build_prompt(sample: Sample, prompt_id: str) -> str:
    if sample.task == "progress":
        return progress_data.build_prompt(sample)
    if sample.task == TASK_FAILURE_MODE:
        return fm_data.build_prompt(sample, prompt_id)
    return mv_data.build_prompt(sample, prompt_id)


def _answer_letter(sample: Sample) -> str:
    return CHOICE_LABELS[sample.correct_answer]


@dataclass
class ProcessedSample:
    input_ids: torch.Tensor        # (seq_len,)
    attention_mask: torch.Tensor   # (seq_len,)
    labels: torch.Tensor           # (seq_len,) — -100 everywhere except answer
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None


class VQADataset(Dataset):
    def __init__(
        self,
        processor: AutoProcessor,
        samples: list[ProcessedSample],
    ):
        self.processor = processor
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ProcessedSample:
        return self.samples[idx]


def _preprocess_sample(
    sample: Sample,
    processor: AutoProcessor,
    prompt_id: str,
) -> ProcessedSample | None:
    from qwen_vl_utils import process_vision_info

    prompt_text = _build_prompt(sample, prompt_id)
    answer = _answer_letter(sample)

    images = sample.all_images

    # Full conversation including answer
    full_messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img} for img in images]
            + [{"type": "text", "text": prompt_text}],
        },
        {"role": "assistant", "content": answer},
    ]

    # Prompt-only conversation (to find where answer starts)
    prompt_messages = full_messages[:1]

    try:
        full_text = processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text_chat = processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        full_imgs, _ = process_vision_info(full_messages)
        prompt_imgs, _ = process_vision_info(prompt_messages)

        full_enc = processor(
            text=full_text,
            images=full_imgs or None,
            padding=False,
            return_tensors="pt",
        )
        prompt_enc = processor(
            text=prompt_text_chat,
            images=prompt_imgs or None,
            padding=False,
            return_tensors="pt",
        )
    except Exception as e:
        logger.warning(f"Skip sample {sample.id}: {e}")
        return None

    input_ids = full_enc.input_ids[0]
    attention_mask = full_enc.attention_mask[0]
    prompt_len = prompt_enc.input_ids.shape[1]

    labels = input_ids.clone()
    labels[:prompt_len] = -100  # mask prompt — only compute loss on answer

    pixel_values = full_enc.get("pixel_values")
    image_grid_thw = full_enc.get("image_grid_thw")

    if pixel_values is not None:
        pixel_values = pixel_values.squeeze(0) if pixel_values.ndim == 4 else pixel_values

    return ProcessedSample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )


def _preprocess_samples(
    raw: list[Sample],
    processor: AutoProcessor,
    prompt_id: str,
    log_every: int = 50,
) -> list[ProcessedSample]:
    processed = []
    for i, s in enumerate(raw):
        p = _preprocess_sample(s, processor, prompt_id)
        if p is not None:
            processed.append(p)
        if (i + 1) % log_every == 0:
            logger.info(f"  {i+1}/{len(raw)} done, kept {len(processed)}")
    return processed


def build_dataset_robo2vlm(
    processor: AutoProcessor,
    limit: int,
    prompt_id: str,
    local_path: str | None,
    split: str,
) -> VQADataset:
    logger.info(f"Loading up to {limit} samples from Robo2VLM-1 {split} split ...")
    raw = list(load_dataset(split=split, limit=limit, local_path=local_path))
    logger.info(f"Loaded {len(raw)} raw samples, preprocessing ...")
    processed = _preprocess_samples(raw, processor, prompt_id)
    logger.info(f"Dataset ready: {len(processed)} samples")
    return VQADataset(processor, processed)


def build_dataset_progress(
    processor: AutoProcessor,
    dataset_path: str,
    prompt_id: str,
    split: str,
) -> tuple[VQADataset, VQADataset]:
    """
    Returns (train_dataset, test_dataset) from the progress subset of
    action_phase_dataset.jsonl.

    Train: step_a and step_b both in {0, 1, 2} (first three images).
    Test : at least one step > 2 (remaining images).
    Scenes with ≤3 total steps appear in train only.
    """
    ds = ProgressDataset(dataset_path)
    logger.info(ds.summary())

    train_raw = ds.train()
    test_raw  = ds.test()

    logger.info(f"Preprocessing {len(train_raw)} train samples ...")
    train_proc = _preprocess_samples(train_raw, processor, prompt_id)

    logger.info(f"Preprocessing {len(test_raw)} test samples ...")
    test_proc  = _preprocess_samples(test_raw,  processor, prompt_id)

    logger.info(f"Train ready: {len(train_proc)}  Test ready: {len(test_proc)}")
    return VQADataset(processor, train_proc), VQADataset(processor, test_proc)


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class VQACollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[ProcessedSample]) -> dict[str, Any]:
        max_len = max(s.input_ids.shape[0] for s in batch)

        input_ids_list, attn_list, labels_list = [], [], []
        for s in batch:
            pad = max_len - s.input_ids.shape[0]
            input_ids_list.append(
                torch.cat([s.input_ids, torch.full((pad,), self.pad_token_id)])
            )
            attn_list.append(
                torch.cat([s.attention_mask, torch.zeros(pad, dtype=torch.long)])
            )
            labels_list.append(
                torch.cat([s.labels, torch.full((pad,), -100)])
            )

        result = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attn_list),
            "labels": torch.stack(labels_list),
        }

        # pixel_values: concatenate along batch dim (patches)
        pv_list = [s.pixel_values for s in batch if s.pixel_values is not None]
        if pv_list:
            result["pixel_values"] = torch.cat(pv_list, dim=0)

        gt_list = [s.image_grid_thw for s in batch if s.image_grid_thw is not None]
        if gt_list:
            result["image_grid_thw"] = torch.cat(gt_list, dim=0)

        return result


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model_and_processor(model_id: str) -> tuple:
    logger.info(f"Loading {model_id} in int8 ...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tune Qwen2.5-VL-7B int8")
    p.add_argument("--source", default="robo2vlm", choices=["robo2vlm", "progress"],
                   help="Data source: 'robo2vlm' (HF) or 'progress' (local action-phase dataset)")
    p.add_argument("--model", default=MODEL_QWEN_7B_INT8,
                   help="Model key from config (default: qwen-7b-int8)")
    # robo2vlm-specific
    p.add_argument("--limit", type=int, default=6000,
                   help="[robo2vlm] Max training samples to load")
    p.add_argument("--split", default="train",
                   help="[robo2vlm] Dataset split (default: train)")
    p.add_argument("--local_path", default=None,
                   help="[robo2vlm] Local dataset directory (skip HF download)")
    # progress-specific
    p.add_argument("--dataset_path", default="../data/action_phase_dataset.jsonl",
                   help="[progress] Path to action_phase_dataset.jsonl")
    # shared
    p.add_argument("--prompt_id", default="paper",
                   help="Prompt format: paper | default | paper_cot")
    p.add_argument("--output", default="../outputs/lora_qwen7b",
                   help="Output directory for checkpoints and final adapter")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Per-device training batch size")
    p.add_argument("--grad_accum", type=int, default=16,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--lr", type=float, default=2e-4)
    return p.parse_args()


def main():
    args = parse_args()

    model_id = QWEN_MODEL_IDS[args.model]
    model, processor = load_model_and_processor(model_id)

    collator = VQACollator(pad_token_id=processor.tokenizer.pad_token_id)

    if args.source == "progress":
        train_dataset, test_dataset = build_dataset_progress(
            processor=processor,
            dataset_path=args.dataset_path,
            prompt_id=args.prompt_id,
            split="train",
        )
        eval_dataset = test_dataset if len(test_dataset) > 0 else None
    else:
        train_dataset = build_dataset_robo2vlm(
            processor=processor,
            limit=args.limit,
            prompt_id=args.prompt_id,
            local_path=args.local_path,
            split=args.split,
        )
        eval_dataset = None

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    logger.info("Starting training ...")
    trainer.train()

    logger.info(f"Saving LoRA adapter to {args.output}/lora_adapter ...")
    model.save_pretrained(f"{args.output}/lora_adapter")
    processor.save_pretrained(f"{args.output}/lora_adapter")
    logger.info("Done.")


if __name__ == "__main__":
    main()
