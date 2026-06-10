#!/usr/bin/env python3
"""LoRA fine-tune Qwen/Qwen3-VL-4B-Instruct on action_phase_dataset.

Data selection: scene-level random split — all questions from a scene
go to the same split, preventing data leakage between train and val.

Usage:
    python scripts/train_lora.py
    python scripts/train_lora.py --n-train-scenes 8 --seed 42 --output-dir outputs/lora-run1
"""

import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scene_split(
    jsonl_path: str,
    n_train_scenes: int,
    n_val_scenes: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], list[str], list[str], list[str]]:
    """Split scenes into train / val (small, for per-epoch eval loss) / test (held-out benchmark)."""
    data = [json.loads(line) for line in open(jsonl_path)]

    all_scenes = sorted({d["scene_id"] for d in data})
    rng = random.Random(seed)
    rng.shuffle(all_scenes)

    train_scenes = set(all_scenes[:n_train_scenes])
    val_scenes = set(all_scenes[n_train_scenes : n_train_scenes + n_val_scenes])
    test_scenes = set(all_scenes[n_train_scenes + n_val_scenes :])

    train_data = [d for d in data if d["scene_id"] in train_scenes]
    val_data = [d for d in data if d["scene_id"] in val_scenes]
    test_data = [d for d in data if d["scene_id"] in test_scenes]

    return train_data, val_data, test_data, sorted(train_scenes), sorted(val_scenes), sorted(test_scenes)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ActionPhaseDataset(Dataset):
    def __init__(self, samples: list[dict], data_root: Path):
        self.samples = samples
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        image = Image.open(self.data_root / s["images"][0]).convert("RGB")
        choices_str = "\n".join(s["choices"])
        user_text = (
            f"{s['question']}\n\n{choices_str}\n\n"
            "Reply with the answer letter only (e.g. A)."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": s["answer"]}],
            },
        ]
        return {"messages": messages, "meta": s}


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

def make_collator(processor: AutoProcessor, max_length: int):
    response_ids = processor.tokenizer.encode(
        RESPONSE_TEMPLATE, add_special_tokens=False
    )
    resp_len = len(response_ids)

    def _find_response_start(input_ids: list[int]) -> int:
        for i in range(len(input_ids) - resp_len + 1):
            if input_ids[i : i + resp_len] == response_ids:
                return i + resp_len
        return 0

    def collate(examples: list[dict]) -> dict:
        from qwen_vl_utils import process_vision_info

        texts = []
        images = []
        for ex in examples:
            texts.append(
                processor.apply_chat_template(
                    ex["messages"], tokenize=False, add_generation_prompt=False
                )
            )
            image_inputs, _ = process_vision_info(ex["messages"])
            images.append(image_inputs)

        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for i, ids in enumerate(batch["input_ids"].tolist()):
            resp_start = _find_response_start(ids)
            labels[i, :resp_start] = -100

        batch["labels"] = labels
        return batch

    return collate


# ---------------------------------------------------------------------------
# Benchmark eval on val scenes
# ---------------------------------------------------------------------------

def run_eval(model, processor, val_data: list[dict], data_root: Path, output_dir: Path) -> float:
    from qwen_vl_utils import process_vision_info

    model.eval()
    device = next(model.parameters()).device
    results = []

    for s in val_data:
        image = Image.open(data_root / s["images"][0]).convert("RGB")
        choices_str = "\n".join(s["choices"])
        user_text = (
            f"{s['question']}\n\n{choices_str}\n\n"
            "Reply with the answer letter only (e.g. A)."
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=[image_inputs], return_tensors="pt").to(device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
            )

        response_raw = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        predicted = re.search(r"[A-D]", response_raw)
        predicted_label = predicted.group(0) if predicted else ""
        correct = predicted_label == s["answer"]

        results.append({
            "id": s["id"],
            "scene_id": s["scene_id"],
            "question_type": s["question_type"],
            "ground_truth": s["answer"],
            "response_raw": response_raw,
            "predicted": predicted_label,
            "correct": correct,
        })

    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0.0

    # Save results
    results_path = output_dir / "eval_results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Per question_type breakdown
    by_type: dict[str, list[bool]] = {}
    for r in results:
        by_type.setdefault(r["question_type"], []).append(r["correct"])

    print(f"\n=== Eval on {len(val_data)} val samples ===")
    print(f"Overall accuracy: {accuracy:.3f}")
    for qt, corrects in sorted(by_type.items()):
        print(f"  {qt}: {sum(corrects)}/{len(corrects)} = {sum(corrects)/len(corrects):.3f}")
    print(f"Results saved to {results_path}")

    return accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/action_phase_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train-scenes", type=int, default=20)
    parser.add_argument("--n-val-scenes", type=int, default=7,
                        help="held-out set for per-epoch eval loss")
    parser.add_argument("--output-dir", default="outputs/lora-qwen3-vl-4b")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    data_root = Path(args.data).parent.parent  # project root
    output_dir = Path(args.output_dir)

    # ---- data ----
    train_data, val_data, test_data, train_scenes, val_scenes, test_scenes = load_scene_split(
        args.data, args.n_train_scenes, args.n_val_scenes, args.seed
    )
    print(f"Train scenes ({len(train_scenes)}): {train_scenes}")
    print(f"Val   scenes ({len(val_scenes)}):   {val_scenes}")
    print(f"Test  scenes ({len(test_scenes)}):  {test_scenes}")
    print(f"Train samples: {len(train_data)}  |  Val samples: {len(val_data)}  |  Test samples: {len(test_data)}")

    # ---- model ----
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    # ---- LoRA ----
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj)",
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---- datasets ----
    train_ds = ActionPhaseDataset(train_data, data_root)
    val_ds = ActionPhaseDataset(val_data, data_root)
    collator = make_collator(processor, args.max_length)

    # ---- training ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\nAdapter saved to {args.output_dir}")

    # ---- benchmark eval on held-out test scenes ----
    # Merge LoRA into base weights so inference runs at full speed
    merged = trainer.model.merge_and_unload()
    run_eval(merged, processor, test_data, data_root, output_dir)


if __name__ == "__main__":
    main()
