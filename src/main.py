"""
Entry point.

Examples:
    python main.py --task failure_mode --model qwen-3b --limit 10
    python main.py --task multiview --model qwen-7b --split train --limit 50
    python main.py --task failure_mode --model qwen-3b --local-data /path/to/data
    python main.py --task failure_mode --model qwen-3b --prompt default
    python main.py --task action_phase --model qwen-7b \
        --action-phase-data data/action_phase_dataset_capped.jsonl
    python main.py --task action_phase --model qwen-7b \
        --action-phase-data data/action_phase_dataset_capped.jsonl \
        --action-phase-type action_phase_id
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    LOG_DIR,
    MODEL_QWEN_3B,
    MODEL_QWEN_7B,
    MODEL_QWEN_7B_INT8,
    MODEL_QWEN3_2B,
    MODEL_QWEN_32B_INT8,
    MODEL_GEMMA_4B,
    MODEL_GEMMA_4B_INT8,
    MODEL_GEMMA_12B_INT8,
    MODEL_PHI35_VISION,
    MODEL_PHI35_VISION_INT8,
    MODEL_PHI4_VISION,
    MODEL_NVLM_12B,
    MODEL_NVLM_12B_INT8,
    OUTPUT_DIR,
    PROMPT_DEFAULT,
    PROMPT_PAPER,
    PROMPT_PAPER_COT,
    PROMPT_TEST,
    TASK_FAILURE_MODE,
    TASK_MULTIVIEW,
    TASK_ACTION_PHASE,
)
from models.qwen import QwenVLM
from models.gemma import GemmaVLM
from models.phi import PhiVLM
from models.nvlm import NemotronVLM
from tasks.action_phase import ActionPhaseTask
from tasks.failure_mode import FailureModeTask
from tasks.multiview import MultiviewTask
import pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="VLM Robotics Benchmark")
    parser.add_argument(
        "--task",
        default=TASK_ACTION_PHASE,
        choices=[TASK_FAILURE_MODE, TASK_MULTIVIEW, TASK_ACTION_PHASE],
    )
    parser.add_argument(
        "--action-phase-data",
        type=str,
        default="data/action_phase_dataset.jsonl",
        help="Path to action_phase dataset JSONL (used when --task action_phase)",
    )
    parser.add_argument(
        "--action-phase-type",
        type=str,
        default=None,
        choices=["action_phase_id", "progress", "next_action", "phase_success", "task_success"],
        help="Filter to one question type within the action_phase task",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="Root directory for resolving relative image paths in action_phase dataset",
    )
    parser.add_argument(
        "--model",
        default=MODEL_QWEN_3B,
        choices=[
            # Qwen
            MODEL_QWEN_3B, MODEL_QWEN_7B, MODEL_QWEN_7B_INT8, MODEL_QWEN3_2B, MODEL_QWEN_32B_INT8,
            # Gemma
            MODEL_GEMMA_4B, MODEL_GEMMA_4B_INT8, MODEL_GEMMA_12B_INT8,
            # Phi
            MODEL_PHI35_VISION, MODEL_PHI35_VISION_INT8, MODEL_PHI4_VISION,
            # NVIDIA Nemotron VL
            MODEL_NVLM_12B, MODEL_NVLM_12B_INT8,
        ],
    )
    parser.add_argument(
        "--prompt",
        default=PROMPT_DEFAULT,
        choices=[PROMPT_DEFAULT, PROMPT_PAPER, PROMPT_PAPER_COT, PROMPT_TEST],
        help="Prompt variant to use (default: 'default')",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N matching samples (useful for testing)",
    )
    parser.add_argument(
        "--local-data",
        type=str,
        default=None,
        help="Path to a local dataset directory (skips HuggingFace download)",
    )
    parser.add_argument(
        "--analyse-categories",
        action="store_true",
        default=False,
        help="Include per-answer-category accuracy breakdown in summary.json",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for authenticated requests (higher rate limits)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Fixed run ID to use (overrides auto-generated timestamp ID). Required for --resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip samples already present in results.jsonl (use with --run-id to continue a previous run).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of samples per inference batch (default: 1).",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        default=False,
        help="Ask the model to describe what it sees before answering (smoke test / qualitative analysis).",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        default=False,
        help="Ask the model to reason step by step before giving its final answer letter.",
    )
    parser.add_argument(
        "--logprobs",
        action="store_true",
        default=False,
        help="Score answer choices by logprob instead of generation. Faster; stores per-choice probs in results.",
    )
    parser.add_argument(
        "--test-pipeline",
        action="store_true",
        default=False,
        help="Enable describe prompt and print full model response to stdout for inspection.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Smoke test: run 5 samples with describe + verbose output. Shorthand for --limit 5 --describe --test-pipeline.",
    )
    return parser.parse_args()


def build_task(args):
    if args.task == TASK_ACTION_PHASE:
        return ActionPhaseTask(
            dataset_path=args.action_phase_data,
            question_type=args.action_phase_type,
            limit=args.limit,
            prompt_id=args.prompt,
            image_root=args.image_root,
            describe=args.describe or args.test_pipeline,
            cot=args.cot,
        )
    kwargs = dict(split=args.split, limit=args.limit, local_path=args.local_data, prompt_id=args.prompt)
    if args.task == TASK_FAILURE_MODE:
        return FailureModeTask(**kwargs)
    if args.task == TASK_MULTIVIEW:
        return MultiviewTask(**kwargs)
    raise ValueError(f"Unknown task: {args.task}")


def build_model(args):
    if args.model in (MODEL_QWEN_3B, MODEL_QWEN_7B, MODEL_QWEN_7B_INT8, MODEL_QWEN3_2B, MODEL_QWEN_32B_INT8):
        return QwenVLM(model_key=args.model)
    if args.model in (MODEL_GEMMA_4B, MODEL_GEMMA_4B_INT8, MODEL_GEMMA_12B_INT8):
        return GemmaVLM(model_key=args.model)
    if args.model in (MODEL_PHI35_VISION, MODEL_PHI35_VISION_INT8, MODEL_PHI4_VISION):
        return PhiVLM(model_key=args.model)
    if args.model in (MODEL_NVLM_12B, MODEL_NVLM_12B_INT8):
        return NemotronVLM(model_key=args.model)
    raise ValueError(f"Unknown model: {args.model}")


SMOKE_SYSTEM_PROMPT = (
    "You are a robot vision assistant. "
    "Before answering, describe in 1-2 sentences what you observe in the image. "
    "Then give your answer as a single letter."
)


def main():
    args = parse_args()

    # --smoke is a shorthand for --limit 5 --describe --test-pipeline
    if args.smoke:
        args.limit = args.limit or 5
        args.describe = True
        args.test_pipeline = True

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    elif not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN not set — unauthenticated HF requests may be rate-limited or fail. Pass --hf-token or set HF_TOKEN env var.", flush=True)

    run_id = args.run_id or f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.task}_{args.model}_{args.prompt}"

    config = {
        "run_id": run_id,
        "task": args.task,
        "model": args.model,
        "prompt": args.prompt,
        "cot": args.cot,
        "describe": args.describe,
        "logprobs": args.logprobs,
        "batch_size": args.batch_size,
        "split": args.split,
        "limit": args.limit,
        "local_data": args.local_data,
    }

    if args.logprobs and (args.cot or args.prompt == PROMPT_PAPER_COT):
        print("Warning: --logprobs incompatible with CoT — first generated token is reasoning, not answer. Scores will be meaningless.", flush=True)

    task = build_task(args)
    model = build_model(args)
    model.load()
    if args.test_pipeline or args.smoke:
        model.system_prompt = SMOKE_SYSTEM_PROMPT

    pipeline.run(
        task=task,
        model=model,
        model_id=args.model,
        prompt_id=args.prompt,
        run_id=run_id,
        output_dir=OUTPUT_DIR / run_id,
        log_dir=LOG_DIR,
        config=config,
        analyse_categories=args.analyse_categories,
        resume=args.resume,
        batch_size=args.batch_size,
        verbose_response=args.test_pipeline,
        logprobs=args.logprobs,
    )


if __name__ == "__main__":
    main()
