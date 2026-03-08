"""
Entry point.

Examples:
    python main.py --task failure_mode --model qwen
    python main.py --task multiview --model qwen --split train --limit 50
    python main.py --task failure_mode --model qwen --local-data /path/to/data
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    LOG_DIR,
    MODEL_QWEN,
    OUTPUT_DIR,
    TASK_FAILURE_MODE,
    TASK_MULTIVIEW,
)
from models.qwen import QwenVLM
from tasks.failure_mode import FailureModeTask
from tasks.multiview import MultiviewTask
import pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="VLM Robotics Benchmark")
    parser.add_argument(
        "--task",
        required=True,
        choices=[TASK_FAILURE_MODE, TASK_MULTIVIEW],
    )
    parser.add_argument(
        "--model",
        default=MODEL_QWEN,
        choices=[MODEL_QWEN],
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
    return parser.parse_args()


def build_task(args):
    kwargs = dict(split=args.split, limit=args.limit, local_path=args.local_data)
    if args.task == TASK_FAILURE_MODE:
        return FailureModeTask(**kwargs)
    if args.task == TASK_MULTIVIEW:
        return MultiviewTask(**kwargs)
    raise ValueError(f"Unknown task: {args.task}")


def build_model(args):
    if args.model == MODEL_QWEN:
        return QwenVLM()
    raise ValueError(f"Unknown model: {args.model}")


def main():
    args = parse_args()
    run_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.task}_{args.model}"

    config = {
        "run_id": run_id,
        "task": args.task,
        "model": args.model,
        "split": args.split,
        "limit": args.limit,
        "local_data": args.local_data,
    }

    task = build_task(args)
    model = build_model(args)
    model.load()

    pipeline.run(
        task=task,
        model=model,
        run_id=run_id,
        output_dir=OUTPUT_DIR / run_id,
        log_dir=LOG_DIR,
        config=config,
    )


if __name__ == "__main__":
    main()
