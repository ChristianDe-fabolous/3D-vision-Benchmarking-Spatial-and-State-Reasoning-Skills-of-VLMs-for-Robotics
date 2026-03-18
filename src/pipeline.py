"""
Main evaluation loop.

Streams samples from the dataset one at a time — no need to load
678k images into memory. Each result is flushed to disk immediately.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from config import OUTPUT_DIR
from evaluation.results import save_config, save_summary
from models.base import BaseVLM
from tasks.base import BaseTask
from utils.cache import ResponseCache
from utils.logging import SampleLogger, setup_logger


def run(
    task: BaseTask,
    model: BaseVLM,
    model_id: str,
    prompt_id: str,
    run_id: str,
    output_dir: Path,
    log_dir: Path,
    config: Optional[dict] = None,
    analyse_categories: bool = False,
) -> List[dict]:
    """
    Stream and evaluate all samples from `task` using `model`.

    Args:
        task:       Task instance (FailureModeTask or MultiviewTask).
        model:      Already-loaded VLM instance.
        model_id:   Model identifier string (used for cache key and logging).
        prompt_id:  Prompt variant identifier (used for cache key and logging).
        run_id:     Unique identifier for this run (used for filenames).
        output_dir: Where to write results.jsonl and summary.json.
        log_dir:    Where to write the human-readable .log file.
        config:     Run parameters to persist alongside results.

    Returns:
        List of per-sample result dicts.
    """
    logger = setup_logger(run_id, log_dir)
    sample_logger = SampleLogger(output_dir)
    cache = ResponseCache(OUTPUT_DIR / "cache.jsonl")

    if config:
        save_config(output_dir, config)

    logger.info(f"Run '{run_id}' | task={task.__class__.__name__} | model={model_id} | prompt={prompt_id}")

    results: List[dict] = []
    i = 0

    for sample in task.get_samples():
        i += 1

        if cache.contains(sample.id, model_id, prompt_id):
            logger.debug(f"[{i}] id={sample.id}  SKIPPED (cached)")
            continue

        prompt = task.build_prompt(sample)

        logger.debug(f"[{i}] id={sample.id}  q={sample.question[:60]}...")

        try:
            response = model.infer(sample.image, prompt)
        except Exception as e:
            logger.error(f"Inference failed for {sample.id}: {e}")
            response = ""

        correct = task.evaluate(response, sample)
        predicted_idx = task.parse_response(response, sample)
        predicted_label = (
            sample.choices[predicted_idx] if predicted_idx is not None else None
        )

        entry = {
            "run_id": run_id,
            "model_id": model_id,
            "prompt_id": prompt_id,
            "entry_id": sample.id,
            "task": sample.task,
            "question": sample.question,
            "choices": sample.choices,
            "ground_truth_index": sample.correct_answer,
            "ground_truth_label": sample.correct_choice,
            "response_raw": response,
            "predicted_index": predicted_idx,
            "predicted_label": predicted_label,
            "correct": correct,
            **sample.metadata,
        }

        cache.write(entry)
        sample_logger.log(entry)
        results.append(entry)

        status = "✓" if correct else "✗"
        logger.info(
            f"[{i}] {status}  response='{response}'  "
            f"gt='{sample.correct_choice}'"
        )

    save_summary(output_dir, results, analyse_categories=analyse_categories)
    logger.info(f"Done. Results saved to {output_dir}")
    return results
