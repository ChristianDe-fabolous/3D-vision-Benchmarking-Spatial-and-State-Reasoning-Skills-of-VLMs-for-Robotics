"""
Main evaluation loop.

Streams samples from the dataset one at a time — no need to load
678k images into memory. Each result is flushed to disk immediately.
Supports resume: if results.jsonl already exists in output_dir, already-
processed entry_ids are skipped and new results are appended.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Set

from evaluation.results import save_config, save_summary
from models.base import BaseVLM
from tasks.base import BaseTask
from utils.logging import SampleLogger, setup_logger


def _load_completed(output_dir: Path) -> tuple[List[dict], Set[str], int]:
    """
    Load already-processed results from results.jsonl, if it exists.
    Returns (results, entry_ids, max_raw_row_index) where max_raw_row_index
    is the dataset row position to skip to on resume.
    """
    results_path = output_dir / "results.jsonl"
    if not results_path.exists():
        return [], set(), 0
    results = []
    ids = set()
    max_row = 0
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                results.append(entry)
                ids.add(str(entry["entry_id"]))
                max_row = max(max_row, entry.get("dataset_row", 0))
            except (json.JSONDecodeError, KeyError):
                continue
    return results, ids, max_row


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
    resume: bool = False,
) -> List[dict]:
    """
    Stream and evaluate all samples from `task` using `model`.

    Args:
        task:       Task instance (FailureModeTask or MultiviewTask).
        model:      Already-loaded VLM instance.
        model_id:   Model identifier string (used for logging).
        prompt_id:  Prompt variant identifier (used for logging).
        run_id:     Unique identifier for this run (used for filenames).
        output_dir: Where to write results.jsonl and summary.json.
        log_dir:    Where to write the human-readable .log file.
        config:     Run parameters to persist alongside results.
        resume:     If True, skip entry_ids already present in results.jsonl.

    Returns:
        List of per-sample result dicts (including any previously completed).
    """
    logger = setup_logger(run_id, log_dir)
    sample_logger = SampleLogger(output_dir)

    if config:
        save_config(output_dir, config)

    completed_results, completed_ids, skip_rows = _load_completed(output_dir) if resume else ([], set(), 0)
    if completed_ids:
        logger.info(f"Resuming — {len(completed_ids)} already done, fast-forwarding {skip_rows} dataset rows.")

    logger.info(f"Run '{run_id}' | task={task.__class__.__name__} | model={model_id} | prompt={prompt_id}")

    results: List[dict] = list(completed_results)
    i = 0
    row_index = skip_rows  # absolute position in the raw dataset stream

    for sample in task.get_samples(skip=skip_rows):
        i += 1
        row_index += 1

        if sample.id in completed_ids:
            # Safety net: skip anything that slipped through (shouldn't happen)
            logger.debug(f"[{i}] skip (already done)  id={sample.id}")
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
            "dataset_row": row_index,
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
