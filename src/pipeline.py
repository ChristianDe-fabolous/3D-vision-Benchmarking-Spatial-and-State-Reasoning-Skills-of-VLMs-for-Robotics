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
    batch_size: int = 1,
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
    pending = []  # buffer of (sample, prompt) up to batch_size

    def _flush(buf):
        nonlocal i
        batch_input = [(s.all_images, p) for s, p in buf]
        try:
            responses = model.infer_batch(batch_input)
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            responses = [""] * len(buf)

        for (sample, prompt), response in zip(buf, responses):
            i += 1
            correct = task.evaluate(response, sample)
            predicted_idx = task.parse_response(response, sample)
            predicted_label = (
                sample.choices[predicted_idx] if predicted_idx is not None else None
            )
            meta = {k: v for k, v in sample.metadata.items() if k != "raw_row"}
            entry = {
                "run_id": run_id,
                "model_id": model_id,
                "prompt_id": prompt_id,
                "entry_id": sample.id,
                "dataset_row": sample.metadata.get("raw_row", 0),
                "task": sample.task,
                "question": sample.question,
                "choices": sample.choices,
                "ground_truth_index": sample.correct_answer,
                "ground_truth_label": sample.correct_choice,
                "response_raw": response,
                "predicted_index": predicted_idx,
                "predicted_label": predicted_label,
                "correct": correct,
                **meta,
            }
            sample_logger.log(entry)
            results.append(entry)
            status = "✓" if correct else "✗"
            logger.info(
                f"[{i}] {status}  response='{response}'  "
                f"gt='{sample.correct_choice}'"
            )

    for sample in task.get_samples(skip=skip_rows):
        if sample.id in completed_ids:
            logger.debug(f"skip (already done)  id={sample.id}")
            continue

        logger.debug(f"id={sample.id}  q={sample.question[:60]}...")
        pending.append((sample, task.build_prompt(sample)))

        if len(pending) >= batch_size:
            _flush(pending)
            pending = []

    if pending:
        _flush(pending)

    save_summary(output_dir, results, analyse_categories=analyse_categories)
    logger.info(f"Done. Results saved to {output_dir}")
    return results
