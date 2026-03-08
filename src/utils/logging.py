import json
import logging
from pathlib import Path


def setup_logger(run_id: str, log_dir: Path) -> logging.Logger:
    """
    Returns a logger that writes human-readable output to console
    and a debug-level .log file under log_dir.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"vlm_bench.{run_id}")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s", "%H:%M:%S")
        )

        file_handler = logging.FileHandler(log_dir / f"{run_id}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
        )

        logger.addHandler(console)
        logger.addHandler(file_handler)

    return logger


class SampleLogger:
    """
    Writes one JSON line per evaluated sample to results.jsonl.
    Safe to use mid-run — each write is flushed immediately so a crash
    does not lose previously logged entries.
    """

    def __init__(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / "results.jsonl"

    def log(self, entry: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
