"""
Persistent JSONL cache for model responses.

All responses from all runs are appended to a single file:
  outputs/cache.jsonl

Each line is a full result entry (same schema as pipeline.py's `entry` dict)
plus a `model_id` field. The cache is keyed by (entry_id, model_id) so
different models never collide.

On resume: already-processed (entry_id, model_id) pairs are skipped,
saving both inference time and API cost.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Set, Tuple


class ResponseCache:
    def __init__(self, cache_path: Path):
        self.path = cache_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._processed: Set[Tuple[str, str]] = self._load_keys()

    def _load_keys(self) -> Set[Tuple[str, str]]:
        if not self.path.exists():
            return set()
        keys = set()
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    keys.add((str(entry["entry_id"]), str(entry["model_id"])))
                except (json.JSONDecodeError, KeyError):
                    continue
        return keys

    def contains(self, entry_id: str, model_id: str) -> bool:
        return (str(entry_id), str(model_id)) in self._processed

    def write(self, entry: dict) -> None:
        """Append entry to cache and mark as processed."""
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self._processed.add((str(entry["entry_id"]), str(entry["model_id"])))
