#!/usr/bin/env python3
"""Quick benchmark: streaming speed without image column."""

import time
from datasets import load_dataset

N = 1000
ds = load_dataset(
    "keplerccc/Robo2VLM-1",
    split="train",
    streaming=True,
    columns=["id", "question", "choices", "correct_answer"],
)

t0 = time.time()
for i, _ in enumerate(ds, 1):
    if i % 200 == 0:
        elapsed = time.time() - t0
        print(f"  {i:>5} rows — {i/elapsed:6.1f} rows/s — {elapsed:.1f}s elapsed", flush=True)
    if i >= N:
        break

elapsed = time.time() - t0
print(f"\nResult: {i} rows in {elapsed:.1f}s = {i/elapsed:.1f} rows/s")
