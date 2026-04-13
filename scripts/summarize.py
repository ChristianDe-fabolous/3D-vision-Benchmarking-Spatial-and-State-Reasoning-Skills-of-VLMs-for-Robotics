"""
Regenerate summary.json from an existing results.jsonl without re-running the model.

Calls save_summary from src/evaluation/results.py — identical to what the pipeline does.

Usage:
    python scripts/summarize.py <results.jsonl or directory containing it>
    python scripts/summarize.py --analyse-categories <path>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src/ importable (same as main.py)
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from evaluation.results import save_summary  # noqa: E402


def load_results(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate summary.json from results.jsonl")
    parser.add_argument(
        "path",
        help="Path to results.jsonl or its parent directory",
    )
    parser.add_argument(
        "--analyse-categories",
        action="store_true",
        default=False,
        help="Include per-answer-category breakdown (same flag as main.py)",
    )
    args = parser.parse_args()

    p = Path(args.path)
    if p.is_dir():
        results_file = p / "results.jsonl"
        output_dir = p
    else:
        results_file = p
        output_dir = p.parent

    if not results_file.exists():
        print(f"Error: {results_file} not found", file=sys.stderr)
        sys.exit(1)

    results = load_results(results_file)
    print(f"Loaded {len(results)} results from {results_file}")

    save_summary(output_dir, results, analyse_categories=args.analyse_categories)


if __name__ == "__main__":
    main()
