"""
Generate scene_ranking.json — every scene ordered by accuracy, hardest to easiest.

Unlike summary.json's scene_analysis (which only flags statistical outliers),
this lists ALL scenes that meet the SCENE_MIN_QUESTIONS threshold.

Usage:
    python scripts/rank_scenes.py <results.jsonl or directory containing it>
    python scripts/rank_scenes.py <path> --top 25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src/ importable (same as main.py)
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from evaluation.results import save_scene_ranking  # noqa: E402


def load_results(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank all scenes by accuracy (hardest to easiest)")
    parser.add_argument(
        "path",
        help="Path to results.jsonl or its parent directory",
    )
    parser.add_argument(
        "--top", type=int, default=15,
        help="Number of hardest/easiest scenes to print (default 15)",
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

    save_scene_ranking(output_dir, results)

    ranking = json.loads((output_dir / "scene_ranking.json").read_text())["ranking"]
    n = args.top

    print(f"\nHardest {n} scenes:")
    for s in ranking[:n]:
        print(f"  {s['accuracy']:>6.1%}  {s['correct']:>3}/{s['n']:<3}  {s['scene_id']}")

    print(f"\nEasiest {n} scenes:")
    for s in ranking[-n:][::-1]:
        print(f"  {s['accuracy']:>6.1%}  {s['correct']:>3}/{s['n']:<3}  {s['scene_id']}")


if __name__ == "__main__":
    main()
