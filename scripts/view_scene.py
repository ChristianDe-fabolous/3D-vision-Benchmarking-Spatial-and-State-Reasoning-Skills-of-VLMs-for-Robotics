#!/usr/bin/env python3
"""
Browse a scene JSONL file entry by entry, showing the image and question.

Usage:
  python scripts/view_scene.py scenes/scene_14346.jsonl
  python scripts/view_scene.py scenes/scene_14346.jsonl --start 5

Controls:
  Enter / Right / n    next entry
  Left / p             previous entry
  q                    quit
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


def load_entries(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


def show(entry: dict, idx: int, total: int, ax_img, ax_txt, fig):
    img_path = entry.get("image_path")
    ax_img.clear()
    if img_path and Path(img_path).exists():
        img = Image.open(io.BytesIO(Path(img_path).read_bytes())).convert("RGB")
        ax_img.imshow(img)
    else:
        ax_img.text(0.5, 0.5, "no image", ha="center", va="center",
                    transform=ax_img.transAxes, color="gray")
    ax_img.axis("off")

    ax_txt.clear()
    ax_txt.axis("off")

    letters = "ABCDEFGHIJ"
    choices_str = "\n".join(
        f"  {'→' if i == entry['correct_answer'] else ' '} {letters[i]}: {c}"
        for i, c in enumerate(entry["choices"])
    )
    parts = entry.get("question_parts", {})
    parts_str = "  " + ", ".join(f"{k}={v}" for k, v in parts.items()) if parts else ""

    text = (
        f"[{idx + 1}/{total}]  scene {entry.get('scene_id', '?')}  —  {entry['task']}\n"
        f"{entry.get('question_group', '')}  {', '.join(entry.get('question_types', []))}\n"
        f"{parts_str}\n\n"
        f"{entry['question']}\n\n"
        f"{choices_str}\n\n"
        f"id: {entry.get('tile_id') or entry.get('id', '?')}"
    )
    ax_txt.text(0.02, 0.98, text, transform=ax_txt.transAxes,
                va="top", ha="left", fontsize=10, fontfamily="monospace",
                wrap=True, color="white",
                bbox=dict(facecolor="#1e1e2e", edgecolor="none", pad=8))
    fig.canvas.draw()


def main():
    p = argparse.ArgumentParser(description="Browse a scene JSONL file with images")
    p.add_argument("file", help="scene JSONL file (e.g. scenes/scene_14346.jsonl)")
    p.add_argument("--start", type=int, default=0, metavar="N",
                   help="start at entry N (0-indexed)")
    args = p.parse_args()

    entries = load_entries(Path(args.file))
    if not entries:
        print("No entries found.")
        sys.exit(1)

    idx = [args.start]

    fig, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("#1e1e2e")
    ax_img.set_facecolor("#11111b")
    ax_txt.set_facecolor("#1e1e2e")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03, wspace=0.03)

    show(entries[idx[0]], idx[0], len(entries), ax_img, ax_txt, fig)

    def on_key(event):
        if event.key in ("right", "enter", "n", " "):
            idx[0] = min(idx[0] + 1, len(entries) - 1)
        elif event.key in ("left", "p"):
            idx[0] = max(idx[0] - 1, 0)
        elif event.key == "q":
            plt.close(fig)
            return
        show(entries[idx[0]], idx[0], len(entries), ax_img, ax_txt, fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
