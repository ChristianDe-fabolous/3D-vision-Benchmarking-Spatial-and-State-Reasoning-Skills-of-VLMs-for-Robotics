#!/usr/bin/env python3
"""
Show 2 sample questions with images, choices, and correct answers
for each category + variant in the action-phase dataset.

Output: multi-page PDF (one page per category/variant).

Usage
-----
  python scripts/show_dataset_samples.py
  python scripts/show_dataset_samples.py --dataset data/action_phase_dataset.jsonl
  python scripts/show_dataset_samples.py --output data/samples.pdf --n 3 --seed 7
"""
from __future__ import annotations

import argparse
import json
import random
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_img_array(rel_path: str, root: Path) -> np.ndarray | None:
    p = Path(rel_path)
    if not p.is_absolute():
        p = root / p
    try:
        return np.array(Image.open(p).convert("RGB"))
    except Exception:
        return None


def wrap(text: str, width: int = 72) -> str:
    lines = []
    for raw in text.split("\n"):
        if len(raw) <= width:
            lines.append(raw)
        else:
            lines.extend(textwrap.wrap(raw, width))
    return "\n".join(lines)


def truncate_block(text: str, max_lines: int = 10) -> str:
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + "\n…"


# ── drawing ───────────────────────────────────────────────────────────────────

PAGE_BG   = "#fafafa"
CARD_BG   = "#ffffff"
Q_BG      = "#f3f4f6"
YES_COLOR = "#1a7f1a"
NO_COLOR  = "#c0392b"
CDN_COLOR = "#7f6000"
DEF_COLOR = "#222222"


def answer_color(text: str) -> str:
    tl = text.lower()
    if "cannot" in tl or "determined" in tl:
        return CDN_COLOR
    if tl in ("yes", "nothing to do", "task already"):
        return YES_COLOR
    return DEF_COLOR


def draw_example(fig: plt.Figure, gs_col, entry: dict, root: Path, label: str) -> None:
    images      = entry.get("images", [])
    question    = entry.get("question", "")
    choices     = entry.get("choices", [])
    ans_letter  = entry.get("answer", "")
    ans_text    = entry.get("answer_text", "")
    entry_id    = entry.get("id", "?")
    special     = entry.get("special_image", "")

    n_imgs = len(images)

    # Column layout: label bar | image area | question+choices
    inner = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs_col,
        height_ratios=[0.04, 0.42, 0.54],
        hspace=0.04,
    )

    # ── label bar ──
    ax_lbl = fig.add_subplot(inner[0])
    ax_lbl.set_facecolor("#dde4f0")
    ax_lbl.axis("off")
    meta = f"  {label}   id={entry_id}"
    if special:
        meta += f"  [{special}]"
    ax_lbl.text(0.01, 0.5, meta, va="center", fontsize=8,
                fontweight="bold", transform=ax_lbl.transAxes, color="#1a2a4a")

    # ── image(s) ──
    if n_imgs >= 2:
        img_gs = gridspec.GridSpecFromSubplotSpec(
            1, n_imgs, subplot_spec=inner[1], wspace=0.03
        )
        for i, rel in enumerate(images[:n_imgs]):
            ax_i = fig.add_subplot(img_gs[i])
            ax_i.axis("off")
            arr = load_img_array(rel, root)
            if arr is not None:
                ax_i.imshow(arr, aspect="auto")
                ax_i.set_title(f"Image {i+1}", fontsize=7, pad=2, color="#444")
            else:
                ax_i.text(0.5, 0.5, "missing", ha="center", va="center",
                          fontsize=8, color="red")
    else:
        ax_i = fig.add_subplot(inner[1])
        ax_i.axis("off")
        if images:
            arr = load_img_array(images[0], root)
            if arr is not None:
                ax_i.imshow(arr, aspect="auto")
            else:
                ax_i.text(0.5, 0.5, "missing", ha="center", va="center",
                          fontsize=8, color="red")
        else:
            ax_i.text(0.5, 0.5, "(no image)", ha="center", va="center",
                      fontsize=8, color="#999")

    # ── question + choices ──
    ax_txt = fig.add_subplot(inner[2])
    ax_txt.set_facecolor(CARD_BG)
    ax_txt.axis("off")

    # question block — truncate long sequence listings
    q_display = truncate_block(wrap(question, 68), max_lines=9)
    ax_txt.text(
        0.02, 0.97, q_display,
        transform=ax_txt.transAxes,
        fontsize=7, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=Q_BG, linewidth=0),
    )

    # choices — estimate vertical start after question block
    q_line_count = q_display.count("\n") + 1
    y_choices    = 0.97 - q_line_count * 0.075 - 0.06
    y_choices    = min(y_choices, 0.52)

    for ch in choices:
        if y_choices < 0.02:
            break
        if len(ch) < 3:
            continue
        letter   = ch[0]
        ch_text  = ch[3:].strip()
        correct  = letter == ans_letter
        color    = answer_color(ans_text) if correct else "#555555"
        weight   = "bold" if correct else "normal"
        prefix   = "✓" if correct else " "
        display  = f" {prefix} {letter}. {ch_text[:70]}"

        ax_txt.text(
            0.02, y_choices, display,
            transform=ax_txt.transAxes,
            fontsize=7.5, va="top", color=color, fontweight=weight,
        )
        y_choices -= 0.087


# ── page-level rendering ──────────────────────────────────────────────────────

VARIANT_LABELS = {
    ("action_phase_id", "A"): "Q1-A  action_phase_id  naive correct pairs",
    ("action_phase_id", "B"): "Q1-B  action_phase_id  first/last × correct/random image",
    ("next_action",     "A"): "Q3-A  next_action  trivial (no context)",
    ("next_action",     "B"): "Q3-B  next_action  with full sequence",
    ("next_action",     "C"): "Q3-C  next_action  with claimed current phase",
    ("phase_success",   ""):  "Q5    phase_success  5 claimed phases per image",
    ("progress",        ""):  "Q2    progress  up to 6 pair types",
    ("task_success",    ""):  "Q6    task_success  goal_understandable ground truth",
}


def render_page(pdf: PdfPages, key: tuple, samples: list[dict], root: Path) -> None:
    qt, variant = key
    page_title  = VARIANT_LABELS.get(key, f"{qt}" + (f" / {variant}" if variant else ""))

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(PAGE_BG)
    fig.suptitle(page_title, fontsize=12, fontweight="bold", y=0.99,
                 color="#1a2a4a", ha="center")

    n = len(samples)
    outer = gridspec.GridSpec(
        1, n, figure=fig,
        wspace=0.06, left=0.01, right=0.99, top=0.95, bottom=0.01,
    )

    for col, entry in enumerate(samples):
        draw_example(fig, outer[col], entry, root, f"Example {col + 1}")

    pdf.savefig(fig, bbox_inches="tight", facecolor=PAGE_BG)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/action_phase_dataset.jsonl")
    p.add_argument("--output",  default="data/dataset_samples.pdf")
    p.add_argument("--n",       type=int, default=2,
                   help="Number of examples to show per category/variant (default 2)")
    p.add_argument("--seed",    type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    root    = Path(__file__).parent.parent
    entries = load_jsonl(root / args.dataset)

    # Group by (question_type, variant)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for e in entries:
        key = (e["question_type"], e.get("variant", ""))
        groups[key].append(e)

    # Render in canonical order
    order = [
        ("action_phase_id", "A"),
        ("action_phase_id", "B"),
        ("progress",        ""),
        ("next_action",     "A"),
        ("next_action",     "B"),
        ("next_action",     "C"),
        ("phase_success",   ""),
        ("task_success",    ""),
    ]
    # append any unexpected keys not in order
    for key in groups:
        if key not in order:
            order.append(key)

    out_path = root / args.output
    print(f"Rendering {len(order)} pages -> {out_path}")

    with PdfPages(out_path) as pdf:
        for key in order:
            items = groups.get(key, [])
            if not items:
                print(f"  skip {key} (no entries)")
                continue
            samples = random.sample(items, min(args.n, len(items)))
            render_page(pdf, key, samples, root)
            qt, v = key
            label = VARIANT_LABELS.get(key, f"{qt}/{v}")
            print(f"  {label}  ({len(items)} total, {len(samples)} shown)")

    print(f"Done.")


if __name__ == "__main__":
    main()
