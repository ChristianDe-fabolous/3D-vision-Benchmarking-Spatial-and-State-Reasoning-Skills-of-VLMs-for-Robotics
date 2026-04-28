#!/usr/bin/env python3
"""
Viewer for action_phase_dataset.jsonl.

Shows each entry's image(s), question, choices, and correct answer.

Usage:
  python scripts/view_dataset.py
  python scripts/view_dataset.py --dataset data/action_phase_dataset.jsonl
  python scripts/view_dataset.py --type action_phase_id

Controls:
  Right / n   next entry
  Left  / p   previous entry
  1-9         filter by question type (shown in status bar)
  0           clear filter
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tkinter as tk
from PIL import Image, ImageTk

BG      = "#1e1e2e"
BG2     = "#11111b"
SURFACE = "#313244"
FG      = "#cdd6f4"
FG_DIM  = "#a6adc8"
FG_OFF  = "#6c7086"
ACCENT  = "#89b4fa"
GREEN   = "#a6e3a1"
RED     = "#f38ba8"
YELLOW  = "#f9e2af"
FONT    = ("monospace", 10)
FONT_SM = ("monospace", 9)
FONT_BD = ("monospace", 10, "bold")
FONT_LG = ("monospace", 12, "bold")

IMG_MAX_W = 380
IMG_MAX_H = 300

QTYPES = [
    "action_phase_id",
    "progress",
    "next_action",
    "phase_success",
]


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


class DatasetViewer:
    def __init__(self, root: tk.Tk, entries: list[dict]):
        self.root    = root
        self.all     = entries
        self.filter  = None   # None = show all, else question_type string
        self.entries = entries
        self.idx     = 0
        self._photo_refs: list[ImageTk.PhotoImage] = []

        root.title("Dataset Viewer")
        root.configure(bg=BG)
        root.geometry("1000x820")
        root.bind("<Right>",  lambda _: self._next())
        root.bind("<Left>",   lambda _: self._prev())
        root.bind("n",        lambda _: self._next())
        root.bind("p",        lambda _: self._prev())
        root.bind("0",        lambda _: self._set_filter(None))
        for i, qt in enumerate(QTYPES, 1):
            root.bind(str(i), lambda _, q=qt: self._set_filter(q))

        self._build_chrome()
        self._render()

    # ── chrome ────────────────────────────────────────────────────────────────

    def _build_chrome(self):
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=10, pady=(8, 4))

        self.title_lbl = tk.Label(top, text="", bg=BG, fg=ACCENT, font=FONT_BD)
        self.title_lbl.pack(side="left")

        btn = {"bg": SURFACE, "fg": FG, "activebackground": "#45475a",
               "font": FONT, "relief": "flat", "padx": 10, "pady": 4}
        tk.Button(top, text="Next →", command=self._next, **btn).pack(side="right", padx=4)
        self.counter_lbl = tk.Label(top, text="", bg=BG, fg=FG_DIM, font=FONT)
        self.counter_lbl.pack(side="right", padx=8)
        tk.Button(top, text="← Prev", command=self._prev, **btn).pack(side="right", padx=4)

        # Jump box
        tk.Label(top, text="Go:", bg=BG, fg=FG_OFF, font=FONT_SM).pack(side="right", padx=(12, 2))
        self._jump_var = tk.StringVar()
        je = tk.Entry(top, textvariable=self._jump_var, width=6, bg=SURFACE, fg=FG,
                      insertbackground=FG, font=FONT, relief="flat", justify="center")
        je.pack(side="right", padx=(0, 4))
        je.bind("<Return>", lambda _: self._jump())

        # Filter buttons
        filter_frame = tk.Frame(self.root, bg=BG)
        filter_frame.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(filter_frame, text="Filter:", bg=BG, fg=FG_OFF, font=FONT_SM).pack(side="left", padx=(0, 6))
        tk.Button(filter_frame, text="All [0]", command=lambda: self._set_filter(None),
                  bg=SURFACE, fg=FG, relief="flat", font=FONT_SM, padx=6).pack(side="left", padx=2)
        for i, qt in enumerate(QTYPES, 1):
            short = qt.replace("_", " ")
            tk.Button(filter_frame, text=f"{short} [{i}]",
                      command=lambda q=qt: self._set_filter(q),
                      bg=SURFACE, fg=FG_DIM, relief="flat", font=FONT_SM, padx=6).pack(side="left", padx=2)

        # Scrollable body
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=6, pady=4)

        self.canvas = tk.Canvas(body, bg=BG, highlightthickness=0)
        sb = tk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner   = tk.Frame(self.canvas, bg=BG)
        self._win_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda _: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(
            self._win_id, width=e.width))
        self.canvas.bind_all("<MouseWheel>", self._on_scroll)
        self.canvas.bind_all("<Button-4>",   self._on_scroll)
        self.canvas.bind_all("<Button-5>",   self._on_scroll)

    def _on_scroll(self, evt):
        self.canvas.yview_scroll(-1 if (evt.num == 4 or evt.delta > 0) else 1, "units")

    # ── rendering ────────────────────────────────────────────────────────────

    def _render(self):
        for w in self.inner.winfo_children():
            w.destroy()
        self._photo_refs.clear()

        if not self.entries:
            tk.Label(self.inner, text="No entries match filter.", bg=BG, fg=FG_DIM,
                     font=FONT_LG).pack(pady=40)
            self.title_lbl.configure(text="— empty —")
            self.counter_lbl.configure(text="0 / 0")
            return

        e = self.entries[self.idx]
        qt = e.get("question_type", "")
        sid = e.get("scene_id", "")

        self.title_lbl.configure(text=f"{qt}  |  {sid[:60]}")
        self.counter_lbl.configure(
            text=f"{self.idx + 1} / {len(self.entries)}"
                 + (f"  (filtered: {qt})" if self.filter else "")
        )

        # ── metadata strip ───────────────────────────────────────────────────
        meta_frame = tk.Frame(self.inner, bg=BG2)
        meta_frame.pack(fill="x", padx=8, pady=(6, 0))
        meta_items = [
            ("id",    str(e.get("id", ""))),
            ("type",  qt),
            ("step",  str(e.get("image_step", e.get("step_a", "—")))),
            ("phase", e.get("image_phase", e.get("phase_a", "—"))),
        ]
        if e.get("special_image"):
            meta_items.append(("image", e["special_image"]))
        for k, v in meta_items:
            tk.Label(meta_frame, text=f"{k}: {v[:60]}", bg=BG2, fg=FG_DIM,
                     font=FONT_SM, anchor="w").pack(side="left", padx=8, pady=3)

        # ── images ───────────────────────────────────────────────────────────
        images = e.get("images", [])
        img_row = tk.Frame(self.inner, bg=BG)
        img_row.pack(padx=8, pady=8, anchor="w")
        for i, img_path in enumerate(images):
            col = tk.Frame(img_row, bg=BG)
            col.pack(side="left", padx=(0, 8))
            label = "Image 1 (earlier)" if i == 0 and len(images) > 1 else \
                    "Image 2 (later)"   if i == 1 else f"Image {i+1}"
            tk.Label(col, text=label, bg=BG, fg=FG_DIM, font=FONT_SM).pack()
            try:
                img = Image.open(img_path).convert("RGB")
                if img.width > IMG_MAX_W:
                    img = img.resize((IMG_MAX_W, int(img.height * IMG_MAX_W / img.width)),
                                     Image.LANCZOS)
                if img.height > IMG_MAX_H:
                    img = img.resize((int(img.width * IMG_MAX_H / img.height), IMG_MAX_H),
                                     Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._photo_refs.append(photo)
                tk.Label(col, image=photo, bg=BG2).pack()
            except Exception:
                tk.Label(col, text=f"[cannot load]\n{Path(img_path).name}",
                         bg=BG2, fg=RED, font=FONT_SM, padx=20, pady=20).pack()

        # ── question ─────────────────────────────────────────────────────────
        tk.Frame(self.inner, bg=SURFACE, height=1).pack(fill="x", padx=8, pady=(4, 0))
        tk.Label(self.inner, text=e.get("question", ""), bg=BG, fg=FG,
                 font=FONT, anchor="w", justify="left", wraplength=940
                 ).pack(fill="x", padx=12, pady=(8, 4))

        # ── choices ──────────────────────────────────────────────────────────
        tk.Frame(self.inner, bg=SURFACE, height=1).pack(fill="x", padx=8, pady=(4, 0))
        correct_letter = e.get("answer", "")
        answer_text    = e.get("answer_text", "")

        choices_frame = tk.Frame(self.inner, bg=BG)
        choices_frame.pack(fill="x", padx=12, pady=8)

        for choice in e.get("choices", []):
            letter = choice[0] if choice else ""
            is_correct = letter == correct_letter
            color  = GREEN if is_correct else FG_DIM
            prefix = "✓ " if is_correct else "  "
            tk.Label(choices_frame, text=f"{prefix}{choice}", bg=BG, fg=color,
                     font=FONT_BD if is_correct else FONT, anchor="w"
                     ).pack(fill="x", pady=1)

        # ── extra metadata for Q3 ────────────────────────────────────────────
        if qt == "next_action" and e.get("claimed_phase"):
            tk.Frame(self.inner, bg=SURFACE, height=1).pack(fill="x", padx=8, pady=(4, 0))
            tk.Label(self.inner,
                     text=f"claimed phase: {e['claimed_phase']}  |  image true phase: {e.get('image_phase','—')}",
                     bg=BG, fg=YELLOW, font=FONT_SM, anchor="w").pack(fill="x", padx=12, pady=4)

        self.canvas.yview_moveto(0)

    # ── navigation ───────────────────────────────────────────────────────────

    def _next(self):
        if self.entries and self.idx < len(self.entries) - 1:
            self.idx += 1
            self._render()

    def _prev(self):
        if self.entries and self.idx > 0:
            self.idx -= 1
            self._render()

    def _jump(self):
        raw = self._jump_var.get().strip()
        if raw.isdigit():
            target = int(raw) - 1
            if 0 <= target < len(self.entries):
                self.idx = target
                self._render()
        self._jump_var.set("")

    def _set_filter(self, qt: str | None):
        self.filter  = qt
        self.entries = [e for e in self.all if qt is None or e.get("question_type") == qt]
        self.idx     = 0
        self._render()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/action_phase_dataset.jsonl")
    p.add_argument("--type",    default=None, help="Filter by question type on startup")
    args = p.parse_args()

    root_dir = Path(__file__).parent.parent
    path = Path(args.dataset) if Path(args.dataset).is_absolute() else root_dir / args.dataset

    if not path.exists():
        print(f"Dataset not found: {path}")
        sys.exit(1)

    entries = load_jsonl(path)
    print(f"Loaded {len(entries)} entries from {path}")

    root = tk.Tk()
    viewer = DatasetViewer(root, entries)
    if args.type:
        viewer._set_filter(args.type)
    root.mainloop()


if __name__ == "__main__":
    main()
