#!/usr/bin/env python3
"""
Annotation viewer for multiview tiles — scene-by-scene view.

Per question annotates:
  - Step number (temporal order within scene; same step = simultaneous)
  - Action phase description (shared across same-step group)
  - Safe to proceed to next phase: Yes / No / Unsure (shared)
  - Task completed at this step: checkbox (shared)
  - End goal understandable: per-tile-position checkbox (can you tell if overall task done?)
  - Action phase understandable: per-tile-position checkbox (can you tell if current phase done?)
  - Two question modifications (text shared scene-wide) with individual "solved here" checkboxes
    (solved cascade: checking step N auto-checks all later steps)

Saves to JSONL (auto-saves on prev/next/close).

Usage:
  python scripts/annotate_tiles.py
  python scripts/annotate_tiles.py --tiles data/multiview_tiles/entries.jsonl
  python scripts/annotate_tiles.py --output data/multiview_tiles/tile_annotations.jsonl

Controls:
  Right    next scene
  Left     previous scene
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
YELLOW  = "#f9e2af"
FONT    = ("monospace", 10)
FONT_SM = ("monospace", 9)
FONT_BD = ("monospace", 10, "bold")

TILE_MAX_W = 280


def load_entries(path: Path) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def group_by_scene(entries: list[dict]) -> list[dict]:
    scene_questions: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    scene_order: list[str] = []
    for e in entries:
        if e.get("split_mode") != "4":
            continue
        if "successfully completed" not in e.get("question", "").lower():
            continue
        sid, oid = e["scene_id"], e["original_id"]
        if sid not in scene_questions:
            scene_order.append(sid)
        scene_questions[sid][oid].append(e)

    scenes = []
    for sid in scene_order:
        questions = []
        for oid, tiles in scene_questions[sid].items():
            split_mode = tiles[0].get("split_mode", "4")
            primary_pos = ["left", "right"] if split_mode == "2" else ["top_left", "top_right", "bottom_left"]
            primary = sorted(
                [t for t in tiles if t["tile_position"] in primary_pos],
                key=lambda t: primary_pos.index(t["tile_position"]) if t["tile_position"] in primary_pos else 99,
            )
            questions.append({"original_id": oid, "tiles": primary})
        scenes.append({"scene_id": sid, "questions": questions})
    return scenes


class TileAnnotator:
    def __init__(self, root: tk.Tk, scenes: list[dict], annotations: dict, out_path: Path):
        self.root        = root
        self.scenes      = scenes
        self.annotations = annotations
        self.out_path    = out_path
        self.idx         = 0

        # Per-oid widget state (cleared on scene change)
        self._step_vars:      dict[str, tk.StringVar]             = {}
        self._phase_vars:     dict[str, tk.StringVar]             = {}
        self._phase_entries:  dict[str, tk.Entry]                 = {}
        self._proceed_vars:   dict[str, tk.StringVar]             = {}
        self._completed_vars: dict[str, tk.BooleanVar]            = {}
        self._mod1_solved:    dict[str, tk.BooleanVar]            = {}
        self._mod2_solved:    dict[str, tk.BooleanVar]            = {}
        self._goal_vars:      dict[str, dict[str, tk.BooleanVar]] = {}
        self._phase_und_vars: dict[str, dict[str, tk.BooleanVar]] = {}
        self._photo_refs:     list[ImageTk.PhotoImage]            = []

        # Scene-level shared vars (one value for all questions in scene)
        self._mod1_text_var: tk.StringVar | None = None
        self._mod2_text_var: tk.StringVar | None = None

        self._cascade_lock: bool = False

        root.title("Tile Annotator")
        root.configure(bg=BG)
        root.geometry("960x860")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_chrome()
        self._load_scene()

    # ------------------------------------------------------------------ chrome

    def _build_chrome(self):
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=10, pady=(8, 4))

        self.scene_lbl = tk.Label(top, text="", bg=BG, fg=ACCENT, font=FONT_BD)
        self.scene_lbl.pack(side="left")

        btn = {"bg": SURFACE, "fg": FG, "activebackground": "#45475a",
               "font": FONT, "relief": "flat", "padx": 12, "pady": 5}
        tk.Button(top, text="Save",   command=self._save, **btn).pack(side="right", padx=4)
        tk.Button(top, text="Next →", command=self._next, **btn).pack(side="right", padx=4)
        self.counter_lbl = tk.Label(top, text="", bg=BG, fg=FG_DIM, font=FONT)
        self.counter_lbl.pack(side="right", padx=8)
        tk.Button(top, text="← Prev", command=self._prev, **btn).pack(side="right", padx=4)

        # Jump-to-scene entry
        tk.Label(top, text="Go:", bg=BG, fg=FG_OFF, font=FONT_SM).pack(side="right", padx=(12, 2))
        self._jump_var = tk.StringVar()
        jump_entry = tk.Entry(top, textvariable=self._jump_var, width=5, bg=SURFACE, fg=FG,
                              insertbackground=FG, font=FONT, relief="flat", justify="center")
        jump_entry.pack(side="right", padx=(0, 4))
        jump_entry.bind("<Return>", lambda _: self._jump_to_scene())

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=6, pady=4)

        self.canvas  = tk.Canvas(body, bg=BG, highlightthickness=0)
        sb           = tk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner   = tk.Frame(self.canvas, bg=BG)
        self._win_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>",  lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self._win_id, width=e.width))
        self.canvas.bind_all("<MouseWheel>", self._on_scroll)
        self.canvas.bind_all("<Button-4>",   self._on_scroll)
        self.canvas.bind_all("<Button-5>",   self._on_scroll)

        self.root.bind("<Right>", lambda _: self._next())
        self.root.bind("<Left>",  lambda _: self._prev())

    def _on_scroll(self, evt):
        self.canvas.yview_scroll(-1 if (evt.num == 4 or evt.delta > 0) else 1, "units")

    # ---------------------------------------------------------------- scene I/O

    def _load_scene(self):
        for w in self.inner.winfo_children():
            w.destroy()
        for d in (self._step_vars, self._phase_vars, self._phase_entries,
                  self._proceed_vars, self._completed_vars,
                  self._mod1_solved, self._mod2_solved,
                  self._goal_vars, self._phase_und_vars):
            d.clear()
        self._photo_refs.clear()

        scene = self.scenes[self.idx]
        self.scene_lbl.configure(text=f"Scene: {scene['scene_id']}")
        self.counter_lbl.configure(text=f"{self.idx + 1} / {len(self.scenes)}")

        # Resolve scene-level mod text from first annotation that has it
        mod1_text, mod2_text = "", ""
        for q in scene["questions"]:
            a = self.annotations.get(q["original_id"], {})
            if not mod1_text:
                mod1_text = a.get("mod1", "").strip()
            if not mod2_text:
                mod2_text = a.get("mod2", "").strip()
            if mod1_text and mod2_text:
                break
        # Default to task description (question minus trailing completion sentence)
        if not mod1_text and scene["questions"]:
            first_q = scene["questions"][0]
            raw = first_q["tiles"][0].get("question", "") if first_q["tiles"] else ""
            mod1_text = raw.split("Has the robot successfully completed")[0].strip().rstrip(".")
        if not mod2_text and scene["questions"]:
            first_q = scene["questions"][0]
            raw = first_q["tiles"][0].get("question", "") if first_q["tiles"] else ""
            mod2_text = raw.split("Has the robot successfully completed")[0].strip().rstrip(".")
        self._mod1_text_var = tk.StringVar(value=mod1_text)
        self._mod2_text_var = tk.StringVar(value=mod2_text)

        # Sort questions by annotated step; unannotated go last
        def _step_key(q):
            raw = str(self.annotations.get(q["original_id"], {}).get("step", ""))
            return (0, int(raw)) if raw.isdigit() else (1, 0)

        questions = sorted(scene["questions"], key=_step_key)

        for q_idx, q in enumerate(questions):
            self._build_question_row(q_idx, q)

        self.canvas.yview_moveto(0)

    def _build_question_row(self, q_idx: int, q: dict):
        oid   = q["original_id"]
        tiles = q["tiles"]
        first = tiles[0] if tiles else {}
        ann   = self.annotations.get(oid, {})

        # ── separator + header ──────────────────────────────────────────────
        tk.Frame(self.inner, bg=SURFACE, height=1).pack(fill="x", padx=10, pady=(14, 0))

        short_id = oid.split("_q")[-1] if "_q" in oid else str(q_idx + 1)
        tk.Label(self.inner, text=f"Q{short_id}  —  {first.get('task', '')}",
                 bg=BG, fg=ACCENT, font=FONT_BD, anchor="w").pack(fill="x", padx=12, pady=(4, 2))

        q_text = first.get("question", "")
        if q_text:
            tk.Label(self.inner, text=q_text, bg=BG, fg=FG_DIM,
                     font=FONT_SM, anchor="w", wraplength=900, justify="left"
                     ).pack(fill="x", padx=12, pady=(0, 4))

        # ── tiles ────────────────────────────────────────────────────────────
        tile_row = tk.Frame(self.inner, bg=BG)
        tile_row.pack(padx=12, pady=4, anchor="w")
        for tile in tiles:
            col = tk.Frame(tile_row, bg=BG)
            col.pack(side="left", padx=(0, 8))
            tk.Label(col, text=tile["tile_position"], bg=BG, fg=FG_DIM, font=FONT_SM).pack()
            img = Image.open(tile["image_path"]).convert("RGB")
            if img.width > TILE_MAX_W:
                ratio = TILE_MAX_W / img.width
                img = img.resize((TILE_MAX_W, int(img.height * ratio)), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._photo_refs.append(photo)
            tk.Label(col, image=photo, bg=BG2).pack()

        # ── per-tile understandability checkboxes ────────────────────────────
        saved_goal  = ann.get("goal_understandable", {})
        saved_phase = ann.get("phase_understandable", {})
        goal_pos_vars:  dict[str, tk.BooleanVar] = {}
        phase_pos_vars: dict[str, tk.BooleanVar] = {}

        for label, saved, pos_vars in [
            ("End goal done:", saved_goal, goal_pos_vars),
            ("Phase done:",    saved_phase, phase_pos_vars),
        ]:
            row = tk.Frame(self.inner, bg=BG)
            row.pack(padx=12, pady=(1, 1), anchor="w")
            tk.Label(row, text=label, bg=BG, fg=FG_DIM, font=FONT_SM, width=15, anchor="w").pack(side="left", padx=(0, 4))
            for tile in tiles:
                pos = tile["tile_position"]
                var = tk.BooleanVar(value=saved.get(pos, False))
                tk.Checkbutton(row, text=pos, variable=var,
                               bg=BG, fg=FG_DIM, selectcolor=SURFACE, activebackground=BG,
                               font=FONT_SM).pack(side="left", padx=(0, 12))
                pos_vars[pos] = var

        self._goal_vars[oid]      = goal_pos_vars
        self._phase_und_vars[oid] = phase_pos_vars

        # ── annotation grid ──────────────────────────────────────────────────
        grid = tk.Frame(self.inner, bg=BG)
        grid.pack(fill="x", padx=12, pady=(6, 0))
        grid.columnconfigure(1, weight=1)

        # Row 0: Step
        tk.Label(grid, text="Step:", bg=BG, fg=FG, font=FONT).grid(row=0, column=0, sticky="w", pady=3)
        step_var = tk.StringVar(value=str(ann.get("step", "")))
        tk.Entry(grid, textvariable=step_var, width=4, bg=SURFACE, fg=FG,
                 insertbackground=FG, font=FONT, relief="flat", justify="center"
                 ).grid(row=0, column=1, sticky="w", padx=(8, 0), pady=3)
        step_var.trace_add("write", lambda *_: self._sync_same_steps())

        # Row 1: Action phase
        tk.Label(grid, text="Action phase:", bg=BG, fg=FG, font=FONT).grid(row=1, column=0, sticky="w", pady=3)
        phase_var = tk.StringVar(value=ann.get("action_phase", ""))
        phase_entry = tk.Entry(grid, textvariable=phase_var, width=70, bg=SURFACE, fg=FG,
                               insertbackground=FG, font=FONT, relief="flat")
        phase_entry.grid(row=1, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=3)

        # Row 2: Safe to proceed + task completed
        tk.Label(grid, text="Safe to proceed:", bg=BG, fg=FG, font=FONT).grid(row=2, column=0, sticky="w", pady=3)
        proceed_frame = tk.Frame(grid, bg=BG)
        proceed_frame.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=3)
        proceed_var = tk.StringVar(value=ann.get("safe_to_proceed", "not_assigned"))
        for val, label, color in [("yes", "Yes", "#a6e3a1"), ("no", "No", "#f38ba8"), ("unsure", "Unsure", FG_DIM), ("not_assigned", "—", FG_OFF)]:
            tk.Radiobutton(proceed_frame, text=label, variable=proceed_var, value=val,
                           bg=BG, fg=color, selectcolor=SURFACE,
                           activebackground=BG, font=FONT).pack(side="left", padx=6)

        completed_var = tk.BooleanVar(value=ann.get("task_completed", False))
        tk.Checkbutton(grid, text="Task completed here", variable=completed_var,
                       bg=BG, fg=YELLOW, selectcolor=SURFACE, activebackground=BG,
                       font=FONT).grid(row=2, column=2, sticky="w", padx=(20, 0), pady=3)

        # Row 3: Modification 1 — text shared scene-wide, solved per-question
        tk.Label(grid, text="Mod 1:", bg=BG, fg=FG_DIM, font=FONT).grid(row=3, column=0, sticky="w", pady=3)
        tk.Entry(grid, textvariable=self._mod1_text_var, width=60, bg=SURFACE, fg=FG,
                 insertbackground=FG, font=FONT, relief="flat"
                 ).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=3)
        mod1_solved = tk.BooleanVar(value=ann.get("mod1_solved", False))
        tk.Checkbutton(grid, text="Solved here", variable=mod1_solved,
                       bg=BG, fg=YELLOW, selectcolor=SURFACE, activebackground=BG,
                       font=FONT).grid(row=3, column=2, sticky="w", padx=(12, 0), pady=3)

        # Row 4: Modification 2 — text shared scene-wide, solved per-question
        tk.Label(grid, text="Mod 2:", bg=BG, fg=FG_DIM, font=FONT).grid(row=4, column=0, sticky="w", pady=3)
        tk.Entry(grid, textvariable=self._mod2_text_var, width=60, bg=SURFACE, fg=FG,
                 insertbackground=FG, font=FONT, relief="flat"
                 ).grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=3)
        mod2_solved = tk.BooleanVar(value=ann.get("mod2_solved", False))
        tk.Checkbutton(grid, text="Solved here", variable=mod2_solved,
                       bg=BG, fg=YELLOW, selectcolor=SURFACE, activebackground=BG,
                       font=FONT).grid(row=4, column=2, sticky="w", padx=(12, 0), pady=3)

        self._step_vars[oid]      = step_var
        self._phase_vars[oid]     = phase_var
        self._phase_entries[oid]  = phase_entry
        self._proceed_vars[oid]   = proceed_var
        self._completed_vars[oid] = completed_var
        self._mod1_solved[oid]    = mod1_solved
        self._mod2_solved[oid]    = mod2_solved

        # Cascade traces registered after dicts are populated
        mod1_solved.trace_add("write", lambda *_, o=oid: self._cascade_solved(o, 1))
        mod2_solved.trace_add("write", lambda *_, o=oid: self._cascade_solved(o, 2))

    # --------------------------------------------------------- same-step sync

    def _sync_same_steps(self):
        scene_oids = [q["original_id"] for q in self.scenes[self.idx]["questions"]]

        step_groups: dict[str, list[str]] = defaultdict(list)
        for oid in scene_oids:
            step = self._step_vars[oid].get().strip()
            if step:
                step_groups[step].append(oid)

        # Reset all to own vars / enabled
        for oid in scene_oids:
            e = self._phase_entries[oid]
            e.configure(textvariable=self._phase_vars[oid], state="normal", bg=SURFACE, fg=FG)

        for step, group in step_groups.items():
            if len(group) < 2:
                continue
            ordered = [o for o in scene_oids if o in group]
            primary = ordered[0]
            p_phase   = self._phase_vars[primary]
            p_proceed = self._proceed_vars[primary]
            p_done    = self._completed_vars[primary]

            if not p_phase.get().strip():
                for o in ordered[1:]:
                    if self._phase_vars[o].get().strip():
                        p_phase.set(self._phase_vars[o].get())
                        break

            for oid in ordered[1:]:
                self._phase_entries[oid].configure(
                    textvariable=p_phase, state="disabled", bg=BG2, fg=FG_OFF)
                self._proceed_vars[oid].set(p_proceed.get())
                self._completed_vars[oid].set(p_done.get())

    # --------------------------------------------------------- solved cascade

    def _cascade_solved(self, oid: str, mod: int):
        if self._cascade_lock:
            return
        solved_dict = self._mod1_solved if mod == 1 else self._mod2_solved
        if not solved_dict.get(oid) or not solved_dict[oid].get():
            return  # only cascade when checking, not unchecking

        step_raw = self._step_vars.get(oid, tk.StringVar()).get().strip()
        if not step_raw.isdigit():
            return
        step = int(step_raw)

        self._cascade_lock = True
        try:
            for other_oid, other_step_var in self._step_vars.items():
                if other_oid == oid:
                    continue
                other_raw = other_step_var.get().strip()
                if other_raw.isdigit() and int(other_raw) > step:
                    solved_dict[other_oid].set(True)
        finally:
            self._cascade_lock = False

    # ----------------------------------------------------------------- save

    def _save_current(self):
        self._sync_same_steps()
        scene_oids = [q["original_id"] for q in self.scenes[self.idx]["questions"]]

        # Resolve shared phase across same-step groups
        step_groups: dict[str, list[str]] = defaultdict(list)
        for oid in scene_oids:
            step = self._step_vars[oid].get().strip()
            if step:
                step_groups[step].append(oid)
        shared_phase: dict[str, str] = {}
        for step, group in step_groups.items():
            if len(group) > 1:
                ordered = [o for o in scene_oids if o in group]
                val = self._phase_vars[ordered[0]].get().strip()
                for oid in group:
                    shared_phase[oid] = val

        mod1_text = self._mod1_text_var.get().strip() if self._mod1_text_var else ""
        mod2_text = self._mod2_text_var.get().strip() if self._mod2_text_var else ""

        for oid in scene_oids:
            step_raw   = self._step_vars[oid].get().strip()
            goal_und   = {pos: var.get() for pos, var in self._goal_vars.get(oid, {}).items()}
            phase_und  = {pos: var.get() for pos, var in self._phase_und_vars.get(oid, {}).items()}
            self.annotations[oid] = {
                "original_id":          oid,
                "scene_id":             self.scenes[self.idx]["scene_id"],
                "step":                 int(step_raw) if step_raw.isdigit() else step_raw,
                "action_phase":         shared_phase.get(oid, self._phase_vars[oid].get().strip()),
                "safe_to_proceed":      self._proceed_vars[oid].get(),
                "task_completed":       self._completed_vars[oid].get(),
                "goal_understandable":  goal_und,
                "phase_understandable": phase_und,
                "mod1":                 mod1_text,
                "mod1_solved":          self._mod1_solved[oid].get(),
                "mod2":                 mod2_text,
                "mod2_solved":          self._mod2_solved[oid].get(),
            }

    # ------------------------------------------------------------------ nav

    def _jump_to_scene(self):
        raw = self._jump_var.get().strip()
        if not raw.isdigit():
            return
        target = int(raw) - 1  # 1-based input → 0-based index
        if 0 <= target < len(self.scenes):
            self._save_current()
            self.idx = target
            self._load_scene()
        self._jump_var.set("")

    def _prev(self):
        self._save_current()
        if self.idx > 0:
            self.idx -= 1
            self._load_scene()

    def _next(self):
        self._save_current()
        if self.idx < len(self.scenes) - 1:
            self.idx += 1
            self._load_scene()

    def _save(self):
        self._save_current()
        with open(self.out_path, "w", encoding="utf-8") as f:
            for ann in self.annotations.values():
                f.write(json.dumps(ann) + "\n")
        print(f"Saved {len(self.annotations)} annotations → {self.out_path}")

    def _on_close(self):
        self._save()
        self.root.destroy()


def main():
    p = argparse.ArgumentParser(description="Annotate multiview tile pairs scene by scene")
    p.add_argument("--tiles",  default="data/multiview_tiles/entries.jsonl", metavar="PATH")
    p.add_argument("--output", default="data/multiview_tiles/tile_annotations.jsonl", metavar="PATH")
    args = p.parse_args()

    project_root = Path(__file__).parent.parent
    tiles_path = Path(args.tiles)  if Path(args.tiles).is_absolute()  else project_root / args.tiles
    out_path   = Path(args.output) if Path(args.output).is_absolute() else project_root / args.output

    if not tiles_path.exists():
        print(f"Tiles not found at {tiles_path} — run tile_multiview.py first")
        sys.exit(1)

    entries = load_entries(tiles_path)
    groups  = group_by_scene(entries)
    print(f"{len(groups)} scenes loaded")

    annotations: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    a = json.loads(line)
                    annotations[a["original_id"]] = a
        print(f"Resuming — {len(annotations)} existing annotations loaded")

    root = tk.Tk()
    TileAnnotator(root, groups, annotations, out_path)
    root.mainloop()


if __name__ == "__main__":
    main()
