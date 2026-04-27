#!/usr/bin/env python3
"""
Interactive dataset curation viewer.

Browse Robo2VLM-1 entries one-by-one, toggle any number of buckets per entry,
confirm to advance. Buckets are defined in a YAML config file — edit it freely
to add, rename, recolor, or rebind buckets without touching this script.

Keyboard shortcuts:
  <bucket key>         toggle that bucket in/out of staging
  Enter / Space        confirm staged selection and advance
                       (with empty staging = skip)
  Escape               clear staging without advancing
  Backspace / Left     undo last confirmed action (removes from all buckets)
  Q                    quit

Bucket config (--buckets, default: <output-dir>/buckets.yaml):
  buckets:
    - id: keep         # → <output-dir>/keep.jsonl
      name: Keep
      key: k           # optional single-char shortcut
      color: "#a6e3a1" # optional hex color

Usage examples:
  python scripts/curate.py --local-data /path/to/parquet
  python scripts/curate.py --task failure_mode --include-types task_success_TS-S
  python scripts/curate.py --exclude-scenes 14346 --limit 200
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import tkinter as tk
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import QUESTION_TYPE_TEMPLATES, TASK_FAILURE_MODE, TASK_MULTIVIEW
from data.dataset import Sample, load_dataset


# ── bucket config ─────────────────────────────────────────────────────────────

@dataclass
class Bucket:
    id: str
    name: str
    key: Optional[str] = None   # single-char shortcut
    color: str = "#585b70"


DEFAULT_COLORS = [
    "#a6e3a1", "#f9e2af", "#f38ba8", "#fab387", "#89b4fa",
    "#b4befe", "#cba6f7", "#94e2d5", "#89dceb", "#f5c2e7",
    "#cdd6f4", "#a6adc8", "#6c7086", "#45475a", "#313244",
]

DEFAULT_BUCKETS_YAML = """\
# Bucket configuration for curate.py
# Edit freely — no code changes needed. Restart the viewer to pick up changes.
#
# Fields per bucket:
#   id:    unique ID, used as the output filename  (required)
#   name:  label shown in the UI                  (required)
#   key:   single-character keyboard shortcut      (optional)
#   color: hex button color                        (optional)
#
# Workflow:
#   Toggle buckets with their key or by clicking.
#   Press Enter/Space to confirm + advance.
#   Escape clears the current selection without advancing.
#   Backspace/Left undoes the last confirmed action.

buckets:

  # ── General quality ────────────────────────────────────────────────────────
  - id: good
    name: Good
    key: g
    color: "#a6e3a1"

  - id: ambiguous
    name: Ambiguous
    key: a
    color: "#f9e2af"

  - id: bad
    name: Bad
    key: "b"
    color: "#f38ba8"

  # ── Failure mode question types ────────────────────────────────────────────
  - id: task_success_TS-S
    name: Task Success (TS-S)
    color: "#89b4fa"

  - id: grasp_stability_TS-G
    name: Grasp Stable (TS-G)
    color: "#b4befe"

  - id: goal_configuration_TS-GL
    name: Goal Config (TS-GL)
    color: "#cba6f7"

  - id: gripper_state_RS
    name: Gripper State (RS)
    color: "#94e2d5"

  - id: obstacle_detection_OS
    name: Obstacle (OS)
    color: "#89dceb"

  - id: relative_direction_SR
    name: Rel. Direction (SR)
    color: "#fab387"

  - id: grasp_phase_current_AU
    name: Grasp Phase (AU)
    color: "#f9e2af"

  - id: temporal_sequence_AU
    name: Temporal Seq (AU)
    color: "#f5c2e7"

  - id: grasp_phase_next_IP
    name: Next Phase (IP)
    color: "#fab387"

  - id: action_direction_IP
    name: Action Dir (IP)
    color: "#f38ba8"

  - id: trajectory_understanding_TU
    name: Trajectory (TU)
    color: "#cdd6f4"

  # ── Multiview question types ───────────────────────────────────────────────
  - id: cross_view_correspondence_MV
    name: Cross-view (MV)
    color: "#89b4fa"

  - id: relative_depth_SU
    name: Rel. Depth (SU)
    color: "#94e2d5"
"""


def load_buckets(path: Path) -> List[Bucket]:
    try:
        import yaml
    except ImportError:
        print("pyyaml not installed — run: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_BUCKETS_YAML, encoding="utf-8")
        print(f"Created bucket config: {path}")
        print("Edit it to add/remove/rename buckets, then rerun.")
        sys.exit(0)

    raw = yaml.safe_load(path.read_text("utf-8"))
    buckets = []
    for i, entry in enumerate(raw.get("buckets", [])):
        color = entry.get("color") or DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        buckets.append(Bucket(
            id=entry["id"],
            name=entry["name"],
            key=str(entry["key"]) if entry.get("key") is not None else None,
            color=color,
        ))
    return buckets


# ── filtering ─────────────────────────────────────────────────────────────────

def samples_from_file(path: Path) -> Iterator[Sample]:
    """Load pre-fetched entries from a JSONL file (images loaded from image_path)."""
    from PIL import Image as PILImage
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        img_path = entry.get("image_path")
        if not img_path or not Path(img_path).exists():
            continue
        try:
            image = PILImage.open(io.BytesIO(Path(img_path).read_bytes())).convert("RGB")
        except Exception:
            continue
        metadata = {k: entry[k] for k in (
            "scene_id", "question_types", "question_group", "question_parts", "raw_row"
        ) if k in entry}
        yield Sample(
            id=entry["id"],
            task=entry["task"],
            image=image,
            question=entry["question"],
            choices=entry["choices"],
            correct_answer=entry["correct_answer"],
            metadata=metadata,
        )


def filtered_samples(
    task_filter: Optional[str],
    include_types: Set[str],
    exclude_types: Set[str],
    include_scenes: Set[str],
    exclude_scenes: Set[str],
    include_ids: Set[str],
    split: str,
    limit: Optional[int],
    local_data: Optional[str],
) -> Iterator[Sample]:
    seen = 0
    for sample in load_dataset(split=split, task_filter=task_filter, local_path=local_data):
        if limit is not None and seen >= limit:
            break
        if include_ids and sample.id not in include_ids:
            continue
        types = set(sample.metadata.get("question_types", []))
        scene = sample.metadata.get("scene_id", "")
        if include_types and not types & include_types:
            continue
        if exclude_types and types & exclude_types:
            continue
        if include_scenes and scene not in include_scenes:
            continue
        if exclude_scenes and scene in exclude_scenes:
            continue
        yield sample
        seen += 1


# ── theme ─────────────────────────────────────────────────────────────────────

DARK   = "#1e1e2e"
MID    = "#2a2a3e"
PANEL  = "#181825"
LIGHT  = "#cdd6f4"
ACCENT = "#89b4fa"
GREEN  = "#a6e3a1"
YELLOW = "#f9e2af"
MUTED  = "#585b70"
STAGED_BORDER = "#f9e2af"


# ── app ───────────────────────────────────────────────────────────────────────

class CurationApp:
    IMG_MAX_W = 600
    IMG_MAX_H = 420
    BUCKETS_PER_ROW = 8

    def __init__(
        self,
        iterator: Iterator[Sample],
        buckets: List[Bucket],
        output_dir: Path,
    ):
        self.iterator = iterator
        self.buckets = buckets
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # key → bucket
        self._key_map: Dict[str, Bucket] = {
            b.key: b for b in buckets if b.key
        }

        self.current: Optional[Sample] = None
        # prev = (sample, frozenset_of_bucket_ids_or_None)
        self.prev: Optional[Tuple[Sample, Optional[frozenset]]] = None
        self.staging: Set[str] = set()   # bucket IDs currently toggled
        self.n_viewed = 0
        self.n_confirmed = 0
        self._photo_ref = None

        self._prefetched: Optional[Sample] = None
        self._prefetch_done = False
        self._prefetch_thread: Optional[threading.Thread] = None

        self._build_ui()
        self._load_next()
        self._start_prefetch()

    # ── ui ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("Dataset Curator")
        self.root.configure(bg=DARK)
        self.root.resizable(True, True)

        # top info bar
        top = tk.Frame(self.root, bg=MID, pady=5, padx=12)
        top.pack(fill=tk.X)
        self.lbl_id = tk.Label(top, text="", bg=MID, fg=LIGHT,
                               font=("Monospace", 10), anchor="w")
        self.lbl_id.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.lbl_prog = tk.Label(top, text="", bg=MID, fg=ACCENT,
                                 font=("Monospace", 10), anchor="e")
        self.lbl_prog.pack(side=tk.RIGHT)

        meta = tk.Frame(self.root, bg=MID, pady=2, padx=12)
        meta.pack(fill=tk.X)
        self.lbl_meta = tk.Label(meta, text="", bg=MID, fg=YELLOW,
                                 font=("Monospace", 9), anchor="w")
        self.lbl_meta.pack(side=tk.LEFT)

        tk.Frame(self.root, bg=MUTED, height=1).pack(fill=tk.X)

        # main: image + question
        main = tk.Frame(self.root, bg=DARK, padx=10, pady=10)
        main.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            main, width=self.IMG_MAX_W, height=self.IMG_MAX_H,
            bg="#11111b", highlightthickness=1, highlightbackground=MUTED,
        )
        self.canvas.pack(side=tk.LEFT, padx=(0, 10))
        self.canvas.create_text(
            self.IMG_MAX_W // 2, self.IMG_MAX_H // 2,
            text="Loading...", fill=MUTED, font=("Monospace", 14),
        )

        qframe = tk.Frame(main, bg=DARK)
        qframe.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(qframe, text="Question", bg=DARK, fg=MUTED,
                 font=("Monospace", 9)).pack(anchor="w")
        self.txt_q = tk.Text(
            qframe, wrap=tk.WORD, height=7, width=44,
            bg=MID, fg=LIGHT, font=("Monospace", 11),
            bd=0, padx=8, pady=8, state=tk.DISABLED,
        )
        self.txt_q.pack(fill=tk.BOTH, expand=True, pady=(2, 8))

        tk.Label(qframe, text="Choices", bg=DARK, fg=MUTED,
                 font=("Monospace", 9)).pack(anchor="w")
        self.txt_c = tk.Text(
            qframe, wrap=tk.WORD, height=9, width=44,
            bg=MID, fg=LIGHT, font=("Monospace", 11),
            bd=0, padx=8, pady=8, state=tk.DISABLED,
        )
        self.txt_c.tag_configure("correct", foreground=GREEN,
                                 font=("Monospace", 11, "bold"))
        self.txt_c.pack(fill=tk.BOTH, expand=True)

        # bucket panel
        tk.Frame(self.root, bg=MUTED, height=1).pack(fill=tk.X)
        bpanel = tk.Frame(self.root, bg=DARK, padx=10, pady=6)
        bpanel.pack(fill=tk.X)

        self._btn_widgets: Dict[str, tk.Button] = {}
        row_frame = None
        for idx, bucket in enumerate(self.buckets):
            if idx % self.BUCKETS_PER_ROW == 0:
                row_frame = tk.Frame(bpanel, bg=DARK)
                row_frame.pack(fill=tk.X, pady=1)
            hint = f"[{bucket.key}] " if bucket.key else ""
            btn = tk.Button(
                row_frame,
                text=f" {hint}{bucket.name} ",
                bg=bucket.color, fg="#1e1e2e",
                font=("Monospace", 9, "bold"),
                relief=tk.FLAT, cursor="hand2", pady=3,
                command=lambda bid=bucket.id: self._toggle(bid),
            )
            btn.pack(side=tk.LEFT, padx=2)
            self._btn_widgets[bucket.id] = btn

        # confirm / nav row
        nav = tk.Frame(self.root, bg=DARK, padx=10, pady=4)
        nav.pack(fill=tk.X)
        tk.Button(nav, text=" ✓ Confirm (Enter) ", bg="#a6e3a1", fg="#1e1e2e",
                  font=("Monospace", 10, "bold"), relief=tk.FLAT, cursor="hand2",
                  pady=4, command=self._confirm).pack(side=tk.LEFT, padx=2)
        tk.Button(nav, text=" ✕ Clear (Esc) ", bg=MUTED, fg=LIGHT,
                  font=("Monospace", 10), relief=tk.FLAT, cursor="hand2",
                  pady=4, command=self._clear_staging).pack(side=tk.LEFT, padx=2)
        tk.Button(nav, text=" ← Undo ", bg=MUTED, fg=LIGHT,
                  font=("Monospace", 10), relief=tk.FLAT, cursor="hand2",
                  pady=4, command=self._undo).pack(side=tk.LEFT, padx=6)

        # staging + status bar
        self.lbl_staging = tk.Label(
            self.root, text="  staging: —", bg=PANEL, fg=YELLOW,
            font=("Monospace", 9), anchor="w", padx=12, pady=2,
        )
        self.lbl_staging.pack(fill=tk.X)
        self.lbl_status = tk.Label(
            self.root, text="  ready", bg=PANEL, fg=ACCENT,
            font=("Monospace", 9), anchor="w", padx=12, pady=3,
        )
        self.lbl_status.pack(fill=tk.X)

        self.root.bind("<Key>", self._on_key)
        self.root.focus_set()

    # ── key handling ──────────────────────────────────────────────────────────

    def _on_key(self, event):
        k = event.keysym
        ch = event.char

        if ch and ch in self._key_map:
            self._toggle(self._key_map[ch].id)
        elif k in ("Return", "space"):
            self._confirm()
        elif k == "Escape":
            self._clear_staging()
        elif k in ("BackSpace", "Left"):
            self._undo()
        elif k.lower() == "q":
            self.root.quit()

    # ── navigation ────────────────────────────────────────────────────────────

    def _toggle(self, bucket_id: str):
        if bucket_id in self.staging:
            self.staging.discard(bucket_id)
        else:
            self.staging.add(bucket_id)
        self._refresh_buttons()
        self._refresh_staging_label()

    def _confirm(self):
        if self.current is None:
            return
        selected = frozenset(self.staging)
        for bid in selected:
            self._save_to_bucket(self.current, bid)
        if selected:
            self.n_confirmed += 1
        self.prev = (self.current, selected if selected else None)
        names = [b.name for b in self.buckets if b.id in selected]
        self._set_status("→ " + ", ".join(names) if names else "skipped")
        self.staging.clear()
        self._load_next()
        self._start_prefetch()

    def _clear_staging(self):
        self.staging.clear()
        self._refresh_buttons()
        self._refresh_staging_label()
        self._set_status("staging cleared")

    def _undo(self):
        if self.prev is None:
            self._set_status("nothing to undo")
            return
        prev_sample, prev_buckets = self.prev
        if prev_buckets:
            for bid in prev_buckets:
                self._remove_last_from_bucket(bid)
            self.n_confirmed -= 1
        self.current = prev_sample
        self.n_viewed -= 1
        self.prev = None
        self.staging.clear()
        self._refresh_buttons()
        self._display(self.current)
        self._set_status("undone — back to previous entry")

    def _load_next(self):
        if self._prefetched is not None:
            sample, self._prefetched = self._prefetched, None
        elif self._prefetch_done:
            self._set_status("dataset exhausted — press Q to quit")
            self.current = None
            return
        else:
            try:
                sample = next(self.iterator)
            except StopIteration:
                self._set_status("dataset exhausted — press Q to quit")
                self.current = None
                return

        self.n_viewed += 1
        self.current = sample
        self._refresh_buttons()
        self._refresh_staging_label()
        self._display(sample)
        self._set_status("ready")

    def _start_prefetch(self):
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            return

        def fetch():
            try:
                self._prefetched = next(self.iterator)
            except StopIteration:
                self._prefetch_done = True

        self._prefetch_thread = threading.Thread(target=fetch, daemon=True)
        self._prefetch_thread.start()

    # ── persistence ───────────────────────────────────────────────────────────

    def _save_to_bucket(self, sample: Sample, bucket_id: str):
        path = self.output_dir / f"{bucket_id}.jsonl"
        row = {
            "id": sample.id,
            "task": sample.task,
            "question": sample.question,
            "choices": sample.choices,
            "correct_answer": sample.correct_answer,
            "correct_choice": sample.correct_choice,
            **sample.metadata,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _remove_last_from_bucket(self, bucket_id: str):
        path = self.output_dir / f"{bucket_id}.jsonl"
        if not path.exists():
            return
        lines = [l for l in path.read_text("utf-8").splitlines() if l.strip()]
        if not lines:
            return
        tail = "\n".join(lines[:-1])
        path.write_text(tail + "\n" if tail else "", encoding="utf-8")

    # ── display ───────────────────────────────────────────────────────────────

    def _display(self, sample: Sample):
        self.lbl_id.config(text=f"  {sample.id}")
        self.lbl_prog.config(
            text=f"viewed: {self.n_viewed}   confirmed: {self.n_confirmed}  "
        )
        types_str = ", ".join(sample.metadata.get("question_types", []))
        group_str = sample.metadata.get("question_group", "")
        scene_str = sample.metadata.get("scene_id", "?")
        self.lbl_meta.config(
            text=f"  {sample.task}  |  {types_str}  |  {group_str}  |  scene {scene_str}"
        )
        self._show_image(sample.image)

        self.txt_q.config(state=tk.NORMAL)
        self.txt_q.delete("1.0", tk.END)
        self.txt_q.insert(tk.END, sample.question)
        self.txt_q.config(state=tk.DISABLED)

        self.txt_c.config(state=tk.NORMAL)
        self.txt_c.delete("1.0", tk.END)
        for i, choice in enumerate(sample.choices):
            line = f"  {'ABCDEFGHIJ'[i]}:  {choice}\n"
            self.txt_c.insert(tk.END, line)
            if i == sample.correct_answer:
                self.txt_c.tag_add("correct", f"{i + 1}.0", f"{i + 1}.end")
        self.txt_c.config(state=tk.DISABLED)

    def _show_image(self, img: Image.Image):
        w, h = img.size
        scale = min(self.IMG_MAX_W / w, self.IMG_MAX_H / h, 1.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        photo = ImageTk.PhotoImage(img.resize((nw, nh), Image.LANCZOS))
        self._photo_ref = photo
        self.canvas.config(width=nw, height=nh)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    def _refresh_buttons(self):
        for bucket in self.buckets:
            btn = self._btn_widgets[bucket.id]
            if bucket.id in self.staging:
                btn.config(
                    relief=tk.SOLID,
                    highlightbackground=STAGED_BORDER,
                    highlightthickness=3,
                    bg=bucket.color,
                )
            else:
                btn.config(relief=tk.FLAT, highlightthickness=0, bg=bucket.color)

    def _refresh_staging_label(self):
        if self.staging:
            names = [b.name for b in self.buckets if b.id in self.staging]
            self.lbl_staging.config(text=f"  staging: {', '.join(names)}")
        else:
            self.lbl_staging.config(text="  staging: —  (Enter to skip)")

    def _set_status(self, msg: str):
        self.lbl_status.config(text=f"  {msg}")

    def run(self):
        self.root.mainloop()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _comma_set(s: Optional[str]) -> Set[str]:
    return set(s.split(",")) if s else set()


def main():
    all_types = sorted(
        t for task_types in QUESTION_TYPE_TEMPLATES.values() for t in task_types
    )
    p = argparse.ArgumentParser(
        description="Interactive dataset curation viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available question types:\n  " + "\n  ".join(all_types),
    )
    p.add_argument("--task", choices=[TASK_FAILURE_MODE, TASK_MULTIVIEW],
                   help="filter by task (default: both)")
    p.add_argument("--include-types", metavar="T1,T2",
                   help="only show entries matching these question type keys")
    p.add_argument("--exclude-types", metavar="T1,T2",
                   help="skip entries matching these question type keys")
    p.add_argument("--include-scenes", metavar="S1,S2",
                   help="only show entries from these scene IDs (comma-separated)")
    p.add_argument("--exclude-scenes", metavar="S1,S2",
                   help="skip entries from these scene IDs (comma-separated)")
    p.add_argument("--scenes-file", metavar="PATH",
                   help="text file with scene IDs to include, one per line (or comma-separated)")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--limit", type=int, metavar="N")
    p.add_argument("--local-data", metavar="PATH",
                   help="local parquet directory instead of HuggingFace")
    p.add_argument("--output-dir", default="curated", metavar="DIR",
                   help="directory for bucket JSONL files (default: curated/)")
    p.add_argument("--buckets", metavar="PATH",
                   help="bucket config YAML (default: <output-dir>/buckets.yaml)")
    p.add_argument("--from-file", metavar="PATH",
                   help="JSONL file of entries to review (e.g. scenes/scene_14346.jsonl); "
                        "streams dataset filtered to those entry IDs")
    args = p.parse_args()

    project_root = Path(__file__).parent.parent
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else project_root / args.output_dir
    buckets_path = Path(args.buckets) if args.buckets else output_dir / "buckets.yaml"
    buckets = load_buckets(buckets_path)

    include_scenes = _comma_set(args.include_scenes)
    if args.scenes_file:
        text = Path(args.scenes_file).read_text("utf-8")
        include_scenes |= {s.strip() for s in text.replace(",", "\n").splitlines() if s.strip()}

    if args.from_file:
        it = samples_from_file(Path(args.from_file))
    else:
        it = filtered_samples(
            task_filter=args.task,
            include_types=_comma_set(args.include_types),
            exclude_types=_comma_set(args.exclude_types),
            include_scenes=include_scenes,
            exclude_scenes=_comma_set(args.exclude_scenes),
            include_ids=set(),
            split=args.split,
            limit=args.limit,
            local_data=args.local_data,
        )

    CurationApp(iterator=it, buckets=buckets, output_dir=output_dir).run()


if __name__ == "__main__":
    main()
