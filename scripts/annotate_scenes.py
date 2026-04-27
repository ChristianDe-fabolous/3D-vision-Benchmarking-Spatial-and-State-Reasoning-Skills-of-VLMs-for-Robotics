#!/usr/bin/env python3
"""
Annotate droid scenes — select frames as subtask steps or multiview candidates.

Shows all images of a scene in a grid. Click to annotate:
  Left-click:   add to subtask sequence (auto-numbered in click order)
                clicking an already-selected subtask frame deselects it
  Right-click:  toggle multiview candidate

Controls:
  Left-click    subtask (ordered)    Right-click   multiview
  C             clear all            S             skip (no save)
  →/N/Enter     save + next          ←/P           save + previous
  Q             quit

Reads scene JSONL files from scenes/ (output of fetch_scenes.py).
Saves annotations to data/annotations.jsonl.

Usage:
  python scripts/annotate_scenes.py
  python scripts/annotate_scenes.py --scenes-file scenes.txt
  python scripts/annotate_scenes.py --scenes-dir my_scenes/
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

MAX_COLS = 4
THUMB_SIZE = (320, 240)


def scene_task_name(scene_id: str) -> str:
    name = re.sub(r"^droid_", "", scene_id)
    name = re.sub(r"_\d+$", "", name)
    return name.replace("_", " ")


def load_scene_entries(scene_jsonl: Path) -> list[dict]:
    entries = []
    with open(scene_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    def q_num(e):
        m = re.search(r"_q(\d+)$", e["id"])
        return int(m.group(1)) if m else 999

    return sorted(entries, key=q_num)


def load_pil(entry: dict) -> Image.Image | None:
    path = entry.get("image_path")
    if not path or not Path(path).exists():
        return None
    try:
        img = Image.open(io.BytesIO(Path(path).read_bytes())).convert("RGB")
        img.thumbnail(THUMB_SIZE)
        return img
    except Exception:
        return None


def load_annotations(out_path: Path) -> dict[str, dict]:
    if not out_path.exists():
        return {}
    result = {}
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                a = json.loads(line)
                result[a["scene_id"]] = a
    return result


def write_annotations(out_path: Path, all_ann: dict):
    with open(out_path, "w", encoding="utf-8") as f:
        for a in all_ann.values():
            f.write(json.dumps(a) + "\n")


class SceneAnnotator:
    def __init__(self, scene_files: list[Path], out_path: Path):
        self.scene_files = scene_files
        self.out_path = out_path
        self.all_ann = load_annotations(out_path)

        self.idx = 0
        self.entries: list[dict] = []
        self.images: list[Image.Image | None] = []
        self.img_axes: list[plt.Axes] = []

        self.subtask: list[str] = []   # entry IDs in click order
        self.multiview: set[str] = set()

        self.fig: plt.Figure = None
        self.ax_title: plt.Axes = None
        self.ax_status: plt.Axes = None

    def scene_id(self) -> str:
        stem = self.scene_files[self.idx].stem
        return stem[len("scene_"):] if stem.startswith("scene_") else stem

    def load_scene(self):
        self.entries = load_scene_entries(self.scene_files[self.idx])
        self.images = [load_pil(e) for e in self.entries]
        sid = self.scene_id()
        if sid in self.all_ann:
            ann = self.all_ann[sid]
            self.subtask = [f["id"] for f in ann.get("subtask", [])]
            self.multiview = {f["id"] for f in ann.get("multiview", [])}
        else:
            self.subtask = []
            self.multiview = set()

    def setup_layout(self):
        n = len(self.entries)
        cols = max(1, min(MAX_COLS, n))
        rows = max(1, math.ceil(n / cols))

        self.fig.clf()
        self.fig.set_facecolor("#1e1e2e")

        gs = GridSpec(
            rows + 2, cols,
            figure=self.fig,
            top=0.97, bottom=0.02,
            left=0.01, right=0.99,
            hspace=0.06, wspace=0.03,
            height_ratios=[0.45] + [1.0] * rows + [0.2],
        )

        self.ax_title = self.fig.add_subplot(gs[0, :])
        self.ax_title.axis("off")
        self.ax_title.set_facecolor("#1e1e2e")

        self.img_axes = []
        for i in range(n):
            r, c = divmod(i, cols)
            ax = self.fig.add_subplot(gs[r + 1, c])
            ax.set_facecolor("#11111b")
            ax.axis("off")
            self.img_axes.append(ax)

        for i in range(n, rows * cols):
            r, c = divmod(i, cols)
            ax = self.fig.add_subplot(gs[r + 1, c])
            ax.axis("off")
            ax.set_facecolor("#1e1e2e")
            for sp in ax.spines.values():
                sp.set_visible(False)

        self.ax_status = self.fig.add_subplot(gs[rows + 1, :])
        self.ax_status.axis("off")
        self.ax_status.set_facecolor("#1e1e2e")

    def draw(self):
        sid = self.scene_id()
        task = scene_task_name(sid)

        self.ax_title.clear()
        self.ax_title.axis("off")
        self.ax_title.text(
            0.5, 0.65, f"[{self.idx + 1}/{len(self.scene_files)}]  {task}",
            transform=self.ax_title.transAxes,
            ha="center", va="center", fontsize=12, color="white", fontweight="bold",
        )
        self.ax_title.text(
            0.5, 0.1,
            f"{sid}   |   subtask: {len(self.subtask)}   multiview: {len(self.multiview)}",
            transform=self.ax_title.transAxes,
            ha="center", va="center", fontsize=8, color="#6c7086", fontfamily="monospace",
        )

        for entry, img, ax in zip(self.entries, self.images, self.img_axes):
            ax.clear()
            ax.axis("off")
            eid = entry["id"]
            in_sub = eid in self.subtask
            in_mv = eid in self.multiview

            if img is not None:
                ax.imshow(img)
            else:
                ax.set_facecolor("#313244")
                ax.text(0.5, 0.5, "no image", ha="center", va="center",
                        transform=ax.transAxes, color="#6c7086", fontsize=8)

            border_color = "#a6e3a1" if in_sub else "#89b4fa" if in_mv else "#313244"
            border_w = 4 if (in_sub or in_mv) else 1
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(border_color)
                sp.set_linewidth(border_w)

            if in_sub:
                order = self.subtask.index(eid) + 1
                ax.text(0.04, 0.93, str(order),
                        transform=ax.transAxes, fontsize=14, fontweight="bold",
                        color="#1e1e2e",
                        bbox=dict(facecolor="#a6e3a1", edgecolor="none", pad=2, boxstyle="round"))
            elif in_mv:
                ax.text(0.04, 0.93, "MV",
                        transform=ax.transAxes, fontsize=10, fontweight="bold",
                        color="#1e1e2e",
                        bbox=dict(facecolor="#89b4fa", edgecolor="none", pad=2, boxstyle="round"))

            m = re.search(r"_q(\d+)$", eid)
            if m:
                ax.text(0.97, 0.03, f"q{m.group(1)}",
                        transform=ax.transAxes, fontsize=7, color="#585b70", ha="right")

        self.ax_status.clear()
        self.ax_status.axis("off")
        self.ax_status.text(
            0.5, 0.5,
            "L-click: subtask (ordered)   R-click: multiview   C: clear   "
            "→/N: save+next   ←/P: save+prev   S: skip   Q: quit",
            transform=self.ax_status.transAxes,
            ha="center", va="center", fontsize=8, color="#585b70",
        )

        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes is None:
            return
        for i, ax in enumerate(self.img_axes):
            if event.inaxes is ax:
                eid = self.entries[i]["id"]
                if event.button == 1:
                    if eid in self.subtask:
                        self.subtask.remove(eid)
                    else:
                        self.multiview.discard(eid)
                        self.subtask.append(eid)
                elif event.button == 3:
                    if eid in self.multiview:
                        self.multiview.discard(eid)
                    else:
                        if eid in self.subtask:
                            self.subtask.remove(eid)
                        self.multiview.add(eid)
                self.draw()
                return

    def on_key(self, event):
        if event.key in ("right", "n", "enter"):
            self.save_current()
            self._go(self.idx + 1)
        elif event.key in ("left", "p"):
            self.save_current()
            self._go(self.idx - 1)
        elif event.key == "s":
            self._go(self.idx + 1)
        elif event.key == "c":
            self.subtask.clear()
            self.multiview.clear()
            self.draw()
        elif event.key == "q":
            plt.close(self.fig)

    def save_current(self):
        sid = self.scene_id()
        by_id = {e["id"]: e for e in self.entries}
        self.all_ann[sid] = {
            "scene_id": sid,
            "subtask": [
                {"id": eid, "order": i + 1, "image_path": by_id[eid].get("image_path")}
                for i, eid in enumerate(self.subtask) if eid in by_id
            ],
            "multiview": [
                {"id": eid, "image_path": by_id[eid].get("image_path")}
                for eid in self.multiview if eid in by_id
            ],
        }
        write_annotations(self.out_path, self.all_ann)

    def _go(self, new_idx: int):
        if new_idx < 0 or new_idx >= len(self.scene_files):
            return
        self.idx = new_idx
        self.load_scene()
        self.setup_layout()
        self.draw()

    def run(self):
        self.fig = plt.figure(figsize=(16, 10), facecolor="#1e1e2e")
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.load_scene()
        self.setup_layout()
        self.draw()
        plt.show()


# ── Curated scene list ────────────────────────────────────────────────────────
# Set SCENE_IDS to annotate specific scenes, or leave empty and set SCENE_COUNT
# to pick the first N droid scenes from the index.
SCENE_IDS: list[str] = [
    # "droid_remove_the_black_object_from_the_bowl_and_put_it_inside_the_box_14652",
    # "droid_remove_the_black_object_from_the_box_1353",
]
SCENE_COUNT: int = 10   # used only when SCENE_IDS is empty
# ──────────────────────────────────────────────────────────────────────────────


def first_n_droid_scenes(index_path: Path, n: int) -> list[str]:
    scenes: list[str] = []
    seen: set[str] = set()
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            sid = e.get("scene_id", "")
            if sid.startswith("droid") and e.get("split") == "train" and sid not in seen:
                seen.add(sid)
                scenes.append(sid)
            if len(scenes) >= n:
                break
    return scenes


def main():
    p = argparse.ArgumentParser(description="Annotate droid scenes")
    p.add_argument("--scenes-dir",  default="scenes", metavar="DIR",
                   help="directory with scene_<id>.jsonl files (default: scenes/)")
    p.add_argument("--scenes-file", default=None, metavar="PATH",
                   help="plain text file with one scene ID per line (e.g. scenes.txt)")
    p.add_argument("--count", type=int, default=None, metavar="N",
                   help="use first N droid scenes from index (overrides SCENE_COUNT)")
    p.add_argument("--index", default="data/index.jsonl", metavar="PATH")
    p.add_argument("--output", default="data/annotations.jsonl", metavar="PATH")
    args = p.parse_args()

    project_root = Path(__file__).parent.parent
    scenes_dir = (project_root / args.scenes_dir) if not Path(args.scenes_dir).is_absolute() else Path(args.scenes_dir)
    out_path = (project_root / args.output) if not Path(args.output).is_absolute() else Path(args.output)
    index_path = (project_root / args.index) if not Path(args.index).is_absolute() else Path(args.index)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve desired scene ID list
    if args.scenes_file:
        sf = Path(args.scenes_file) if Path(args.scenes_file).is_absolute() else project_root / args.scenes_file
        scene_ids = [l.strip() for l in sf.read_text().splitlines() if l.strip()]
    elif SCENE_IDS:
        scene_ids = SCENE_IDS
    else:
        count = args.count if args.count is not None else SCENE_COUNT
        scene_ids = first_n_droid_scenes(index_path, count)

    # Match scene files by reading scene_id from inside each JSONL
    # (handles truncated filenames produced by fetch_scenes.py)
    scene_id_set = set(scene_ids)
    order = {sid: i for i, sid in enumerate(scene_ids)}

    def _scene_id_from_file(path: Path) -> str | None:
        try:
            first = path.read_text(encoding="utf-8").split("\n")[0].strip()
            return json.loads(first).get("scene_id") if first else None
        except Exception:
            return None

    scene_files = []
    for f in scenes_dir.glob("scene_*.jsonl"):
        sid = _scene_id_from_file(f)
        if sid and sid in scene_id_set:
            scene_files.append((f, sid))

    scene_files.sort(key=lambda x: order.get(x[1], 9999))
    scene_files = [f for f, _ in scene_files]

    if not scene_files:
        print(f"No scene JSONL files found in {scenes_dir}/")
        sys.exit(1)

    print(f"{len(scene_files)} scenes  →  annotations: {out_path}")
    SceneAnnotator(scene_files, out_path).run()


if __name__ == "__main__":
    main()
