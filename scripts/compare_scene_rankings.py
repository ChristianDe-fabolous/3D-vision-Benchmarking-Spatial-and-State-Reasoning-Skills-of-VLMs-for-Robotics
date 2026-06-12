"""
Compare per-scene action_phase difficulty across the 7 zero-shot models, and
compare difficulty patterns across the 4 action_phase sub-tasks
(action_phase_id, progress, phase_success, task_success).

Reads the canonical "default" zero-shot action_phase results.jsonl + the
scene_ranking.json files produced by scripts/rank_scenes.py. Writes new files
only:
    outputs/scene_difficulty_analysis/cross_model_and_subtask_analysis.json
    outputs/scene_difficulty_analysis/SUMMARY.md
    outputs/scene_difficulty_analysis/subtask_correlation_heatmap.png

Usage:
    python scripts/compare_scene_rankings.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent

# model_id -> canonical "default" zero-shot action_phase results dir
MODEL_DIRS = {
    "gemma4-e2b": "outputs/2026-05-31_14-48-00_action_phase_gemma4-e2b_default_done",
    "gemma4-e4b": "outputs/slurm92952_action_phase_gemma4-e4b_default_done",
    "gemma4-31b": "outputs/slurm93218_action_phase_gemma4-31b_default_done",
    "internvl3-14b": "outputs/slurm97555_action_phase_internvl3-14b_default",
    "qwen3-4b": "outputs/slurm98113_action_phase_qwen3-4b_default",
    "qwen3-8b": "outputs/slurm10000_action_phase_qwen3-8b_default_done",
    "qwen3-32b": "outputs/slurm98014_action_phase_qwen3-32b_default",
}

# Models clearly above chance (overall acc 43.7-47.5%, vs ~33% uniform random).
# gemma4-e2b (30.0%) and gemma4-e4b (36.5%) are excluded from the "v2" denoised
# cross-subtask correlation: near-chance per-scene accuracy is mostly sampling
# noise and dilutes the signal.
STRONG_MODELS = ["gemma4-31b", "internvl3-14b", "qwen3-4b", "qwen3-8b", "qwen3-32b"]

SUBTASKS = ["action_phase_id", "progress", "phase_success", "task_success"]
SUBTASK_LABELS = {
    "action_phase_id": "Action Phase ID",
    "progress": "Progress",
    "phase_success": "Phase Success",
    "task_success": "Task Success",
}
OUT_DIR = PROJECT_ROOT / "outputs" / "scene_difficulty_analysis"


def load_results(model: str) -> list[dict]:
    path = PROJECT_ROOT / MODEL_DIRS[model] / "results.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def version2_matrix(
    models_subset: list[str],
    subtask_scene_acc: dict[str, dict[str, dict[str, float]]],
    scenes: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Average per-scene accuracy across models_subset first, then correlate
    the 4 subtask vectors once ("denoised" cross-subtask correlation)."""
    avg = {
        t: np.array([np.mean([subtask_scene_acc[m][t][s] for m in models_subset]) for s in scenes])
        for t in SUBTASKS
    }
    M = np.zeros((4, 4))
    for i, ti in enumerate(SUBTASKS):
        for j, tj in enumerate(SUBTASKS):
            M[i, j] = spearman(avg[ti], avg[tj])
    return M, avg


def bootstrap_ci(avg: dict[str, np.ndarray], n_boot: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """95% CI on the 4x4 correlation matrix via resampling scenes with replacement."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(avg.values())))
    boots = np.zeros((n_boot, 4, 4))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for i, ti in enumerate(SUBTASKS):
            for j, tj in enumerate(SUBTASKS):
                boots[b, i, j] = spearman(avg[ti][idx], avg[tj][idx])
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    return lo, hi


def plot_heatmap(matrix: np.ndarray, ci_lo: np.ndarray, ci_hi: np.ndarray, out_path: Path) -> None:
    labels = [SUBTASK_LABELS[t] for t in SUBTASKS]
    n = len(labels)

    display = matrix.copy()
    np.fill_diagonal(display, np.nan)  # diagonal is trivially 1.0, not informative

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e0e0e0")

    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    im = ax.imshow(display, cmap=cmap, vmin=-0.35, vmax=0.35)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "1.00", ha="center", va="center", color="#666", fontsize=10)
                continue
            val = matrix[i, j]
            sig = "*" if (ci_lo[i, j] > 0 or ci_hi[i, j] < 0) else ""
            color = "white" if abs(val) > 0.22 else "black"
            ax.text(j, i, f"{val:.2f}{sig}", ha="center", va="center", color=color, fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman correlation")

    ax.set_title(
        "Cross-sub-task scene-difficulty correlation\n"
        "(5 above-chance models, averaged; n=47 scenes)\n"
        "* = 95% bootstrap CI excludes 0",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    models = list(MODEL_DIRS.keys())

    # ---- load per-model results + scene rankings -------------------------
    all_results = {m: load_results(m) for m in models}

    scene_acc: dict[str, dict[str, float]] = {}  # model -> scene_id -> accuracy
    task_descs: dict[str, str] = {}
    for m in models:
        ranking = json.loads((PROJECT_ROOT / MODEL_DIRS[m] / "scene_ranking.json").read_text())["ranking"]
        scene_acc[m] = {r["scene_id"]: r["accuracy"] for r in ranking}
        for r in ranking:
            task_descs.setdefault(r["scene_id"], r["task_desc"])

    scenes = sorted(scene_acc[models[0]].keys())
    n_scenes = len(scenes)

    # ---- overall + per-subtask accuracy per model -------------------------
    overall_acc: dict[str, float] = {}
    subtask_acc: dict[str, dict[str, float]] = {}  # model -> subtask -> acc
    for m in models:
        rows = all_results[m]
        overall_acc[m] = sum(r["correct"] for r in rows) / len(rows)
        by_task: dict[str, list[bool]] = defaultdict(list)
        for r in rows:
            by_task[r["task"]].append(r["correct"])
        subtask_acc[m] = {t: sum(v) / len(v) for t, v in by_task.items()}

    # ---- Part 1: cross-model scene-difficulty similarity -------------------
    mat = np.array([[scene_acc[m][s] for m in models] for s in scenes])  # (scenes, models)

    corr = np.zeros((len(models), len(models)))
    for i in range(len(models)):
        for j in range(len(models)):
            corr[i, j] = spearman(mat[:, i], mat[:, j])

    iu = np.triu_indices(len(models), k=1)
    avg_pairwise_corr = float(np.nanmean(corr[iu]))

    scene_mean = mat.mean(axis=1)
    scene_std = mat.std(axis=1)

    consensus_order = np.argsort(scene_mean)  # ascending: hardest first
    consensus_hardest = []
    for idx in consensus_order[:10]:
        s = scenes[idx]
        consensus_hardest.append({
            "scene_id": s,
            "task_desc": task_descs[s],
            "mean_accuracy": round(float(scene_mean[idx]), 3),
            "std_accuracy": round(float(scene_std[idx]), 3),
            "per_model": {m: scene_acc[m][s] for m in models},
        })
    consensus_easiest = []
    for idx in consensus_order[-10:][::-1]:
        s = scenes[idx]
        consensus_easiest.append({
            "scene_id": s,
            "task_desc": task_descs[s],
            "mean_accuracy": round(float(scene_mean[idx]), 3),
            "std_accuracy": round(float(scene_std[idx]), 3),
            "per_model": {m: scene_acc[m][s] for m in models},
        })

    disagreement_order = np.argsort(scene_std)[::-1]  # descending: most disagreement first
    most_disagreement = []
    for idx in disagreement_order[:10]:
        s = scenes[idx]
        most_disagreement.append({
            "scene_id": s,
            "task_desc": task_descs[s],
            "mean_accuracy": round(float(scene_mean[idx]), 3),
            "std_accuracy": round(float(scene_std[idx]), 3),
            "per_model": {m: scene_acc[m][s] for m in models},
        })

    # ---- Part 2: cross-subtask difficulty patterns -------------------------
    # per-model 47 x 4 (scene x subtask) accuracy matrices
    per_model_subtask_corr = {}  # model -> 4x4 list
    subtask_scene_acc_all = {}  # model -> subtask -> {scene_id: acc}
    for m in models:
        rows = all_results[m]
        bucket: dict[tuple[str, str], list[bool]] = defaultdict(list)
        for r in rows:
            bucket[(r["scene_id"], r["task"])].append(r["correct"])
        subtask_scene_acc_all[m] = {
            t: {s: sum(bucket[(s, t)]) / len(bucket[(s, t)]) for s in scenes}
            for t in SUBTASKS
        }
        sub_mat = np.array([[subtask_scene_acc_all[m][t][s] for t in SUBTASKS] for s in scenes])
        c = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                c[i, j] = spearman(sub_mat[:, i], sub_mat[:, j])
        per_model_subtask_corr[m] = c

    avg_subtask_corr = np.nanmean(np.stack(list(per_model_subtask_corr.values())), axis=0)

    # subtask difficulty ranking: mean accuracy across 7 models
    subtask_mean_acc = {t: float(np.mean([subtask_acc[m][t] for m in models])) for t in SUBTASKS}
    subtask_ranking = sorted(subtask_mean_acc.items(), key=lambda kv: kv[1])

    # ---- Part 3: "v2" denoised cross-subtask correlation -------------------
    # average per-scene accuracy across models FIRST, then correlate once.
    v2_5_matrix, v2_5_avg = version2_matrix(STRONG_MODELS, subtask_scene_acc_all, scenes)
    v2_5_lo, v2_5_hi = bootstrap_ci(v2_5_avg)
    v2_7_matrix, _ = version2_matrix(models, subtask_scene_acc_all, scenes)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_heatmap(v2_5_matrix, v2_5_lo, v2_5_hi, OUT_DIR / "subtask_correlation_heatmap.png")

    # ---- assemble output ----------------------------------------------------
    out = {
        "models": models,
        "model_dirs": MODEL_DIRS,
        "n_scenes": n_scenes,
        "overall_accuracy": overall_acc,
        "subtask_accuracy": subtask_acc,
        "subtask_mean_accuracy_ranking": subtask_ranking,
        "cross_model_scene_correlation": {
            "method": "spearman, per-scene overall accuracy across 47 scenes",
            "matrix": corr.round(3).tolist(),
            "model_order": models,
            "average_pairwise_correlation": round(avg_pairwise_corr, 3),
        },
        "consensus_hardest_scenes": consensus_hardest,
        "consensus_easiest_scenes": consensus_easiest,
        "most_model_disagreement_scenes": most_disagreement,
        "cross_subtask_scene_difficulty_correlation": {
            "method": "v1 (naive): spearman per model across 47 scenes, then averaged over 7 models",
            "subtask_order": SUBTASKS,
            "average_matrix": avg_subtask_corr.round(3).tolist(),
            "per_model_matrix": {m: c.round(3).tolist() for m, c in per_model_subtask_corr.items()},
        },
        "subtask_difficulty_correlation_v2": {
            "method": "v2 (denoised): average per-scene accuracy across models first, then spearman once",
            "subtask_order": SUBTASKS,
            "5_strong_models": {
                "models": STRONG_MODELS,
                "matrix": v2_5_matrix.round(3).tolist(),
                "ci_lo": v2_5_lo.round(3).tolist(),
                "ci_hi": v2_5_hi.round(3).tolist(),
            },
            "7_models": {
                "models": models,
                "matrix": v2_7_matrix.round(3).tolist(),
            },
            "figure": "subtask_correlation_heatmap.png",
        },
    }

    json_path = OUT_DIR / "cross_model_and_subtask_analysis.json"
    json_path.write_text(json.dumps(out, indent=2))

    # ---- markdown summary -----------------------------------------------------
    md = []
    md.append("# Scene-difficulty & sub-task comparison (action_phase, 7 zero-shot models)\n")
    md.append(f"Scenes: {n_scenes}. Models: {', '.join(models)}.\n")

    md.append("\n## Overall + per-sub-task accuracy\n")
    md.append("| model | overall | " + " | ".join(SUBTASKS) + " |")
    md.append("|---|---|" + "---|" * len(SUBTASKS))
    for m in models:
        row = [f"{overall_acc[m]:.1%}"] + [f"{subtask_acc[m].get(t, float('nan')):.1%}" for t in SUBTASKS]
        md.append(f"| {m} | " + " | ".join(row) + " |")

    md.append("\n## Cross-model scene-difficulty similarity\n")
    md.append(f"Average pairwise Spearman correlation of per-scene accuracy across the 47 "
              f"scenes: **{avg_pairwise_corr:.3f}**.\n")
    md.append("Pairwise Spearman correlation matrix:\n")
    md.append("| model | " + " | ".join(models) + " |")
    md.append("|---|" + "---|" * len(models))
    for i, m in enumerate(models):
        row = [f"{corr[i, j]:.2f}" for j in range(len(models))]
        md.append(f"| {m} | " + " | ".join(row) + " |")

    md.append("\n## Consensus hardest scenes (lowest mean accuracy across all 7 models)\n")
    md.append("| scene | mean acc | std | task |")
    md.append("|---|---|---|---|")
    for s in consensus_hardest:
        md.append(f"| {s['scene_id']} | {s['mean_accuracy']:.1%} | {s['std_accuracy']:.3f} | {s['task_desc']} |")

    md.append("\n## Consensus easiest scenes (highest mean accuracy across all 7 models)\n")
    md.append("| scene | mean acc | std | task |")
    md.append("|---|---|---|---|")
    for s in consensus_easiest:
        md.append(f"| {s['scene_id']} | {s['mean_accuracy']:.1%} | {s['std_accuracy']:.3f} | {s['task_desc']} |")

    md.append("\n## Scenes with most cross-model disagreement (highest std across 7 models)\n")
    md.append("| scene | mean acc | std | task |")
    md.append("|---|---|---|---|")
    for s in most_disagreement:
        md.append(f"| {s['scene_id']} | {s['mean_accuracy']:.1%} | {s['std_accuracy']:.3f} | {s['task_desc']} |")

    md.append("\n## Sub-task difficulty ranking (mean accuracy across 7 models, hardest first)\n")
    md.append("| sub-task | mean accuracy |")
    md.append("|---|---|")
    for t, a in subtask_ranking:
        md.append(f"| {t} | {a:.1%} |")

    md.append("\n## Cross-sub-task scene-difficulty correlation - v1 (naive, avg over 7 models)\n")
    md.append("Spearman correlation of per-scene accuracy between sub-task pairs, computed per "
               "model then averaged. Near-zero everywhere because per-model estimates are noisy "
               "and inconsistent in sign (see v2 below for the denoised version).\n")
    md.append("| | " + " | ".join(SUBTASKS) + " |")
    md.append("|---|" + "---|" * len(SUBTASKS))
    for i, t in enumerate(SUBTASKS):
        row = [f"{avg_subtask_corr[i, j]:.2f}" for j in range(len(SUBTASKS))]
        md.append(f"| {t} | " + " | ".join(row) + " |")

    md.append("\n## Cross-sub-task scene-difficulty correlation - v2 (denoised, 5 above-chance models)\n")
    md.append(f"Models: {', '.join(STRONG_MODELS)} (gemma4-e2b/e4b excluded, near-chance overall "
              f"accuracy). Per-scene accuracy averaged across these 5 models first, then "
              f"correlated once. 95% CI from bootstrap resampling over the 47 scenes (2000 reps). "
              f"See `subtask_correlation_heatmap.png`.\n")
    md.append("| | " + " | ".join(SUBTASKS) + " |")
    md.append("|---|" + "---|" * len(SUBTASKS))
    for i, t in enumerate(SUBTASKS):
        row = [f"{v2_5_matrix[i, j]:.2f}" for j in range(len(SUBTASKS))]
        md.append(f"| {t} | " + " | ".join(row) + " |")
    md.append("\n95% CI (lower, upper) per pair:\n")
    for i, ti in enumerate(SUBTASKS):
        for j in range(i + 1, len(SUBTASKS)):
            md.append(f"- {ti} vs {SUBTASKS[j]}: r={v2_5_matrix[i, j]:.2f}, "
                      f"CI [{v2_5_lo[i, j]:.2f}, {v2_5_hi[i, j]:.2f}]")

    md.append("\n## Cross-sub-task scene-difficulty correlation - v2, all 7 models (robustness check)\n")
    md.append("| | " + " | ".join(SUBTASKS) + " |")
    md.append("|---|" + "---|" * len(SUBTASKS))
    for i, t in enumerate(SUBTASKS):
        row = [f"{v2_7_matrix[i, j]:.2f}" for j in range(len(SUBTASKS))]
        md.append(f"| {t} | " + " | ".join(row) + " |")

    md_path = OUT_DIR / "SUMMARY.md"
    md_path.write_text("\n".join(md) + "\n")

    # ---- console summary -----------------------------------------------------
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {OUT_DIR / 'subtask_correlation_heatmap.png'}")
    print(f"\nAverage pairwise scene-ranking Spearman correlation across 7 models: {avg_pairwise_corr:.3f}")
    print("\nSub-task difficulty ranking (hardest -> easiest, mean accuracy across models):")
    for t, a in subtask_ranking:
        print(f"  {a:.1%}  {t}")
    print("\nCross-sub-task scene-difficulty correlation - v2, 5 strong models (with 95% CI):")
    for i, ti in enumerate(SUBTASKS):
        for j in range(i + 1, len(SUBTASKS)):
            print(f"  {ti} vs {SUBTASKS[j]}: r={v2_5_matrix[i, j]:.2f}  CI [{v2_5_lo[i, j]:.2f}, {v2_5_hi[i, j]:.2f}]")


if __name__ == "__main__":
    main()
