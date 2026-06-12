# Report Plan

Planning doc for the written report (3DV-style, CVPR template — same `project-proposal/`
format as the proposal, ~6–8 pages). This is a working document: Section 1 is the proposed
report structure, Section 2 is the narrative we want to tell, Section 3 is the figure/image
inventory, Section 4 is the outstanding TODO list before we can write the final version.

---

## 1. Proposed Report Structure

### 1. Introduction (~0.75 page)
- Motivation: VLMs proposed as zero-shot "robot supervisors" — scene understanding + semantic
  feedback without task-specific training (cite RT-2 / VLM-as-judge style works).
- Two concrete capability questions:
  1. **Failure-mode / state reasoning**: from a single image + stated goal, can a VLM judge
     task progress, current action phase, and success/failure?
  2. **Multi-view consistency**: do answers extracted from different camera angles of the
     *same* moment agree with each other?
- Contribution summary: benchmark built from Robo2VLM-1 (DROID+OXE), 7 models across 3
  families evaluated zero-shot, CoT ablation, scene-difficulty analysis, and a LoRA
  fine-tuning case study with a detailed gain-decomposition.

### 2. Related Work (~0.5 page)
- Robo2VLM-1 / VQA-from-robot-trajectories datasets.
- VLMs as evaluators/supervisors (success detection, reward modeling).
- Known VLM weaknesses in spatial/3D reasoning and CoT for visual tasks.

### 3. Dataset & Tasks (~1 page)
- Robo2VLM-1 origin (DROID ~50%, OXE ~50%), how questions/images are generated.
- **Action-phase benchmark** (4 sub-tasks): `action_phase_id`, `phase_success`, `progress`,
  `task_success`. Explain the **CnBD adversarial variant** (`special_image=="random_scene"`:
  image swapped for an unrelated scene → correct answer is "Cannot be determined").
- **Multi-view benchmark**: same scene/question from top-left/top-right/wrist cameras;
  consistency = do per-view answers agree.
- Dataset statistics table: # scenes, # questions per task, train/val/test split sizes
  (20/7/20 scenes, 2535/983/2706 questions for the LoRA split).
- Baselines defined: uniform random, yes/no random, majority-class, "always-CnBD-correct".

### 4. Method (~0.75 page)
- Models: Gemma4 (E2B/E4B/31B), Qwen3 (4B/8B/32B), InternVL3-14B. Zero-shot multiple-choice
  prompting (A/B/C/D/E letters), single forward pass.
- CoT variant: free-form reasoning before the final letter.
- LoRA fine-tuning setup on Qwen3-4B: scene-level split (no scene overlap train/test),
  hyperparameters, what `eval_results.jsonl` → `results.jsonl` conversion does
  (`scripts/convert_lora_eval_results.py`).

### 5. Results (~2.5–3 pages)
- **5.1 Action-phase / failure-mode detection** — main 7-model table (poster table, by_task
  accuracy + overall), vs. random/yes-no/majority baselines.
- **5.2 Multi-view consistency** — main 7-model table (TL↔TR, ≥1 Top↔Wrist, All-3-agree).
- **5.3 Chain-of-Thought ablation** — Gemma4-E4B (poster) + new Gemma4-E2B-CoT full run
  (28.8% overall, task_success collapses to 15.5%, *below* random 29.8%) as a second,
  independent data point. Qwen3-4B-CoT pending (see TODO #1).
- **5.4 Scene-difficulty / qualitative analysis** — accuracy spread across scenes (31pp for
  best model), easy-vs-hard scene examples for both tasks.
- **5.5 LoRA fine-tuning case study (Qwen3-4B)** — this is new content not in the poster and
  should be the most detailed results subsection:
  - Headline: 44.16% → 55.06% on identical 2706-question test set (+10.9pp, +295 correct).
  - Gain decomposition (see narrative §2.3): CnBD/OOD-detection chunk vs. phase_success
    coarse-judgment chunk vs. the IPI-collapse tradeoff.
  - Generalization check: CnBD recall by donor-scene origin (train-seen / val / unseen-test)
    — all comparable, so it's a generalizing rule, not memorization.

### 6. Discussion (~0.75 page)
- Why are absolute numbers so close to random? → object *state* (deformation, contact,
  orientation) is underrepresented in web-scale VLM training data vs. object *identity/position*.
- "Scale helps but isn't enough" across both tasks.
- LoRA framing: an 11pp headline gain that is mostly two coarse heuristics, one of which is a
  trade-off (not a pure win) — caution against over-reading fine-tuning numbers without
  decomposition.
- Multi-view: models are most self-consistent when uniformly right or uniformly wrong, least
  consistent in the "some views correct" middle ground — consistency tracks confidence, not
  necessarily correctness.

### 7. Limitations & Future Work (~0.5 page)
- Manual annotation → limited scene/question scale.
- In-distribution train/test split (same data distribution) — doesn't test sim-to-real or
  cross-dataset generalization.
- OXE episode traceability issue (can't always trace back to raw source).
- Future: automated annotation pipeline, LoRA on larger models (Qwen3-8B run pending),
  test the list-position-only hypothesis for `phase_success` (TODO #5).

### 8. Conclusion (~0.25 page)
- Restate the three findings: marginal zero-shot gains over random, CoT hurts, fine-tuning
  gains are real but mostly coarse/heuristic with a hidden trade-off.

---

## 2. Narrative — The Story We Want To Tell

The report should read as **"VLMs show weak but non-trivial robot-state reasoning
out of the box; chain-of-thought makes it worse; and when we fine-tune, most of the
apparent improvement is two coarse heuristics rather than genuine phase/temporal
reasoning — one of which is a generalizable skill, the other a trade-off dressed as a gain."**
This is a more defensible and more interesting claim than a flat "numbers went up" story,
and it directly continues the poster's "not built for robot scene understanding" theme with
hard evidence from the LoRA breakdown.

### 2.1 Zero-shot baseline story (poster content, carry over)
- 38.6% overall vs. 33.3% uniform random / 39.6% yes-no random — marginal.
- Scale helps within a family (Gemma E2B→E4B→31B; similar for Qwen) but even 31B/32B-class
  models plateau in the high-40s%.
- 31pp accuracy spread across 47 scenes — driven by scene visual properties (clutter,
  contrast, distinct objects), not task semantics.

### 2.2 CoT story (extend poster with new data)
- Poster: Gemma4-E4B CoT drops task_success to 20.9% (below random).
- New: Gemma4-E2B CoT, full 6224-question test set → overall 28.8% (below the 29.8% random
  baseline for this set!), task_success collapses to 15.5%. **Two independent models now
  show CoT hurting, and one shows it dragging overall accuracy below random** — strengthens
  the claim from "one model's anomaly" to "a pattern."
- Open: need an error-taxonomy pass on the CoT transcripts (failure-analysis pipeline step 2)
  to confirm/refute the poster's claim that "models describe scenes correctly but draw false
  conclusions" — right now that's an assertion, not yet evidenced with transcript examples.

### 2.3 LoRA story (new — the centerpiece)
This is the most novel material and should get the most narrative care. Structure as:

1. **The headline number**: 44.16% → 55.06% (+10.9pp), apples-to-apples on the same
   2706-question / 20-scene test set.
2. **The catch — decompose it**:
   - **Chunk A (~44% of the gain, 130/2706 questions, 6.5% of test set)**: CnBD/random-scene
     cells in `action_phase_id` and `task_success`. BASE ~0–15% → LoRA 78–88%. The model
     learned "image scene ≠ goal-text scene → answer Cannot be determined."
   - **Chunk B (~55% of the gain, 161 questions)**: `phase_success`, specifically MIDDLE
     positions (image_step == label_step, i.e. neither first-nor-last phase). Coarse
     before/at/after judgment improves 36.9% → 49.9%.
   - Same-scene "normal" (non-CnBD) accuracy on `action_phase_id` (+2.4pp) and `progress`
     (−1.3pp) is **flat within noise** — no improvement on fine-grained exact-phase-matching
     or 2-image pairwise temporal direction.
3. **Chunk A is real, not memorization**: CnBD recall split by donor-scene origin
   (train-seen 0.92/0.65, val 1.0/1.0, unseen-test-other 0.82/0.82 for action_phase_id /
   task_success respectively) — unseen-donor recall is *not* lower, so the
   "scene mismatch → CnBD" rule generalizes to novel scene pairings. **Frame this
   positively** — it's a real, transferable skill, just not "phase reasoning."
4. **Chunk B is a trade-off, not a clean win**: the phase_success MIDDLE improvement is
   bought by the model nearly abandoning "Is currently performing it" (IPI) as an answer —
   recall 35.8% → 6.6%, raw IPI predictions 383 → 43 across the whole test set. Sanity-checked
   under a lenient "Yes also counts for IPI" scoring — LoRA is *still* worse (58.0% → 48.9%).
   Of LoRA's 274 IPI ground-truth cases, 140 get predicted "No". **Frame this as a
   regression that happens to net positive under this dataset's label distribution** — a
   model deployed as a "is the robot doing X right now?" supervisor would get *worse* at
   exactly the borderline/transitional moments that matter most for failure detection.
5. **`progress` didn't pick up the same CnBD cue** despite more training exposure (110 vs.
   130 examples) — hypothesis: in `progress` only one of the two side-by-side images is the
   random-scene tile, making the cue less salient (single full-image swap vs. half-image
   swap). Worth a sentence — shows the OOD-detection skill is salience-dependent, not a
   universal "any weirdness → CnBD" reflex.

**One-line takeaway for abstract/conclusion**: *"Fine-tuning lifts accuracy by 11pp, but
~80% of that gain decomposes into two coarse heuristics — a generalizable but
phase-agnostic OOD-scene detector, and a bias shift away from 'currently in progress'
that is a regression for failure-detection use cases — rather than improved phase or
temporal reasoning."*

### 2.4 Multi-view story (new analysis available, not yet in poster narrative)
`analyze_multiview_consistency.py` (qwen3-32b run, 98266) gives a `by_correctness`
breakdown that's a nice addition:
- `all_correct` groups: all-3-agree rate 0.72
- `none_correct` groups: all-3-agree rate 0.66
- `some_correct` groups (model gets it right from *some* viewpoints but not others):
  all-3-agree rate **0.34** — the lowest by a wide margin.

**Narrative**: the model is most self-consistent when it's uniformly right or uniformly
wrong, and least consistent exactly in the ambiguous middle — i.e., view-consistency tracks
something like "confidence/clarity of the scene" rather than correctness per se. This is a
nice complement to the scene-difficulty finding: hard scenes don't just lower accuracy, they
also destabilize the model's answer across viewpoints.

---

## 3. Figures / Images

### 3.1 Already available in `pics-for-poster/` (reuse directly)
| File | Use |
|---|---|
| `action_phase_id1_..._align_the_rope_..._q14.jpg` | Action Phase Identification example |
| `action_phase_id700_..._close_the_tap_..._q3.jpg` | Task Success example |
| `action_phase_id524_..._scrunch_up_the_towel_..._q10.jpg` + `_q15.jpg` | Progress Detection pair |
| `action_phase_id355_..._put_the_green_object_..._q13.jpg` | Task Success (Adversarial / CnBD) example |
| `action_phase_droid_put_the_green_marker_..._q10.jpg` | "Easy scene" example |
| `action_phase_droid_stack_the_two_paper_cups_..._q10.jpg` | "Difficult scene" example |
| `multiview_id1/355/524/700_..._top_right.jpg` | Multi-view example crops |

These cover the intro/method figures fine. For Results, we want **new** figures (below).

### 3.2 New figures needed
1. **LoRA gain-decomposition bar chart**: grouped bars per sub-task (action_phase_id,
   phase_success, progress, task_success), each with BASE vs LoRA, *and* a third
   "normal-only" (CnBD-stripped) bar pair for action_phase_id/task_success/progress, plus a
   phase_success-MIDDLE-only bar pair. Visualizes "headline gain vs. real gain" directly —
   probably the single most important new figure in the report.
2. **CnBD example with BASE vs LoRA prediction**: take one `random_scene` example (image
   doesn't match goal text), show BASE answer (wrong, e.g. "Yes") vs LoRA answer (correct,
   "Cannot be determined"). Can reuse `action_phase_id355` or pick a fresh one via
   `results.jsonl` diff.
3. **IPI-collapse example**: a `phase_success` MIDDLE-position question where ground truth =
   "Is currently performing it", BASE answers IPI (correct) but LoRA answers "No" (incorrect)
   — illustrates the trade-off concretely.
4. **Multi-view consistency: most-consistent vs most-inconsistent scene** (qwen3-32b,
   `summary_multiview.json`):
   - Most inconsistent: `droid_remove_the_black_marker_from_the_black_cup_1683` (all_agree
     0.27), `droid_put_the_green_object_inside_the_bowl_2798` (0.06!), `droid_pick_the_orange_ball_and_put_it_in_the_pot_on_the_stove_2529` (0.31).
   - Most consistent: `droid_pick_the_orange_object_and_put_it_in_the_bowl_12206` (0.83),
     `droid_move_the_white_mug_backwards_3759` (0.81).
   - Note: `droid_put_the_green_object_inside_the_bowl_2798` and
     `droid_pick_the_orange_object_and_put_it_in_the_bowl_12206` are *both already in*
     `pics-for-poster/` (as `action_phase_id355` and via action-phase scenes) — convenient,
     may already have multi-view tiles for them.
5. **"some_correct" ambiguous-scene example**: 3 camera views of one scene where the model's
   answer differs across views — illustrates the by_correctness finding (§2.4).
6. **Scene-difficulty scatter/bar**: accuracy per scene (47 scenes) for the best zero-shot
   model, sorted — visualizes the 31pp spread more rigorously than the two cherry-picked
   easy/hard examples. `scripts/rank_scenes.py` output feeds this directly.

---

## 4. Outstanding TODOs Before Writing

1. **Qwen3-4B CoT run incomplete**: `outputs/slurm98361_action_phase_qwen3-4b_default_cot/`
   has only 768/6224 result rows (and `slurm98356_...` only 46) — no `summary.json`. Check
   `slurm-98361.err` / `slurm-98356.err` for why it stopped (matches the "runs stop out of
   nowhere" issue noted in the last commit), rerun to completion, then run
   `save_summary` (or `scripts/convert_lora_eval_results.py`-style script) to get a
   comparable CoT table entry for Qwen3-4B.
2. **Qwen3-8B multiview run incomplete**: `slurm97947_multiview_qwen3-8b_default` only has
   899/1367 source groups (~66%). Rerun for a clean number in the multi-view table.
3. **LoRA Qwen3-8B**: `train_lora_qwen3_8b.py` + `run_slurm-train-lora-qwen3-8b.sh` exist but
   `outputs/lora-qwen3-vl-8b/` is empty. Running this gives a second LoRA data point — does
   the CnBD-generalization / IPI-collapse pattern hold at 8B too, or is it 4B-specific?
   (Nice-to-have, not blocking — flag as stretch goal given CoT/multiview reruns above are
   higher priority for completing the *existing* tables.)
4. **Failure-analysis pipeline (agreed steps, not yet executed for the report)**:
   - Step 2 — error taxonomy on raw CoT responses (format failures / confident-wrong /
     hedging / hallucination) to back up the "describes correctly, concludes wrongly" claim.
   - Step 4 — cross-model disagreement analysis (join on `entry_id`): e.g. Qwen3-4B BASE vs
     LoRA, or Gemma4-31B vs Qwen3-32B, into 4 buckets (both correct / both wrong / A-only /
     B-only) — could surface complementary strengths.
   - Step 5 — per-question-type-group hypotheses (spatial / goal / interaction reasoning),
     stated explicitly with counter-arguments per the agreed documentation standard
     ([[project_failure_analysis_strategy]]).
5. **Untested hypothesis from LoRA analysis**: does `phase_success` rely only on
   `label_phase`'s *position* in the ordered phase list (text-derivable, no image needed) for
   variant-B (list-present) prompts? Test: swap `label_phase` text for the same-position
   phase from a different scene, same image, run through LoRA model — if answer is
   unchanged, list-position-only is confirmed. Would directly explain the
   phase_success-MIDDLE gain without invoking "improved temporal grounding."
6. **Build the new figures** (§3.2) — needs a small script to diff BASE vs LoRA
   `results.jsonl` on `entry_id` for items 2/3, and to render item 1's grouped bar chart
   (matplotlib) from `summary.json` + a CnBD-stripped re-aggregation (the breakdown numbers
   already exist in the LoRA memory — just need a script that recomputes "normal-only" and
   "phase_success-MIDDLE-only" subsets and saves a chart).
7. **Scope decision**: with LoRA as a new section, the report may be tight on the ~6-8 page
   budget. Options: (a) keep all 7 zero-shot models in the main tables but trim the
   discussion of weaker models, or (b) feature only the 3-4 most relevant models (e.g.
   Gemma4-31B, Qwen3-4B, Qwen3-4B+LoRA, InternVL3-14B) in-depth and move the full 7-model
   table to an appendix/supplementary.
8. **Confirm report authorship/audience**: per memory, teammates work on the poster only —
   confirm whether this written report is solo, and whether teammates need a
   simplified/different narrative summary for poster updates (poster currently doesn't
   mention LoRA at all — may be worth a 1-2 line LoRA teaser on the poster too, separate
   decision from the report).
