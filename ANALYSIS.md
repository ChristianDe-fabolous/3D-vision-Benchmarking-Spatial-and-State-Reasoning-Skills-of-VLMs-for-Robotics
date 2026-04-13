# VLM Failure Analysis

This document defines the exact methodology for understanding failure cases of Qwen 3B and 7B on the Robo2VLM-1 benchmark. Every step is stated before results are examined. Hypotheses are explicit. Counter-arguments are addressed. No vague conclusions.

---

## Models Under Analysis

| Model | Identifier | VRAM | Quantization |
|-------|-----------|------|-------------|
| Qwen2.5-VL-3B-Instruct | `qwen-3b` | ~6GB | bf16 |
| Qwen2.5-VL-7B-Instruct | `qwen-7b-int8` | ~8GB | int8 |

TODO: check for model specific archteicture and vision understanding
TODO2: check for literature of understanding failure
Qwen2.5-7B: 
Qwen2.5-3B: 


---

## Question Type Taxonomy

Questions are classified into three groups (following the paper):

**Spatial Reasoning**
- `gripper_state_RS` — is the gripper open or closed?
- `obstacle_detection_OS` — is the path to the target clear?
- `relative_direction_SR` — direction of object relative to end effector
- `relative_depth_SU` — which marked point is closest/farthest from camera?
- `cross_view_correspondence_MV` — same 3D point across two camera views

**Goal-Conditioned Reasoning**
- `grasp_stability_TS-G` — is the grasp stable?
- `task_success_TS-S` — has the robot completed the task?
- `goal_configuration_TS-GL` — which image shows the goal state?

**Interaction Reasoning**
- `grasp_phase_current_AU` — which grasp phase is shown?
- `temporal_sequence_AU` — correct ordering of action phases
- `grasp_phase_next_IP` — what is the next action phase?
- `action_direction_IP` — which arrow shows the next movement direction?
- `trajectory_understanding_TU` — which instruction matches the trajectory?

---

Idea 1: Controlled visual ablation                                            
                                                            
  Run the same question with three input variants:
  1. Original image — normal evaluation
  2. Blank/noise image — model gets no visual information                       
  3. Cropped/zoomed image — model gets only the relevant region
                                                                                
  If accuracy on (1) ≈ accuracy on (2): model is not using the image at all     
  (Layer 1).                                  
  If accuracy on (3) > accuracy on (1): model can perceive the element but fails
   to locate it in the full image (spatial attention failure within Layer 2).
                                                                                
  Problem: requires modifying images per question, which needs to know which
  region is relevant. Feasible for simple questions like gripper_state_RS       
  (gripper region is predictable), harder for cross_view_correspondence_MV.
                                                                                
  ---                                                       
  Idea 2: CoT scene description extraction                                      
                                                            
  Run CoT prompt. Extract the factual claims the model makes about the image
  before it reasons. Compare those claims to ground truth labels.
                                          
  For gripper_state_RS: model should state "the gripper is open" or "the gripper
   is closed." Ground truth is the correct answer. If the model states the wrong
   state → Layer 2. If it states the right state but answers wrong → Layer 3.
                                                                                
  This requires no image modification — just careful reading of the reasoning
  chain. Can be partially automated: extract sentences containing "gripper is   
  open/closed" and check against ground truth label.        
                                                                                
  This is the most practical approach to start with.
                                                                                
  ---                                                       
  Idea 3: Forced binary perception questions
                                              
  For questions where perception of a specific element determines the answer,
  construct a direct probe: instead of the original question, ask the model only
   about the visual element.                                                    
  
  Example — for grasp_stability_TS-G:                                           
  - Original: "Is the robot's grasp of the [object] stable?"
  - Probe: "Is the robot's gripper currently open or closed?"                   
                                                                                
  If the model answers the probe correctly but the original wrongly → Layer 3
  (reasoning failure). If it answers the probe wrongly → Layer 2 (perception    
  failure).                                                 
                                                                                
  Problem: requires writing probe questions per question type. Feasible since we
   only have 13 types.                                                          
                                                                                
  ---                                                                           
  Idea 4: Consistency check across related questions
                                                                                
  The dataset has multiple questions per scene. Some questions about the same
  scene implicitly require perceiving the same visual element.
                                                                                
  Example: gripper_state_RS ("is the gripper open?") and grasp_stability_TS-G
  ("is the grasp stable?") both require perceiving the gripper state. If the    
  model answers gripper_state_RS correctly (gripper is open) but answers
  grasp_stability_TS-G as "stable" → it perceived correctly but reasoned wrong  
  (Layer 3). If it answers gripper_state_RS wrongly → Layer 2 already caught at
  the simpler question.                                                         
                                                            
  This requires no extra runs — cross-reference existing results within the same
   scene and across related question types.

  ---                                                                           
  Idea 5: Attention map analysis (if accessible)
                                                                                
  Some transformer models expose attention weights. High attention on irrelevant
   image regions when answering a spatial question = perception failure.
  Qwen2.5-VL uses a vision encoder (ViT-based) + cross-attention to the language
   model. Attention maps are accessible via hooks on the model.
                                                                                
  Problem: attention maps are noisy and hard to interpret rigorously. Better as
  a qualitative illustration of a finding already established quantitatively.  


## Ideas and brain storming:

**What do we want to understand?** - Failure modes of VLMs
Means: We are interested in cases where the model answes wrong, and only for comparison we regard correct answers.

We have to understand how good the model can perceive the image. Maybe that expalines the difference between different model families. Then we want to check perception in general. 


How to compare scenes? 

* Did the model perceive the image correctly?
It maybe misidentifies what's in it? 
Cot could reveal that, but could be hallucination. We could check for reasoning that is dissimilar to the scene? Human or llm as a judge

### Experiments that could lead to answers:
* What questions, do bigger models get right compared to small ones. 3B vs 7B vs 30B, each one 3 times? 
Hope: we understand questions that geniunly need more pattern matching or understanding. Probably provided in a bigger model. 

Follow up question: How can we compare the questions across the models? 
Maybe we can check for certain charateristics in questions that differntiate between other questions?, also scenes?

* Did the model actually look at the image: answers correctly without using image? Just input other image or no image as ablation.


* Compute output of wrongly answered question: backpropagate how to increase "attention" to relevant part of answer and then check if the model performs better. That could mean that it is more an perception problem. 

* Revomving relevant part of the image can defiently help to understand how well perception is working.

* Does improved perception strongly improve the accrucay? Only possible for quesitons where relationships are not implicit. 

* Why is OS: Object State (object reachability/manipulability), and TS-GL: Task State-goal (goal configuration understanding), is much better for CoT. Could this help to understand implicit relationships? Espeically the former!

* What is enabled through CoT



## Notes 
OS has biggest improvement from CoT, followed by TS-S, TS-GL, IP, AU (no ordering). 
IP is the only one, where bigger is really much better - difference in questions?





## Step 1: Quantitative Breakdown

**What we measure:**
- Overall accuracy (3B vs 7B)
- Accuracy per question type
- Accuracy per question group (spatial / goal / interaction)
- Unparseable response rate (model did not output a valid letter)
- Answer distribution bias (does the model systematically favor certain labels?)
- Scene-level accuracy distribution (are failures concentrated in specific scenes?)

**Why before anything else:** quantitative breakdown determines where to focus qualitative effort. Examining raw responses without knowing which question types fail most wastes time.

---

## Step 4: Chain-of-Thought Analysis

Both models are re-run with the `paper_cot` prompt, which requires the model to reason step by step before stating the final letter. Wrong answers are examined to identify where the reasoning chain breaks.

```bash
MODEL=qwen-7b-int8 PROMPT=paper_cot sbatch scripts/run_slurm.sh
MODEL=qwen-3b      PROMPT=paper_cot sbatch scripts/run_slurm.sh
```

**What to look for in wrong CoT responses:**
- Correct reasoning chain but wrong final letter (output format failure)
- Reasoning chain that ignores the image entirely (language prior dominates)
- Reasoning chain that misidentifies the scene (visual grounding failure)
- Reasoning chain that is internally inconsistent (model contradicts itself)

**Results:** *(to be filled in)*

---

## Step 5: Cross-Model Disagreement Analysis

3B and 7B results are joined on `entry_id`. Every question falls into one of four buckets:

| Bucket | Meaning |
|--------|---------|
| Both correct | Question is easy for current VLMs |
| Both wrong | Genuine hard case — primary failure signal |
| 7B correct, 3B wrong | Scale helps |
| 3B correct, 7B wrong | Scale regression — worth investigating |

**Primary focus:** the "both wrong" bucket. These are questions where scale provides no benefit and represent the actual capability ceiling of this model family on this task.

**Secondary focus:** the "3B correct, 7B wrong" bucket. Scale regression is unexpected and may reveal overfitting to instruction-following at the expense of direct visual reasoning.

**Script:** `scripts/compare_models.py` *(to be written)*

**Results:** *(to be filled in)*

---

## Step 6: Conclusions and Open Questions

*(to be filled in after steps 1–5 are complete)*

Each conclusion must cite specific question types, entry counts, and accuracy numbers. No conclusion of the form "the model struggles with spatial reasoning" without stating which types, what accuracy, and compared to what baseline.

---

## Proposed Experiments

### Experiment 1: Text-only baseline (no image)

Run both models on all questions with a blank image or no image. Compare per-type accuracy to the with-image run.

**What it tells us:**
- Type where text-only ≈ with-image → model not using the image (Layer 1 dominance)
- Type where image improves → model extracts something visual
- Type where image *hurts* vs text-only → image actively confuses the model

**Key question for TS-GL:** does the below-random score (6.72%) persist without the image, or does text-only give ~25% (random)? If text-only gives random but image gives below-random → the image is triggering a wrong heuristic. If text-only also gives below-random → the wrong heuristic comes from the question text itself.

**Implementation:** pass a 1×1 white pixel image or omit image tokens. One inference pass per question — same cost as a normal run.

---

### Experiment 2: Answer choice position shuffling

Re-run all questions with answer choices in a shuffled order (fixed random seed for reproducibility). Compare per-question accuracy to original.

**What it tells us:**
- If accuracy changes significantly for a type → positional bias (model always picks A, or always picks the last option)
- If accuracy is stable → model responds to content not position
- For TS-GL specifically: if shuffling moves accuracy from 6% toward 25% → the model was always picking the same letter, and the correct answer was consistently in a different position

**Implementation:** shuffle `sample.choices` with a fixed seed before building the prompt. Map predicted letter back to original choice order for scoring. Requires a modified inference pass — no model changes.

---

### Comparing CoT vs non-CoT: transition matrix

Aggregate accuracy comparison hides the structure. Per-question transition matrix reveals it:

| | CoT correct | CoT wrong |
|---|---|---|
| **non-CoT correct** | stable correct — no change | CoT regression — CoT introduced wrong reasoning |
| **non-CoT wrong** | CoT helped — reasoning unlocked answer | stable wrong — CoT made no difference |

For TS-GL: CoT likely moves most entries from "stable wrong (same letter)" to "stable wrong (different letters)" — bias broken, no understanding gained. That is itself a finding: CoT removes a heuristic without providing a replacement.

For OS and TS-S where CoT helps most: look at the "CoT helped" entries specifically. What changed in the reasoning chain? Was it the scene description step or the inference step?

---

## Perception Analysis Methods

### What perception analysis asks

Not "did the model get the answer right?" but "did the model correctly extract the specific visual feature that the question depends on?" These are different. A model can get the right answer without perceiving correctly (language prior), and can perceive correctly but still answer wrong (reasoning failure).

The goal is to localize failures at the visual extraction step.

---

### Method 1: Free-form image captioning as perception probe

Before the multiple-choice question, ask the model: "Describe what you see in this image in detail." No answer choices, no task framing. Read the description and check:
- Does it mention the relevant visual element?
- Is the description factually correct about that element?
- For wrong answers: what did the model actually perceive?

**Why:** the multiple-choice format may suppress what the model perceived — it forces a letter output. Free-form description reveals the raw visual representation. A model that describes "the gripper is open" correctly but then answers "the grasp is stable" has a reasoning failure, not a perception failure.

**Cost:** one extra inference pass per question. Interpretable without image modification.

**Most useful for:** `gripper_state_RS`, `obstacle_detection_OS`, `action_direction_IP` — types where the relevant element is a discrete, nameable thing that would naturally appear in a description.

**Less useful for:** `relative_depth_SU`, `cross_view_correspondence_MV` — types where the relevant feature is a precise spatial relationship that free-form description rarely captures.

---

### Method 2: GradCAM / gradient saliency on vision encoder

Backpropagate the gradient of the predicted answer's probability through the vision encoder to the input image. High-gradient regions = image areas the model relied on for its prediction.

**What it shows:** whether the model was attending to the right region. For `gripper_state_RS` — did it attend to the gripper region? For `action_direction_IP` — did it attend to the arrows? For `cross_view_correspondence_MV` — did it attend to the red dot?

**Implementation:** register a backward hook on the final layer of Qwen2.5-VL's ViT vision encoder. Run inference, compute loss on predicted token, backpropagate. Map gradients back to image spatial dimensions. Overlay as heatmap on original image.

**Limitation:** attention maps and gradients are noisy and do not directly tell you whether the model understood what it attended to — only where it looked. Use quantitatively to check if high-gradient region overlaps with the known relevant region. Use qualitatively as illustration.

**Most useful for:** `gripper_state_RS`, `action_direction_IP`, `cross_view_correspondence_MV`, `relative_depth_SU` — types with a specific, localizable relevant region.

**Less useful for:** `task_success_TS-S`, `goal_configuration_TS-GL`, `temporal_sequence_AU` — holistic scene judgments with no single localized relevant region.

---

### Method 3: Region masking / occlusion sensitivity

Systematically mask regions of the image (e.g. divide into a 3×3 grid, mask each cell one at a time) and record how accuracy changes per masked region. The region whose masking most degrades accuracy is the one the model relies on.

**More targeted version:** for question types with a known relevant region, mask specifically that region. Compare accuracy with and without the region visible.

- `gripper_state_RS` → mask the end-effector region
- `action_direction_IP` → mask the colored arrow region
- `cross_view_correspondence_MV` → mask the red dot in the left image; mask the correct correspondence point in the right image

**What it shows:**
- Masking relevant region hurts accuracy → model was using that region
- Masking relevant region doesn't change accuracy → model was not using that region (Layer 1 or attending to wrong region)
- Masking irrelevant region hurts accuracy → model was attending to the wrong region

**Most useful for:** `gripper_state_RS`, `action_direction_IP` — types where the relevant region is tightly localizable and small.

**Least useful for:** `goal_configuration_TS-GL` — the relevant region is the entire image (or multiple images).

---

### Method 4: Probing the vision encoder representations

Extract intermediate representations from Qwen2.5-VL's vision encoder (patch embeddings before projection into the language model). Train a linear probe on these representations to predict the ground truth answer for specific question types.

**What it shows:** if a linear probe can predict `gripper_state_RS` answers from vision encoder patch embeddings → the visual information IS encoded in the representation. The failure must be in how the language model uses that representation (reasoning or integration failure). If the probe fails → the visual encoder does not capture the feature at all (deep perception failure).

**Why this matters:** separates encoder failure from integration failure. A model where the encoder represents the gripper state correctly but the language model ignores it has a different failure than one where the encoder never encoded the state.

**Cost:** requires extracting activations across many examples and training a small classifier. Moderate implementation effort.

**Most useful for:** `gripper_state_RS` (binary, clean ground truth), `relative_depth_SU` (3-class), `action_direction_IP` (directional).

---

### Question type × method relevance matrix

| Question type | Captioning | GradCAM | Masking | Probing |
|--------------|-----------|---------|---------|---------|
| `gripper_state_RS` | ★★★ | ★★★ | ★★★ | ★★★ |
| `obstacle_detection_OS` | ★★★ | ★★ | ★★ | ★ |
| `relative_direction_SR` | ★★ | ★★ | ★★ | ★★ |
| `relative_depth_SU` | ★ | ★★ | ★★ | ★★★ |
| `cross_view_correspondence_MV` | ★ | ★★★ | ★★★ | ★★ |
| `grasp_stability_TS-G` | ★★ | ★★ | ★★ | ★★ |
| `task_success_TS-S` | ★★ | ★ | ★ | ★ |
| `goal_configuration_TS-GL` | ★★ | ★ | ★ | ★ |
| `grasp_phase_current_AU` | ★★ | ★★ | ★★ | ★★ |
| `action_direction_IP` | ★★★ | ★★★ | ★★★ | ★★ |
| `trajectory_understanding_TU` | ★★ | ★ | ★ | ★ |

★★★ = primary method for this type / ★★ = informative / ★ = limited value

**Priority for perception analysis:** `gripper_state_RS` and `action_direction_IP` first — both have localizable relevant regions, discrete ground truth, and are not solvable from language prior alone. `cross_view_correspondence_MV` second — most challenging perceptually and architecturally.
