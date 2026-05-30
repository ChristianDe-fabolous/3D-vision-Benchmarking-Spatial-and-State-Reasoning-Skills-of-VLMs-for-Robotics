import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = Path(os.environ.get("VLM_OUTPUT_DIR", ROOT_DIR / "outputs"))
LOG_DIR = Path(os.environ.get("VLM_LOG_DIR", ROOT_DIR / "logs"))

# Known invalid dataset entries — skip during evaluation
# INVALID_ENTRIES: set[int] = {5, 35, 37, 67, 69}
INVALID_ENTRIES: set[int] = {}

# Task identifiers
TASK_FAILURE_MODE  = "failure_mode"
TASK_MULTIVIEW     = "multiview"
TASK_ACTION_PHASE  = "action_phase"

# Prompt identifiers
PROMPT_DEFAULT = "default"
PROMPT_TEST = "test"
PROMPT_PAPER = "paper"
PROMPT_PAPER_COT = "paper_cot"

# Question type buckets, nested by task.
# Maps task -> (type name -> list of substrings, case-insensitive).
# A question can match multiple types — all matches are recorded.
# See scripts/list_question_type_overlaps.py for overlap analysis.
# QUESTION_TYPES: dict[str, dict[str, list[str]]] = {
#     TASK_FAILURE_MODE: {
#         # Did the robot complete the overall task goal?
#         "task_success": ["successfully completed the task"],
#         # What grasp phase is currently shown in the image?
#         "grasp_phase_current": ["which phase of the grasp action is shown"],
#         # What comes next after the current action phase?
#         "grasp_phase_next": ["what will be the robot's next action phase"],
#         # Is the robot's gripper currently open or closed?
#         "gripper_state": ["is the robot's gripper open"],
#         # Is the robot holding the object securely?
#         "grasp_stability": ["is the robot's grasp of the", "stable"],
#         # Is the path to the target object clear?
#         "obstacle_detection": ["is there any obstacle blocking the robot from reaching"],
#     },
#     TASK_MULTIVIEW: {
#         # Which point in the second view corresponds to the same 3D location?
#         "cross_view_correspondence": ["corresponding to the same 3d location", "same 3d location"],
#         # Which marked point is closest/farthest from the camera (depth from single view)?
#         "relative_depth": ["which colored point is closest", "which colored point is farthest",
#                            "closest to the camera", "farthest from the camera"],
#     },
# }

# Question type groups (paper taxonomy)
GROUP_SPATIAL      = "spatial_reasoning"
GROUP_GOAL         = "goal_reasoning"
GROUP_INTERACTION  = "interaction_reasoning"

# Maps each question type key → paper group.
# Keys match the keys in QUESTION_TYPE_TEMPLATES below.
QUESTION_TYPE_GROUPS: dict[str, str] = {
    # Spatial Reasoning
    "gripper_state_RS":              GROUP_SPATIAL,
    "obstacle_detection_OS":         GROUP_SPATIAL,
    "relative_direction_SR":         GROUP_SPATIAL,
    "relative_depth_SU":             GROUP_SPATIAL,
    "cross_view_correspondence_MV":  GROUP_SPATIAL,
    # Goal-Conditioned Reasoning
    "grasp_stability_TS-G":          GROUP_GOAL,
    "task_success_TS-S":             GROUP_GOAL,
    "goal_configuration_TS-GL":      GROUP_GOAL,
    # Interaction Reasoning
    "grasp_phase_current_AU":        GROUP_INTERACTION,
    "temporal_sequence_AU":          GROUP_INTERACTION,
    "grasp_phase_next_IP":           GROUP_INTERACTION,
    "action_direction_IP":           GROUP_INTERACTION,
    "trajectory_understanding_TU":   GROUP_INTERACTION,
}

# -----------------------------------------------------------------------------------------------------------------------------------------

# From paper:
# Category Abbreviations: Spatial Reasoning: RS: Robot State (gripper/arm position estimation), OS: Object State (object reachability/manipula-
# bility), SR: Spatial Relationship (relative positioning between robot and objects), SU: Scene Understanding (spatial layout comprehension),
# MV: Multiple View (cross-view correspondence). Goal-Conditioned Reasoning: TS-G: Task State-grasp (grasp stability assessment), TS-S:
# Task State-success (task completion status), TS-GL: Task State-goal (goal configuration understanding), Interaction Reasoning: AU: Action
# Understanding (robot’s current action phase), IP: Interaction Phase (prediction of next robot action), TU: Trajectory Understanding (overall task
# interpretation)

# -----------------------------------------------------------------------------------------------------------------------------------------


# Regex-based question type templates derived directly from the robo2vlm vqa.py
# generator functions. Variable slots (object names, camera names, step indices,
# language instructions) are replaced with regex wildcards. Use re.search with
# re.IGNORECASE.
# Keys use the format <descriptive_name>_<paper_abbreviation> so that results
# are self-documenting and directly comparable to the paper's Table 1.
QUESTION_TYPE_TEMPLATES: dict[str, dict[str, list[str]]] = {
    TASK_FAILURE_MODE: {
        # S1 — vqa_robot_gripper_open — Spatial / Robot State
        "gripper_state_RS":            [r"is the robot's gripper open\?"],
        # S3 — vqa_object_reachable — Spatial / Object State
        "obstacle_detection_OS":       [r"is there any obstacle blocking the robot from reaching .+\?"],
        # I1 — vqa_task_success_state — Goal / Task State-success
        "task_success_TS-S":           [r"the robot is to .+\. has the robot successfully completed the task\?"],
        # I3 — vqa_goal_configuration — Goal / Task State-goal
        "goal_configuration_TS-GL":    [r"the robot's task is to .+\. which configuration shows the goal state that the robot should achieve\?"],
        # I4 — vqa_action_understanding — Interaction / Action Understanding
        "grasp_phase_current_AU":      [r"the robot is tasked to .+\. the robot is interacting with the .+\. which phase of the grasp action is shown in the image\?"],
        # I5 — vqa_temporal_sequence — Interaction / Action Understanding
        "temporal_sequence_AU":        [r"for the task '.+', what is the correct sequence of action phases shown in the images from left to right\?",
                                        r"what task is the robot performing in this sequence of images\?"],
        # I4 — vqa_next_action — Interaction / Interaction Phase
        "grasp_phase_next_IP":         [r"the robot is tasked to .+\. after .+, what will be the robot's next action phase\?"],
        # I6 — vqa_trajectory_understanding — Interaction / Trajectory Understanding
        "trajectory_understanding_TU": [r"which language instruction best describes the robot's trajectory shown in the image\?"],
        # S2 — vqa_relative_direction — Spatial / Spatial Relationship
        "relative_direction_SR":       [r"which direction is the .+ relative to the robot's end effector\?"],
        # I2 — vqa_grasp_stability — Goal / Task State-grasp
        "grasp_stability_TS-G":        [r"is the robot's grasp of the .+ stable\?"],
        # I4 — vqa_action_direction — Interaction / Interaction Phase
        "action_direction_IP":         [r"which colored arrow correctly shows the direction the robot will move next\?"],
    },
    TASK_MULTIVIEW: {
        # S8 — vqa_multi_view_correspondence — Spatial / Multiple View
        "cross_view_correspondence_MV": [r"in the left image \(.+ camera\), a red dot is marked\. which point is the closest point in the right image \(.+ camera\) corresponding to the same 3d location\?"],
        # S6 — vqa_relative_depth — Spatial / Scene Understanding
        "relative_depth_SU":            [r"in the image from .+, which colored point is (closest|farthest) (to|from) the camera\?"],
    },
}

# Named-group extraction patterns — same slots as QUESTION_TYPE_TEMPLATES but
# with (?P<name>...) groups so variable parts can be pulled out of each question.
# Keys match QUESTION_TYPE_TEMPLATES. Only types with meaningful variables are listed.
QUESTION_TYPE_EXTRACT: dict[str, list[str]] = {
    # failure_mode
    "obstacle_detection_OS":       [r"is there any obstacle blocking the robot from reaching (?P<object>.+)\?"],
    "relative_direction_SR":       [r"in the image from (?P<camera>.+) at step (?P<step>\d+), which direction is the (?P<object>.+) relative to the robot's end effector\?"],
    "task_success_TS-S":           [r"the robot is to (?P<task>.+)\. has the robot successfully completed the task\?"],
    "grasp_stability_TS-G":        [r"is the robot's grasp of the (?P<object>.+) stable\?"],
    "goal_configuration_TS-GL":    [r"the robot's task is to (?P<task>.+)\. which configuration shows the goal state that the robot should achieve\?"],
    "grasp_phase_current_AU":      [r"the robot is tasked to (?P<task>.+)\. the robot is interacting with the (?P<object>.+)\. which phase of the grasp action is shown in the image\?"],
    "temporal_sequence_AU":        [r"for the task '(?P<task>.+)', what is the correct sequence of action phases shown in the images from left to right\?"],
    "grasp_phase_next_IP":         [r"the robot is tasked to (?P<task>.+)\. after (?P<current_phase>.+), what will be the robot's next action phase\?"],
    "action_direction_IP":         [r"the robot task is to (?P<task>.+)\. which colored arrow correctly shows the direction the robot will move next\?"],
    # multiview
    "cross_view_correspondence_MV": [r"in the left image \((?P<left_camera>.+) camera\), a red dot is marked\. which point is the closest point in the right image \((?P<right_camera>.+) camera\) corresponding to the same 3d location\?"],
    "relative_depth_SU":            [r"in the image from (?P<camera>.+), which colored point is (?P<direction>closest|farthest) (?:to|from) the camera\?"],
}

# Model identifiers — Qwen
MODEL_QWEN_3B       = "qwen-3b"
MODEL_QWEN_7B       = "qwen-7b"
MODEL_QWEN_7B_INT8  = "qwen-7b-int8"
MODEL_QWEN3_2B      = "qwen3-2b"
MODEL_QWEN_32B_INT8 = "qwen-32b-int8"  # Qwen2.5-VL-32B, int8 ~32GB VRAM — requires gb10 node

QWEN_MODEL_IDS = {
    MODEL_QWEN_3B:       "Qwen/Qwen2.5-VL-3B-Instruct",
    MODEL_QWEN_7B:       "Qwen/Qwen2.5-VL-7B-Instruct",
    MODEL_QWEN_7B_INT8:  "Qwen/Qwen2.5-VL-7B-Instruct",   # same weights, loaded in 8-bit
    MODEL_QWEN3_2B:      "Qwen/Qwen3-VL-2B-Instruct",
    MODEL_QWEN_32B_INT8: "Qwen/Qwen2.5-VL-32B-Instruct",  # same weights, loaded in 8-bit
}

QWEN_INT8_KEYS      = {MODEL_QWEN_7B_INT8, MODEL_QWEN_32B_INT8}
QWEN_MAX_NEW_TOKENS = 512

# Model identifiers — Gemma
MODEL_GEMMA_4B       = "gemma-4b"        # ~8GB VRAM bfloat16  — 2080ti
MODEL_GEMMA_4B_INT8  = "gemma-4b-int8"   # ~4GB VRAM int8     — 1080ti
MODEL_GEMMA_12B      = "gemma-12b"       # ~24GB VRAM bfloat16 — gb10
MODEL_GEMMA_12B_INT8 = "gemma-12b-int8"  # ~12GB VRAM int8    — 5060ti

GEMMA_MODEL_IDS = {
    MODEL_GEMMA_4B:       "google/gemma-3-4b-it",
    MODEL_GEMMA_4B_INT8:  "google/gemma-3-4b-it",   # same weights, loaded in 8-bit
    MODEL_GEMMA_12B:      "google/gemma-3-12b-it",
    MODEL_GEMMA_12B_INT8: "google/gemma-3-12b-it",  # same weights, loaded in 8-bit
}

GEMMA_INT8_KEYS      = {MODEL_GEMMA_4B_INT8, MODEL_GEMMA_12B_INT8}
GEMMA_MAX_NEW_TOKENS = 512

# Model identifiers — Phi (Microsoft)
MODEL_PHI35_VISION      = "phi-3.5-vision"       # ~4GB bf16 — 1080ti
MODEL_PHI35_VISION_INT8 = "phi-3.5-vision-int8"  # ~2GB int8 — any GPU
MODEL_PHI4_VISION       = "phi-4-vision"          # ~10GB bf16 — 5060ti

PHI_MODEL_IDS = {
    MODEL_PHI35_VISION:      "microsoft/Phi-3.5-vision-instruct",
    MODEL_PHI35_VISION_INT8: "microsoft/Phi-3.5-vision-instruct",   # same weights, 8-bit
    MODEL_PHI4_VISION:       "microsoft/Phi-4-multimodal-instruct",
}

PHI_INT8_KEYS      = {MODEL_PHI35_VISION_INT8}
PHI_MAX_NEW_TOKENS = 512

# Model identifiers — NVIDIA Nemotron VL
# TODO: verify HF model ID matches the release on huggingface.co/nvidia
MODEL_NVLM_12B      = "nvlm-12b"       # nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 — bf16 ~24GB — gb10
MODEL_NVLM_12B_INT8 = "nvlm-12b-int8"  # same weights, int8 — ~12GB — 5060ti

NVLM_MODEL_IDS = {
    MODEL_NVLM_12B:      "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
    MODEL_NVLM_12B_INT8: "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",  # same weights, 8-bit
}

NVLM_INT8_KEYS      = {MODEL_NVLM_12B_INT8}
NVLM_MAX_NEW_TOKENS = 512

# Scene analysis settings
SCENE_MIN_QUESTIONS = 5       # scenes with fewer questions are excluded from analysis
SCENE_OUTLIER_STD = 1.0       # std deviations from mean to flag a scene as outlier
