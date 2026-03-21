import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = Path(os.environ.get("VLM_OUTPUT_DIR", ROOT_DIR / "outputs"))
LOG_DIR = Path(os.environ.get("VLM_LOG_DIR", ROOT_DIR / "logs"))

# Known invalid dataset entries — skip during evaluation
INVALID_ENTRIES: set[int] = {5, 35, 37, 67, 69}

# Task identifiers
TASK_FAILURE_MODE = "failure_mode"
TASK_MULTIVIEW = "multiview"

# Prompt identifiers
PROMPT_DEFAULT = "default"
PROMPT_TEST = "test"

# Question type buckets, nested by task.
# Maps task -> (type name -> list of substrings, case-insensitive).
# A question can match multiple types — all matches are recorded.
# See scripts/list_question_type_overlaps.py for overlap analysis.
QUESTION_TYPES: dict[str, dict[str, list[str]]] = {
    TASK_FAILURE_MODE: {
        # Did the robot complete the overall task goal?
        "task_success": ["successfully completed the task"],
        # What grasp phase is currently shown in the image?
        "grasp_phase_current": ["which phase of the grasp action is shown"],
        # What comes next after the current action phase?
        "grasp_phase_next": ["what will be the robot's next action phase"],
        # Is the robot's gripper currently open or closed?
        "gripper_state": ["is the robot's gripper open"],
        # Is the robot holding the object securely?
        "grasp_stability": ["is the robot's grasp of the", "stable"],
        # Is the path to the target object clear?
        "obstacle_detection": ["is there any obstacle blocking the robot from reaching"],
    },
    TASK_MULTIVIEW: {
        # Which point in the second view corresponds to the same 3D location?
        "cross_view_correspondence": ["corresponding to the same 3d location", "same 3d location"],
        # Which marked point is closest/farthest from the camera (depth from single view)?
        "relative_depth": ["which colored point is closest", "which colored point is farthest",
                           "closest to the camera", "farthest from the camera"],
    },
}

# Model identifiers
MODEL_QWEN_3B = "qwen-3b"
MODEL_QWEN_7B = "qwen-7b"

QWEN_MODEL_IDS = {
    MODEL_QWEN_3B: "Qwen/Qwen2.5-VL-3B-Instruct",
    MODEL_QWEN_7B: "Qwen/Qwen2.5-VL-7B-Instruct",
}
QWEN_MAX_NEW_TOKENS = 512

# Scene analysis settings
SCENE_MIN_QUESTIONS = 5       # scenes with fewer questions are excluded from analysis
SCENE_OUTLIER_STD = 1.0       # std deviations from mean to flag a scene as outlier
