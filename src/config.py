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

# Regex-based question type templates derived directly from the robo2vlm vqa.py
# generator functions. Variable slots (object names, camera names, step indices,
# language instructions) are replaced with regex wildcards. Use re.search with
# re.IGNORECASE. This is an alternative to the substring-based QUESTION_TYPES
# above — see _classify_task_and_types_template in data/dataset.py.
QUESTION_TYPE_TEMPLATES: dict[str, dict[str, list[str]]] = {
    TASK_FAILURE_MODE: {
        # S1 — vqa_robot_gripper_open
        "gripper_state":            [r"is the robot's gripper open\?"],
        # S3 — vqa_object_reachable
        "obstacle_detection":       [r"is there any obstacle blocking the robot from reaching .+\?"],
        # S4 — vqa_relative_direction
        "relative_direction":       [r"in the image from .+ at step \d+, which direction is the .+ relative to the robot's end effector\?"],
        # I1 — vqa_task_success_state
        "task_success":             [r"the robot is to .+\. has the robot successfully completed the task\?"],
        # I2 — is_stable_grasp
        "grasp_stability":          [r"is the robot's grasp of the .+ stable\?"],
        # I3 — vqa_goal_configuration
        "goal_configuration":       [r"the robot's task is to .+\. which configuration shows the goal state that the robot should achieve\?"],
        # I4 — vqa_action_understanding (grasp phase current)
        "grasp_phase_current":      [r"the robot is tasked to .+\. the robot is interacting with the .+\. which phase of the grasp action is shown in the image\?"],
        # I4 — vqa_next_action (grasp phase next)
        "grasp_phase_next":         [r"the robot is tasked to .+\. after .+, what will be the robot's next action phase\?"],
        # S6/I6 — vqa_trajectory_understanding
        "trajectory_understanding": [r"which language instruction best describes the robot's trajectory shown in the image\?"],
        # S6 — vqa_action_direction_selection
        "action_direction":         [r"the robot task is to .+\. which colored arrow correctly shows the direction the robot will move next\?"],
        # I5 — vqa_temporal_sequence
        "temporal_sequence":        [r"for the task '.+', what is the correct sequence of action phases shown in the images from left to right\?",
                                     r"what task is the robot performing in this sequence of images\?"],
    },
    TASK_MULTIVIEW: {
        # S8 — vqa_multi_view_correspondence
        "cross_view_correspondence": [r"in the left image \(.+ camera\), a red dot is marked\. which point is the closest point in the right image \(.+ camera\) corresponding to the same 3d location\?"],
        # S6 — vqa_relative_depth
        "relative_depth":            [r"in the image from .+, which colored point is (closest|farthest) to the camera\?"],
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
