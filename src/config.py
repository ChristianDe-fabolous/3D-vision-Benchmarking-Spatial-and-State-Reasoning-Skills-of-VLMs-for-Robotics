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

# Allowed question patterns per task.
# A sample is kept if its question contains ANY of the listed substrings (case-insensitive).
# Empty list = no filter, all questions for that task are allowed.
ALLOWED_QUESTION_PATTERNS: dict[str, list[str]] = {
    TASK_FAILURE_MODE: [],
    TASK_MULTIVIEW: [],
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
