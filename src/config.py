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

# Model identifiers
MODEL_QWEN_3B = "qwen-3b"
MODEL_QWEN_7B = "qwen-7b"

QWEN_MODEL_IDS = {
    MODEL_QWEN_3B: "Qwen/Qwen2.5-VL-3B-Instruct",
    MODEL_QWEN_7B: "Qwen/Qwen2.5-VL-7B-Instruct",
}
QWEN_MAX_NEW_TOKENS = 512
