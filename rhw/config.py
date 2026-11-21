"""Shared settings for data building, scoring and analysis."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
UNITS_PATH = ROOT / "data" / "units.jsonl"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

MAIN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SMALL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# full-transcript truncation: keep head + tail when over MAX_TOKENS
MAX_TOKENS = 7000
HEAD_TOKENS = 1500
TAIL_TOKENS = 5500

LAST_K = (2, 4, 8)
WINDOW_SIZES = (2, 4, 8)
# max sliding windows scored per unit (evenly spaced); None = all. Set for free-Colab compute.
MAX_WINDOWS = 8
TOOL_OUTPUT_CHARS = 300

SEEDS = (0, 1, 2)
NEG_RATIO = 4
NEG_CAP = 400
