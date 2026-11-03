"""Central configuration for the BCNF LLM controlled-evidence study.

API keys are NEVER hard-coded. Load them from environment / .env only.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
# Sibling frozen algorithm experiments repo (import-only; do not modify).
EXPERIMENTS_ROOT = PROJECT_ROOT.parent / "experiments"

load_dotenv(PROJECT_ROOT / ".env")

DATASET_VERSION = "bcnf_llm_v1"
PROMPT_VERSION = "v1"
GENERATION_SEED = 42
EVIDENCE_ORDER_SEED = 42

STUDENT_COUNT = 1000
COURSE_COUNT = 100
ENROLLMENT_COUNT = 15000

QUESTION_COUNT = 1200
CATEGORY_COUNTS = {
    "No-Conflict": 360,
    "Irrelevant-Conflict": 360,
    "Answer-Critical-Conflict": 480,
}

# Unified generation parameters (paper-fixed).
TEMPERATURE = 0.01
MAX_TOKENS = 32
STREAM = False
TIMEOUT_SEC = 60
MAX_RETRIES = 5
RETRY_BACKOFF_SEC = (2, 4, 8, 16, 32)
DEFAULT_CONCURRENCY = 4

# Model snapshots (no floating aliases in formal runs).
_DEFAULT_ALIYUN_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# Prefer user-configured workspace OpenAI-compatible endpoint; never invent Workspace URLs.
ALIYUN_BASE_URL = (os.getenv("ALIYUN_BASE_URL") or "").strip() or _DEFAULT_ALIYUN_BASE

MODELS = {
    "zhipu": {
        "provider": "zhipu",
        "model": "glm-4-flash-250414",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "api_key_env": "ZAI_API_KEY",
        "enable_thinking": None,
        "pricing_cny_per_m": {"input": 0.0, "output": 0.0},
        "pricing_note": "Treated as free official API snapshot as of 2026-08-17.",
    },
    "bailian": {
        "provider": "bailian",
        "model": "qwen3.7-flash-2026-07-15",
        "base_url": ALIYUN_BASE_URL,
        "api_key_env": "DASHSCOPE_API_KEY",
        "enable_thinking": False,
        "pricing_cny_per_m": {
            "input": float(os.getenv("QWEN_INPUT_CNY_PER_M", "0.2")),
            "output": float(os.getenv("QWEN_OUTPUT_CNY_PER_M", "0.8")),
        },
        "pricing_note": "Pricing snapshot date = 2026-08-17; override via env.",
    },
}

QWEN_INPUT_CNY_PER_M = float(os.getenv("QWEN_INPUT_CNY_PER_M", "0.2"))
QWEN_OUTPUT_CNY_PER_M = float(os.getenv("QWEN_OUTPUT_CNY_PER_M", "0.8"))
MAX_QWEN_BUDGET_CNY = float(os.getenv("MAX_QWEN_BUDGET_CNY", "20"))
PRICING_SNAPSHOT_DATE = "2026-08-17"

SMOKE_SEED = 20260817
SMOKE_CATEGORY_COUNTS = {
    "No-Conflict": 30,
    "Irrelevant-Conflict": 30,
    "Answer-Critical-Conflict": 40,
}
EXP2_ERROR_RATES = [0.01, 0.05, 0.10, 0.20]
EXP2_CORRUPTION_SEEDS = [101, 202, 303]

DATA_DIR = PROJECT_ROOT / "data"
DATABASE_DIR = DATA_DIR / "database"
QA_DIR = DATA_DIR / "qa"
CANDIDATE_DIR = DATA_DIR / "candidate_repairs"
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
FINAL_DIR = RESULTS_DIR / "final"
LOGS_DIR = RESULTS_DIR / "logs"


def get_api_key(provider: str) -> str:
    """Read API key from environment; never log the full value."""
    meta = MODELS[provider]
    env_name = meta["api_key_env"]
    key = os.getenv(env_name, "").strip()
    return key


def mask_secret(value: str, keep: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}...{value[-keep:]}"
