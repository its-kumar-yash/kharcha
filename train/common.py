"""Shared bits for training/eval: data loading, renderer choice, pricing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE_MODEL = "Qwen/Qwen3.5-4B"
RENDERER = "qwen3_5_disable_thinking"

# USD per million tokens on Tinker (docs, Oct 2026). Sampling price used for inference cost.
PRICES = {
    "Qwen/Qwen3.5-4B": {"train": 0.737, "sample": 1.005},
    "Qwen/Qwen3.5-9B": {"train": 1.463, "sample": 1.995},
    "openai/gpt-oss-120b": {"train": 0.737, "sample": 0.84},
    "openai/gpt-oss-20b": {"train": 0.396, "sample": 0.45},
    "Qwen/Qwen3.8-27B": {"train": 4.103, "sample": 5.595},
}

RENDERERS = {
    "Qwen/Qwen3.5-4B": "qwen3_5_disable_thinking",
    "Qwen/Qwen3.5-9B": "qwen3_5_disable_thinking",
    "Qwen/Qwen3.8-27B": "qwen3_8_disable_thinking",
    "openai/gpt-oss-120b": "gpt_oss_low_reasoning",
    "openai/gpt-oss-20b": "gpt_oss_low_reasoning",
}


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def renderer_for(model: str) -> str:
    return RENDERERS.get(model, "qwen3_5_disable_thinking")
