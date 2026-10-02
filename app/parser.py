"""Parser backends. Both take raw text (+ optional memory hints) and return the label dict.

- OllamaBackend : local, private. The Tinker LoRA merged into Qwen3.5-4B and served by Ollama.
- TinkerBackend : hosted demo on Render. Same LoRA, served by Tinker's sampling API.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.taxonomy import FIELDS, build_system_prompt, validate  # noqa: E402

JSON_RE = re.compile(r"\{.*\}", re.S)


class ParseError(Exception):
    pass


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply; tolerate stray prose/thinking."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = JSON_RE.search(text)
    if not m:
        raise ParseError(f"no JSON in reply: {text[:200]!r}")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ParseError(f"bad JSON: {e}: {m.group(0)[:200]!r}")
    if not isinstance(obj, dict):
        raise ParseError("JSON is not an object")
    obj = {k: obj.get(k) for k in FIELDS}
    if isinstance(obj.get("amount"), str):
        try:
            obj["amount"] = float(obj["amount"].replace(",", ""))
        except ValueError:
            pass
    if obj.get("account_last4") is not None:
        obj["account_last4"] = str(obj["account_last4"]).strip()[-4:] or None
    errs = validate(obj)
    if errs:
        raise ParseError(f"invalid fields {errs}: {obj}")
    return obj


class ParserBackend:
    name = "base"

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    def parse(self, text: str, hints: list[str] | None = None) -> tuple[dict, int]:
        t0 = time.perf_counter()
        reply = self.complete(build_system_prompt(hints), text)
        label = extract_json(reply)
        return label, int((time.perf_counter() - t0) * 1000)


class OllamaBackend(ParserBackend):
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "kharcha")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")

    def complete(self, system: str, user: str) -> str:
        r = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 200},
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]


class TinkerBackend(ParserBackend):
    """Serve the fine-tuned LoRA from Tinker's sampling API (used by the Render demo)."""

    name = "tinker"

    def __init__(self, sampler_path: str | None = None):
        import tinker
        from tinker_cookbook.renderers import get_renderer

        self.sampler_path = sampler_path or os.environ["KHARCHA_SAMPLER_PATH"]
        self.service = tinker.ServiceClient()
        self.client = self.service.create_sampling_client(model_path=self.sampler_path)
        self.tokenizer = self.client.get_tokenizer()
        self.renderer = get_renderer(os.environ.get("TINKER_RENDERER", "qwen3_5_disable_thinking"), self.tokenizer)
        self.params = tinker.SamplingParams(max_tokens=200, temperature=0.0, stop=self.renderer.get_stop_sequences())

    def complete(self, system: str, user: str) -> str:
        prompt = self.renderer.build_generation_prompt(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        result = self.client.sample(prompt=prompt, num_samples=1, sampling_params=self.params).result()
        return self.tokenizer.decode(result.sequences[0].tokens)


def get_backend() -> ParserBackend:
    kind = os.environ.get("PARSER_BACKEND", "ollama").lower()
    if kind == "tinker":
        return TinkerBackend()
    return OllamaBackend()
