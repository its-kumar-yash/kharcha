"""Answering 'is mahine kirane pe kitna gaya?'.

Preferred path: Backboard chat (memory=Auto) when the account has LLM credits.
Fallback (always open-weight): Backboard memories + ledger summary -> an open model we run
ourselves — Ollama locally, or Tinker sampling of gpt-oss-120b / Qwen for the hosted demo.
"""
from __future__ import annotations

import logging
import os

import requests

from app.memory import Memory

log = logging.getLogger("kharcha.chat")

SYSTEM = (
    "You are Kharcha, a warm, brief household expense assistant for an Indian family. "
    "Answer in the same language the user writes in (Hindi, English or Hinglish). Use the ledger summary and the "
    "remembered household facts given to you. Amounts are INR; use the ₹ sign. Keep answers to 1-3 sentences. "
    "If the ledger does not contain the answer, say so plainly."
)


class OllamaChat:
    def __init__(self):
        self.model = os.environ.get("OLLAMA_CHAT_MODEL", "qwen3.5:4b")
        self.host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

    def complete(self, system: str, user: str) -> str:
        r = requests.post(f"{self.host}/api/chat", json={
            "model": self.model, "stream": False, "think": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0.3, "num_predict": 300}}, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"]


class TinkerChat:
    def __init__(self):
        import tinker
        from tinker_cookbook.renderers import get_renderer

        self.model = os.environ.get("TINKER_CHAT_MODEL", "openai/gpt-oss-120b")
        renderer = os.environ.get("TINKER_CHAT_RENDERER", "gpt_oss_low_reasoning" if "gpt-oss" in self.model else "qwen3_5_disable_thinking")
        self.client = tinker.ServiceClient().create_sampling_client(base_model=self.model)
        self.tok = self.client.get_tokenizer()
        self.renderer = get_renderer(renderer, self.tok)
        self.params = tinker.SamplingParams(max_tokens=400, temperature=0.3, stop=self.renderer.get_stop_sequences())

    def complete(self, system: str, user: str) -> str:
        prompt = self.renderer.build_generation_prompt([{"role": "system", "content": system}, {"role": "user", "content": user}])
        res = self.client.sample(prompt=prompt, num_samples=1, sampling_params=self.params).result()
        text = self.tok.decode(res.sequences[0].tokens)
        # gpt-oss harmony: keep only the final channel if present
        if "<|channel|>final<|message|>" in text:
            text = text.split("<|channel|>final<|message|>")[-1]
        return text.replace("<|return|>", "").replace("<|end|>", "").strip()


class Asker:
    def __init__(self, memory: Memory):
        self.memory = memory
        self._fallback = None

    def fallback(self):
        if self._fallback is None:
            kind = os.environ.get("CHAT_BACKEND") or os.environ.get("PARSER_BACKEND", "ollama")
            self._fallback = TinkerChat() if kind == "tinker" else OllamaChat()
            log.info("chat fallback: %s", type(self._fallback).__name__)
        return self._fallback

    def ask(self, question: str, ledger_summary: str, thread_id: str | None = None) -> dict:
        facts = []
        if self.memory.enabled:
            try:
                facts = [m["content"] for m in self.memory.search(question, limit=6)]
            except Exception as e:
                log.warning("memory search failed: %s", e)
            try:
                answer, tid = self.memory.chat(f"Ledger summary:\n{ledger_summary}\n\nQuestion: {question}", thread_id)
                if answer:
                    return {"answer": answer, "thread_id": tid, "via": "backboard-chat", "facts": facts}
            except Exception as e:
                log.warning("backboard chat failed: %s", e)
        user = f"Remembered household facts:\n" + ("\n".join(f"- {f}" for f in facts) or "- (none)") + \
               f"\n\nLedger summary:\n{ledger_summary}\n\nQuestion: {question}"
        answer = self.fallback().complete(SYSTEM, user)
        return {"answer": answer, "thread_id": thread_id, "via": type(self.fallback()).__name__, "facts": facts}
