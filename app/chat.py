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
        from app.tinker_client import TinkerSampler

        self.model = os.environ.get("TINKER_CHAT_MODEL", "openai/gpt-oss-120b")
        self.sampler = TinkerSampler(base_model=self.model, max_tokens=400, temperature=0.3)

    def complete(self, system: str, user: str) -> str:
        return self.sampler.complete(system, user)


class Asker:
    def __init__(self, memory: Memory):
        self.memory = memory
        self._fallback = None
        # Backboard chat needs LLM credits on the account. Try once; if it reports no credits,
        # stop paying the round-trip on every question until the process restarts.
        self.backboard_chat = os.environ.get("BACKBOARD_CHAT", "auto").lower() != "off"

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
            if self.backboard_chat:
                try:
                    answer, tid = self.memory.chat(f"Ledger summary:\n{ledger_summary}\n\nQuestion: {question}", thread_id)
                    if answer:
                        return {"answer": answer, "thread_id": tid, "via": "backboard-chat", "facts": facts}
                    self.backboard_chat = False
                    log.warning("Backboard chat disabled for this process (no LLM credits); using open-weight fallback")
                except Exception as e:
                    log.warning("backboard chat failed: %s", e)
        user = f"Remembered household facts:\n" + ("\n".join(f"- {f}" for f in facts) or "- (none)") + \
               f"\n\nLedger summary:\n{ledger_summary}\n\nQuestion: {question}"
        answer = self.fallback().complete(SYSTEM, user)
        return {"answer": answer, "thread_id": thread_id, "via": type(self.fallback()).__name__, "facts": facts}
