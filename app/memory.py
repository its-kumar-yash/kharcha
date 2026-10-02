"""Backboard memory layer.

One Backboard assistant = one household. Facts the user teaches us ("Treat Sharma Kirana as
groceries", "Rahul Kumar is family") are stored as assistant-level memories, so they survive
across threads, devices and sessions. Before parsing, we semantically search those memories
with the incoming SMS and pass the hits to the fine-tuned model as a "Known rules" block,
which it was trained to honour.

Endpoints used (base https://app.backboard.io/api, header X-API-Key):
  POST /assistants                              create the household assistant (once)
  POST /assistants/{id}/memories {content}      add a memory directly (no LLM call, free tier OK)
  POST /assistants/{id}/memories/search {query} semantic search, returns scores
  GET  /assistants/{id}/memories                list
  POST /threads/messages                        chat with memory (needs LLM credits on Backboard)
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import requests

log = logging.getLogger("kharcha.memory")
BASE_URL = os.environ.get("BACKBOARD_BASE_URL", "https://app.backboard.io/api").rstrip("/")
STATE_FILE = Path(os.environ.get("KHARCHA_STATE", Path(__file__).resolve().parents[1] / "out" / "backboard_state.json"))

ASSISTANT_INSTRUCTIONS = (
    "You are Kharcha, a gentle Hinglish-speaking household expense assistant for an Indian family. "
    "You remember facts the user tells you about merchants, people and categories. "
    "When asked about spending, answer briefly in the user's language (Hindi, English or Hinglish), "
    "using the ledger summary provided in the message. Amounts are INR."
)

STOPWORDS = {"store", "bank", "ltd", "pvt", "the", "and", "shop", "mr", "mrs", "wale", "wala", "ji"}
RULE_RE = re.compile(r"treat\s+(.+?)\s+as\s+([a-z_]+)\.?$", re.I)


class Memory:
    def __init__(self):
        self.api_key = os.environ.get("BACKBOARD_API_KEY")
        self.enabled = bool(self.api_key)
        self.provider = os.environ.get("BACKBOARD_LLM_PROVIDER", "openrouter")
        self.model = os.environ.get("BACKBOARD_MODEL_NAME", "qwen/qwen3-32b")
        self.state = self._load_state()
        if not self.enabled:
            log.warning("BACKBOARD_API_KEY not set: memory disabled, hints will be empty")

    # ------------------------------------------------------------------ plumbing
    def _headers(self):
        return {"X-API-Key": self.api_key, "Content-Type": "application/json"}

    def _load_state(self) -> dict:
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_state(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(self.state, indent=2))

    def _req(self, method: str, path: str, **kw) -> dict:
        r = requests.request(method, f"{BASE_URL}{path}", headers=self._headers(), timeout=60, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"Backboard {method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    def assistant_id(self) -> str:
        """Create the household assistant once; reuse its id forever (memories hang off it)."""
        if aid := (os.environ.get("BACKBOARD_ASSISTANT_ID") or self.state.get("assistant_id")):
            return aid
        resp = self._req("POST", "/assistants", json={
            "name": "Kharcha household", "system_prompt": ASSISTANT_INSTRUCTIONS, "description": "Kharcha expense memory"})
        aid = resp["assistant_id"]
        self.state["assistant_id"] = aid
        self._save_state()
        log.info("created Backboard assistant %s", aid)
        return aid

    # ------------------------------------------------------------------ writes
    def remember_rule(self, counterparty: str, category: str) -> str:
        """User corrected a category: store it as a durable, machine-readable rule."""
        fact = f"Treat {counterparty.strip()} as {category}."
        if self.enabled:
            self._add(fact)
        return fact

    def remember_fact(self, text: str) -> str:
        """Free-form household fact, e.g. 'Rahul Kumar is family (beta); transfers to them are family_transfer.'"""
        text = text.strip().rstrip(".") + "."
        if self.enabled:
            self._add(text)
        return text

    def _add(self, content: str):
        # de-dupe: if an identical rule exists, skip; if same subject with a different category, replace
        subj = _subject(content)
        for m in self.list_memories():
            if m["content"].strip().lower() == content.lower():
                return
            if subj and _subject(m["content"]) == subj:
                try:
                    self._req("DELETE", f"/assistants/{self.assistant_id()}/memories/{m['id']}")
                except Exception as e:
                    log.warning("could not delete stale memory: %s", e)
        self._req("POST", f"/assistants/{self.assistant_id()}/memories", json={"content": content})

    # ------------------------------------------------------------------ reads
    def search(self, query: str, limit: int = 8) -> list[dict]:
        if not self.enabled:
            return []
        resp = self._req("POST", f"/assistants/{self.assistant_id()}/memories/search", json={"query": query, "limit": limit})
        return resp.get("memories", [])

    def hints_for(self, text: str, limit: int = 5) -> list[str]:
        """Memories whose subject (merchant / person) actually appears in this SMS or voice note.
        Backboard's semantic search ranks candidates; embeddings of short rules score high for
        almost anything, so the final gate is lexical: at least half of the subject's words must
        occur in the message."""
        if not self.enabled:
            return []
        try:
            hits = self.search(text, limit=limit * 3)
        except Exception as e:  # memory is best-effort; parsing must never fail because of it
            log.warning("hints_for failed: %s", e)
            return []
        words = set(re.findall(r"[a-z0-9@]{3,}", text.lower()))
        out = []
        for h in hits:
            content = h["content"].strip()
            subj = _subject(content) or content.lower()
            subj_words = set(re.findall(r"[a-z0-9@]{3,}", subj)) - STOPWORDS
            if subj_words and len(subj_words & words) / len(subj_words) >= 0.5:
                out.append(content)
        return out[:limit]

    def list_memories(self) -> list[dict]:
        if not self.enabled:
            return []
        try:
            return self._req("GET", f"/assistants/{self.assistant_id()}/memories").get("memories", [])
        except Exception as e:
            log.warning("list_memories failed: %s", e)
            return []

    def forget(self, memory_id: str):
        self._req("DELETE", f"/assistants/{self.assistant_id()}/memories/{memory_id}")

    # ------------------------------------------------------------------ chat
    def chat(self, content: str, thread_id: str | None = None) -> tuple[str | None, str | None]:
        """Chat through Backboard. memory=Readonly on purpose: the ledger summary we send must be
        *used* for the answer but never *stored* as a memory. Returns (answer or None, thread_id)."""
        body = {"content": content, "assistant_id": self.assistant_id(), "memory": "Readonly",
                "llm_provider": self.provider, "model_name": self.model, "stream": False}
        if thread_id:
            body["thread_id"] = thread_id
        resp = self._req("POST", "/threads/messages", json=body)
        if resp.get("status") == "FAILED":
            log.warning("Backboard chat unavailable: %s", (resp.get("content") or "")[:120])
            return None, resp.get("thread_id")
        return resp.get("content"), resp.get("thread_id")


def _subject(rule: str) -> str | None:
    m = RULE_RE.match(rule.strip())
    if m:
        return m.group(1).strip().lower()
    m = re.match(r"(.+?)\s+is\s+family", rule.strip(), re.I)
    return m.group(1).strip().lower() if m else None
