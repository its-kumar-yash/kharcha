"""Backboard memory layer.

One Backboard assistant = one household. Facts the user teaches us ("Sharma Kirana is groceries",
"Rahul is my son") are stored as assistant-level memories, so they survive across threads and
sessions. Before parsing, we pull relevant memories and pass them to the model as hints; the
fine-tuned model was trained to honour a "Known rules from the user" block.

API: https://app.backboard.io/api  (X-API-Key header).  One-call messaging via POST /threads/messages.
Model routing goes to an open-weight model via OpenRouter by default (see .env.example).
"""
from __future__ import annotations

import json
import logging
import os
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
            return json.loads(STATE_FILE.read_text())
        return {}

    def _save_state(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(self.state, indent=2))

    def _post(self, path: str, body: dict) -> dict:
        r = requests.post(f"{BASE_URL}{path}", headers=self._headers(), json=body, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"Backboard {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    def _get(self, path: str, params: dict | None = None):
        r = requests.get(f"{BASE_URL}{path}", headers=self._headers(), params=params, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"Backboard {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    def assistant_id(self) -> str:
        """Create the household assistant once; reuse its id forever (memories hang off it)."""
        if aid := (os.environ.get("BACKBOARD_ASSISTANT_ID") or self.state.get("assistant_id")):
            return aid
        resp = self._post(
            "/assistants",
            {"name": "Kharcha household", "system_prompt": ASSISTANT_INSTRUCTIONS, "description": "Kharcha expense memory"},
        )
        aid = resp.get("assistant_id") or resp.get("id")
        self.state["assistant_id"] = aid
        self._save_state()
        return aid

    def _message(self, content: str, memory: str = "Auto", thread_id: str | None = None) -> dict:
        body = {
            "content": content,
            "assistant_id": self.assistant_id(),
            "memory": memory,
            "llm_provider": self.provider,
            "model_name": self.model,
            "stream": False,
        }
        if thread_id:
            body["thread_id"] = thread_id
        return self._post("/threads/messages", body)

    # ------------------------------------------------------------------ public API
    def remember_rule(self, counterparty: str, category: str) -> str:
        """User corrected a category: store it as a durable fact."""
        fact = f"Treat {counterparty} as {category}."
        if not self.enabled:
            return fact
        self._message(
            f"Remember this rule for my expenses: {fact} Reply with just 'noted'.",
            memory="Auto",
        )
        return fact

    def remember_fact(self, text: str) -> None:
        """Free-form household fact, e.g. 'Rahul Kumar is my son'."""
        if not self.enabled:
            return
        self._message(f"Remember: {text}. Reply with just 'noted'.", memory="Auto")

    def hints_for(self, text: str, limit: int = 6) -> list[str]:
        """Pull memories relevant to this SMS/voice note and phrase them as rules the parser understands."""
        if not self.enabled:
            return []
        try:
            resp = self._message(
                "List every remembered rule or household fact that could apply to this message, one per line, "
                "phrased as 'Treat <name> as <category>.' or '<Name> is family (<relation>); transfers to them are family_transfer.' "
                "If none apply, reply exactly 'none'.\n\nMessage: " + text,
                memory="Readonly",
            )
            content = _content(resp)
            if not content or content.strip().lower().startswith("none"):
                return []
            hints = [ln.strip("-• ").strip() for ln in content.splitlines() if ln.strip()]
            return [h for h in hints if h.lower() != "none"][:limit]
        except Exception as e:  # memory is best-effort; parsing must never fail because of it
            log.warning("hints_for failed: %s", e)
            return []

    def ask(self, question: str, ledger_summary: str, thread_id: str | None = None) -> tuple[str, str | None]:
        """Hinglish Q&A over the ledger. Returns (answer, thread_id)."""
        if not self.enabled:
            return "Backboard memory is not configured (set BACKBOARD_API_KEY).", None
        resp = self._message(
            f"Ledger summary:\n{ledger_summary}\n\nQuestion: {question}",
            memory="Auto",
            thread_id=thread_id,
        )
        return _content(resp) or "(no answer)", resp.get("thread_id")

    def list_memories(self) -> list:
        if not self.enabled:
            return []
        try:
            return self._get(f"/assistants/{self.assistant_id()}/memories")
        except Exception as e:
            log.warning("list_memories failed: %s", e)
            return []


def _content(resp: dict) -> str | None:
    """Backboard responses carry the reply under a few possible keys; normalise."""
    for k in ("content", "message", "response", "text"):
        v = resp.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict) and isinstance(v.get("content"), str):
            return v["content"]
    return None
