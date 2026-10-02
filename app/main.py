"""Kharcha API + UI.

  PARSER_BACKEND=ollama .venv/bin/uvicorn app.main:app --reload      # laptop, private
  PARSER_BACKEND=tinker  uvicorn app.main:app --host 0.0.0.0 --port $PORT   # Render demo
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app import db  # noqa: E402
from app.chat import Asker  # noqa: E402
from app.memory import Memory  # noqa: E402
from app.parser import ParseError, get_backend  # noqa: E402
from data.taxonomy import CATEGORIES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("kharcha")

app = FastAPI(title="Kharcha", version="0.1.0")
STATIC = ROOT / "app" / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_backend = None
memory = Memory()
asker = Asker(memory)


def backend():
    global _backend
    if _backend is None:
        _backend = get_backend()
        log.info("parser backend: %s", _backend.name)
    return _backend


class ParseIn(BaseModel):
    text: str
    use_memory: bool = True


class CorrectIn(BaseModel):
    id: int
    category: str
    remember: bool = True


class AskIn(BaseModel):
    question: str
    month: str | None = None
    thread_id: str | None = None


class FactIn(BaseModel):
    text: str


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "backend": os.environ.get("PARSER_BACKEND", "ollama"), "memory": memory.enabled}


@app.get("/categories")
def categories():
    return CATEGORIES


@app.post("/parse")
def parse(inp: ParseIn):
    text = inp.text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    hints = memory.hints_for(text) if inp.use_memory else []
    try:
        label, ms = backend().parse(text, hints)
    except ParseError as e:
        raise HTTPException(422, f"model output not parseable: {e}")
    except Exception as e:  # ollama down, tinker error ...
        log.exception("parse failed")
        raise HTTPException(502, f"parser backend error: {e}")
    entry = db.insert(text, label, backend().name, ms)
    entry["hints"] = hints
    return entry


@app.get("/ledger")
def ledger(month: str | None = None):
    month = month or date.today().strftime("%Y-%m")
    return {"entries": db.list_entries(month), "summary": db.summary(month)}


@app.post("/correct")
def correct(inp: CorrectIn):
    if inp.category not in CATEGORIES:
        raise HTTPException(400, "unknown category")
    entry = db.get(inp.id)
    if not entry:
        raise HTTPException(404, "no such entry")
    entry = db.set_category(inp.id, inp.category)
    fact = None
    if inp.remember and entry.get("counterparty"):
        try:
            fact = memory.remember_rule(entry["counterparty"], inp.category)
        except Exception as e:
            log.warning("remember_rule failed: %s", e)
    return {"entry": entry, "remembered": fact}


@app.delete("/entries/{entry_id}")
def delete_entry(entry_id: int):
    db.delete(entry_id)
    return {"ok": True}


@app.post("/fact")
def fact(inp: FactIn):
    try:
        stored = memory.remember_fact(inp.text.strip())
    except Exception as e:
        raise HTTPException(502, str(e))
    return {"ok": True, "remembered": stored}


@app.delete("/memories/{memory_id}")
def forget(memory_id: str):
    try:
        memory.forget(memory_id)
    except Exception as e:
        raise HTTPException(502, str(e))
    return {"ok": True}


@app.post("/ask")
def ask(inp: AskIn):
    month = inp.month or date.today().strftime("%Y-%m")
    try:
        return asker.ask(inp.question, db.summary_text(month), inp.thread_id)
    except Exception as e:
        log.exception("ask failed")
        raise HTTPException(502, str(e))


@app.get("/memories")
def memories():
    return memory.list_memories()
