"""SQLite ledger. Single file, no ORM. Ephemeral on Render free tier (fine for the demo)."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, datetime
from pathlib import Path

DB_PATH = Path(os.environ.get("KHARCHA_DB", Path(__file__).resolve().parents[1] / "kharcha.db"))


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            direction TEXT, amount REAL, currency TEXT, counterparty TEXT,
            channel TEXT, account_last4 TEXT, date TEXT, category TEXT,
            backend TEXT, latency_ms INTEGER, corrected INTEGER DEFAULT 0
        )"""
    )
    return conn


def insert(raw_text: str, label: dict, backend: str, latency_ms: int) -> dict:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO entries (created_at, raw_text, direction, amount, currency, counterparty, channel,
               account_last4, date, category, backend, latency_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.utcnow().isoformat(timespec="seconds"),
                raw_text,
                label.get("direction"), label.get("amount"), label.get("currency"), label.get("counterparty"),
                label.get("channel"), label.get("account_last4"), label.get("date") or date.today().isoformat(),
                label.get("category"), backend, latency_ms,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM entries WHERE id=?", (cur.lastrowid,)).fetchone()
        return dict(row)


def get(entry_id: int) -> dict:
    with connect() as conn:
        row = conn.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone()
        return dict(row) if row else None


def set_category(entry_id: int, category: str) -> dict:
    with connect() as conn:
        conn.execute("UPDATE entries SET category=?, corrected=1 WHERE id=?", (category, entry_id))
        conn.commit()
    return get(entry_id)


def delete(entry_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM entries WHERE id=?", (entry_id,))


def list_entries(month: str | None = None, limit: int = 500) -> list[dict]:
    with connect() as conn:
        if month:
            rows = conn.execute(
                "SELECT * FROM entries WHERE substr(date,1,7)=? ORDER BY date DESC, id DESC LIMIT ?", (month, limit)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM entries ORDER BY date DESC, id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def summary(month: str) -> dict:
    """Totals by category for a YYYY-MM month, plus overall debit/credit."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT category, direction, SUM(amount) AS total, COUNT(*) AS n
               FROM entries WHERE substr(date,1,7)=? GROUP BY category, direction""",
            (month,),
        ).fetchall()
    by_cat, debit, credit = {}, 0.0, 0.0
    for r in rows:
        if r["direction"] == "debit":
            debit += r["total"]
            by_cat[r["category"]] = by_cat.get(r["category"], 0.0) + r["total"]
        else:
            credit += r["total"]
    return {"month": month, "debit": round(debit, 2), "credit": round(credit, 2),
            "by_category": dict(sorted(by_cat.items(), key=lambda kv: -kv[1]))}


def summary_text(month: str) -> str:
    """Compact plain-text summary handed to the Ask model as context."""
    s = summary(month)
    lines = [f"Month {month}: spent INR {s['debit']:.0f}, received INR {s['credit']:.0f}."]
    for cat, amt in s["by_category"].items():
        lines.append(f"- {cat}: INR {amt:.0f}")
    entries = list_entries(month, limit=60)
    if entries:
        lines.append("Recent entries:")
        for e in entries[:60]:
            lines.append(f"  {e['date']} {e['direction']} {e['amount']:.0f} {e['counterparty'] or ''} [{e['category']}]")
    return "\n".join(lines)
