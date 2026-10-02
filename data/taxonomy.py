"""Shared schema for Kharcha: categories, field order, and the system prompt
used both for training and at inference. Keep this the single source of truth."""
from __future__ import annotations

import json

CATEGORIES = [
    "groceries",
    "food_delivery",
    "transport",
    "utilities",
    "medical",
    "rent_emi",
    "shopping",
    "family_transfer",
    "salary_income",
    "subscriptions",
    "cash_withdrawal",
    "other",
]

CHANNELS = ["upi", "card", "neft", "imps", "atm", "cash", "unknown"]
DIRECTIONS = ["debit", "credit"]

# Fixed key order so the model learns one canonical serialization.
FIELDS = [
    "direction",
    "amount",
    "currency",
    "counterparty",
    "channel",
    "account_last4",
    "date",
    "category",
]

SYSTEM_PROMPT = (
    "You convert one Indian bank/UPI SMS or a short Hinglish spoken note about money "
    "into a single JSON object and nothing else.\n"
    "Keys in this order: direction, amount, currency, counterparty, channel, account_last4, date, category.\n"
    "direction: debit|credit. amount: number. currency: INR. counterparty: the merchant, person or VPA as written, "
    "or null. channel: upi|card|neft|imps|atm|cash|unknown. account_last4: last 4 digits of the account/card or null. "
    "date: YYYY-MM-DD or null if absent. category: one of "
    + ", ".join(CATEGORIES)
    + ".\nTransfers to a named person are family_transfer only if the text or the known rules say so; otherwise other."
)

HINTS_HEADER = "Known rules from the user:"


def build_system_prompt(hints: list[str] | None = None) -> str:
    """System prompt, optionally with user-specific memory hints (from Backboard)."""
    if not hints:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + "\n" + HINTS_HEADER + "\n" + "\n".join(f"- {h}" for h in hints)


def to_json(label: dict) -> str:
    """Canonical compact serialization in FIELDS order."""
    ordered = {k: label.get(k) for k in FIELDS}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def validate(label: dict) -> list[str]:
    errs = []
    if label.get("direction") not in DIRECTIONS:
        errs.append("direction")
    if not isinstance(label.get("amount"), (int, float)):
        errs.append("amount")
    if label.get("currency") != "INR":
        errs.append("currency")
    if label.get("channel") not in CHANNELS:
        errs.append("channel")
    if label.get("category") not in CATEGORIES:
        errs.append("category")
    l4 = label.get("account_last4")
    if l4 is not None and not (isinstance(l4, str) and len(l4) == 4 and l4.isdigit()):
        errs.append("account_last4")
    return errs
