"""Evaluate parsers on data/test.jsonl and write out/results.md (+ results.json, chart).

Systems (pick with --systems, comma separated):
  base      Qwen3.5-4B zero-shot via Tinker sampling
  big       openai/gpt-oss-120b zero-shot via Tinker sampling
  lora      Qwen3.5-4B + Kharcha LoRA via Tinker sampling (needs KHARCHA_SAMPLER_PATH or out/run.json)
  ollama    Kharcha LoRA merged + quantised, served locally by Ollama (needs `ollama create kharcha`)

  .venv/bin/python train/eval.py --systems base,big,lora,ollama
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import BASE_MODEL, PRICES, ROOT, load_jsonl, renderer_for

sys.path.insert(0, str(ROOT))
from app.parser import OllamaBackend, ParseError, extract_json  # noqa: E402
from data.taxonomy import HINTS_HEADER  # noqa: E402

OUT = ROOT / "out"


def norm(s):
    if s is None:
        return None
    return re.sub(r"[^a-z0-9@]", "", str(s).lower())


def score(pred: dict | None, gold: dict) -> dict:
    if pred is None:
        return {"valid": 0, "amount": 0, "direction": 0, "counterparty": 0, "channel": 0, "last4": 0, "date": 0, "category": 0, "all": 0}
    s = {
        "valid": 1,
        "amount": int(abs(float(pred.get("amount") or 0) - gold["amount"]) < 0.005),
        "direction": int(pred.get("direction") == gold["direction"]),
        "counterparty": int(norm(pred.get("counterparty")) == norm(gold["counterparty"])),
        "channel": int(pred.get("channel") == gold["channel"]),
        "last4": int((pred.get("account_last4") or None) == gold["account_last4"]),
        "date": int((pred.get("date") or None) == gold["date"]),
        "category": int(pred.get("category") == gold["category"]),
    }
    s["all"] = int(all(s[k] for k in ("amount", "direction", "counterparty", "channel", "last4", "date", "category")))
    return s


class TinkerSampler:
    """Zero-shot (base_model) or fine-tuned (model_path) sampler on Tinker."""

    def __init__(self, base_model: str | None = None, model_path: str | None = None):
        import tinker
        from tinker_cookbook.renderers import get_renderer

        self.service = tinker.ServiceClient()
        self.client = self.service.create_sampling_client(base_model=base_model, model_path=model_path)
        self.model = base_model or self.client.get_base_model() if hasattr(self.client, "get_base_model") else base_model
        if not isinstance(self.model, str):
            self.model = BASE_MODEL
        self.tok = self.client.get_tokenizer()
        self.renderer = get_renderer(renderer_for(self.model), self.tok)
        self.params = tinker.SamplingParams(max_tokens=260, temperature=0.0, stop=self.renderer.get_stop_sequences())

    def run(self, messages):
        prompt = self.renderer.build_generation_prompt(messages[:2])
        t0 = time.perf_counter()
        res = self.client.sample(prompt=prompt, num_samples=1, sampling_params=self.params).result()
        dt = (time.perf_counter() - t0) * 1000
        toks = res.sequences[0].tokens
        return self.tok.decode(toks), dt, prompt.length, len(toks)


class OllamaSampler:
    def __init__(self):
        self.b = OllamaBackend()

    def run(self, messages):
        t0 = time.perf_counter()
        reply = self.b.complete(messages[0]["content"], messages[1]["content"])
        dt = (time.perf_counter() - t0) * 1000
        return reply, dt, 0, 0


def evaluate(name: str, sampler, rows, workers: int, price_model: str | None):
    def one(r):
        try:
            reply, ms, p_tok, c_tok = sampler.run(r["messages"])
        except Exception as e:
            return {"error": str(e)[:200], "ms": None, "pred": None, "p": 0, "c": 0}
        try:
            pred = extract_json(reply)
        except ParseError as e:
            return {"error": str(e)[:200], "ms": ms, "pred": None, "p": p_tok, "c": c_tok, "raw": reply[:200]}
        return {"pred": pred, "ms": ms, "p": p_tok, "c": c_tok}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(one, rows))

    scores = [score(o["pred"], r["label"]) for o, r in zip(outs, rows)]
    n = len(rows)
    agg = {k: sum(s[k] for s in scores) / n for k in scores[0]}
    lat = [o["ms"] for o in outs if o["ms"] is not None]
    agg["p50_ms"] = statistics.median(lat) if lat else None
    agg["p95_ms"] = sorted(lat)[int(len(lat) * 0.95) - 1] if len(lat) >= 20 else (max(lat) if lat else None)
    tok_p = sum(o["p"] for o in outs); tok_c = sum(o["c"] for o in outs)
    if price_model and tok_p + tok_c:
        agg["usd_per_1k"] = (tok_p + tok_c) / n * 1000 / 1e6 * PRICES[price_model]["sample"]
    else:
        agg["usd_per_1k"] = 0.0 if name == "ollama" else None
    agg["n"] = n
    # per-kind breakdown + hinted subset
    kinds = {}
    for s, r in zip(scores, rows):
        kinds.setdefault(r["kind"], []).append(s["all"])
    agg["by_kind"] = {k: sum(v) / len(v) for k, v in sorted(kinds.items())}
    hinted = [s["category"] for s, r in zip(scores, rows) if HINTS_HEADER in r["messages"][0]["content"]]
    agg["hinted_category_acc"] = sum(hinted) / len(hinted) if hinted else None
    errors = [{"text": r["messages"][1]["content"], "gold": r["label"], "out": o} for o, r, s in zip(outs, rows, scores) if not s["all"]]
    return agg, errors


def write_report(results: dict):
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    labels = {"base": "Qwen3.5-4B zero-shot (Tinker)", "big": "gpt-oss-120b zero-shot (Tinker)",
              "lora": "Qwen3.5-4B + Kharcha LoRA (Tinker)", "ollama": "Kharcha LoRA, Ollama on M1 Pro (offline)"}
    cols = ["valid", "amount", "direction", "counterparty", "channel", "last4", "date", "category", "all"]
    lines = ["# Kharcha eval: 150 held-out messages", "",
             "| System | " + " | ".join(cols) + " | p50 ms | p95 ms | $/1k msgs |",
             "|---|" + "---|" * (len(cols) + 3)]
    for k, r in results.items():
        pct = " | ".join(f"{r[c]*100:.0f}%" for c in cols)
        p50 = f"{r['p50_ms']:.0f}" if r["p50_ms"] else "-"
        p95 = f"{r['p95_ms']:.0f}" if r["p95_ms"] else "-"
        usd = "$0 (local)" if r["usd_per_1k"] == 0 else (f"${r['usd_per_1k']:.3f}" if r["usd_per_1k"] is not None else "-")
        lines.append(f"| {labels.get(k,k)} | {pct} | {p50} | {p95} | {usd} |")
    lines += ["", "`all` = every field exact. `$/1k` uses Tinker sampling prices; Ollama is local so marginal cost is zero.", ""]
    lines.append("## Category accuracy on messages that carried a user hint (Backboard memory)")
    for k, r in results.items():
        if r.get("hinted_category_acc") is not None:
            lines.append(f"- {labels.get(k,k)}: {r['hinted_category_acc']*100:.0f}%")
    lines += ["", "## All-fields accuracy by template"]
    kinds = sorted({kk for r in results.values() for kk in r["by_kind"]})
    lines.append("| template | " + " | ".join(results) + " |")
    lines.append("|---|" + "---|" * len(results))
    for kk in kinds:
        lines.append(f"| {kk} | " + " | ".join(f"{results[s]['by_kind'].get(kk, 0)*100:.0f}%" for s in results) + " |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    try:
        chart(results, labels)
    except Exception as e:
        print("chart skipped:", e)


def chart(results, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [labels.get(k, k).replace(" (", "\n(") for k in results]
    acc = [results[k]["all"] * 100 for k in results]
    cat = [results[k]["category"] * 100 for k in results]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], acc, width=0.4, label="all fields exact", color="#2a9d8f")
    ax.bar([i + 0.2 for i in x], cat, width=0.4, label="category correct", color="#e9c46a")
    ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 100); ax.set_ylabel("% of 150 test messages")
    ax.legend(loc="lower right"); ax.set_title("Kharcha: SMS/voice -> ledger JSON")
    for i, v in enumerate(acc):
        ax.text(i - 0.2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "results.png", dpi=160)
    print("chart ->", OUT / "results.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="base,big,lora")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sampler-path", default=os.environ.get("KHARCHA_SAMPLER_PATH"))
    ap.add_argument("--big-model", default="openai/gpt-oss-120b")
    args = ap.parse_args()

    rows = load_jsonl(ROOT / "data" / "test.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    sampler_path = args.sampler_path
    if not sampler_path and (OUT / "run.json").exists():
        sampler_path = json.loads((OUT / "run.json").read_text()).get("sampler_path")

    results = {}
    if (OUT / "results.json").exists():
        results = json.loads((OUT / "results.json").read_text())
    for sysname in args.systems.split(","):
        sysname = sysname.strip()
        print(f"\n=== {sysname} on {len(rows)} messages")
        if sysname == "base":
            s, price = TinkerSampler(base_model=BASE_MODEL), BASE_MODEL
        elif sysname == "big":
            s, price = TinkerSampler(base_model=args.big_model), args.big_model
        elif sysname == "lora":
            if not sampler_path:
                raise SystemExit("need --sampler-path or out/run.json")
            s, price = TinkerSampler(model_path=sampler_path), BASE_MODEL
        elif sysname == "ollama":
            s, price = OllamaSampler(), None
        else:
            raise SystemExit(f"unknown system {sysname}")
        agg, errors = evaluate(sysname, s, rows, args.workers if sysname != "ollama" else 1, price)
        results[sysname] = agg
        (OUT / f"errors_{sysname}.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False, default=str))
        print(json.dumps({k: v for k, v in agg.items() if k != "by_kind"}, indent=1, default=str))
    write_report(results)


if __name__ == "__main__":
    main()
