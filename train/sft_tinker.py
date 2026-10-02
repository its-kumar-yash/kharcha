"""LoRA supervised fine-tune of Qwen3.5-4B on Tinker for the Kharcha SMS->JSON task.

  export TINKER_API_KEY=...
  .venv/bin/python train/sft_tinker.py            # ~100 steps, prints sampler path
  .venv/bin/python train/sft_tinker.py --epochs 2 --batch 32 --lr 2e-4

Writes out/run.json with {sampler_path, state_path, steps, train_loss[], val_loss[]}.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import tinker
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised.data import conversation_to_datum

from common import BASE_MODEL, RENDERER, ROOT, load_jsonl

OUT = ROOT / "out"


def to_datums(rows, renderer, max_length):
    return [
        conversation_to_datum(r["messages"], renderer, max_length=max_length, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
        for r in rows
    ]


def _arr(x):
    return x.to_numpy() if hasattr(x, "to_numpy") else x


def _field(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def mean_loss(fwdbwd_output) -> float:
    """Mean NLL over supervised tokens. Tinker returns per-datum logprobs; weights come back
    normalised (reduction='mean' in conversation_to_datum), so sum(-lp*w) is already a mean."""
    total, n = 0.0, 0
    for out in fwdbwd_output.loss_fn_outputs:
        lp = _field(out, "logprobs")
        if lp is None:
            break
        lp = _arr(lp)
        w = _field(out, "weights")
        if w is not None:
            w = _arr(w)
            total += float(-(lp * w).sum())
            n += 1
        else:
            total += float(-lp.mean())
            n += 1
    if n:
        return total / n
    m = getattr(fwdbwd_output, "metrics", {}) or {}
    for k in ("loss", "nll", "mean_nll", "loss:sum"):
        if k in m:
            return float(m[k])
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--renderer", default=RENDERER)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=640)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--name", default="kharcha-v1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    train_rows = load_jsonl(ROOT / "data" / "train.jsonl")
    val_rows = load_jsonl(ROOT / "data" / "val.jsonl")
    print(f"train={len(train_rows)} val={len(val_rows)} base={args.base_model} rank={args.rank}")

    service = tinker.ServiceClient()
    tc = service.create_lora_training_client(base_model=args.base_model, rank=args.rank)
    renderer = get_renderer(args.renderer, tc.get_tokenizer())
    print("console:", tc.get_console_url() if hasattr(tc, "get_console_url") else "n/a")

    train_data = to_datums(train_rows, renderer, args.max_length)
    val_data = to_datums(val_rows, renderer, args.max_length)
    lens = [d.model_input.length for d in train_data]
    print(f"tokens/example: mean={sum(lens)/len(lens):.0f} max={max(lens)}")

    steps_per_epoch = len(train_data) // args.batch
    total_steps = int(steps_per_epoch * args.epochs)
    print(f"steps/epoch={steps_per_epoch} total_steps={total_steps}")

    log = {"base_model": args.base_model, "rank": args.rank, "lr": args.lr, "batch": args.batch,
           "epochs": args.epochs, "train_loss": [], "val_loss": [], "started": time.time()}
    step = 0
    while step < total_steps:
        random.shuffle(train_data)
        for i in range(0, len(train_data) - args.batch + 1, args.batch):
            if step >= total_steps:
                break
            batch = train_data[i : i + args.batch]
            # linear warmup for first 5 steps, then cosine-ish decay to 10%
            frac = step / max(total_steps - 1, 1)
            lr = args.lr * min(1.0, (step + 1) / 5) * (0.1 + 0.9 * (1 - frac))
            fb = tc.forward_backward(batch, "cross_entropy")
            op = tc.optim_step(adam_params=tinker.AdamParams(learning_rate=lr))
            fb_out = fb.result()
            op.result()
            loss = mean_loss(fb_out)
            log["train_loss"].append([step, loss])
            msg = f"step {step:4d}/{total_steps} lr={lr:.2e} train_nll={loss:.4f}"
            if step % args.eval_every == 0 or step == total_steps - 1:
                v = tc.forward(val_data, "cross_entropy").result()
                vloss = mean_loss(v)
                log["val_loss"].append([step, vloss])
                msg += f" val_nll={vloss:.4f}"
            print(msg, flush=True)
            step += 1

    sampler_path = tc.save_weights_for_sampler(name=args.name).result().path
    state_path = tc.save_state(name=f"{args.name}-state").result().path
    log.update({"sampler_path": sampler_path, "state_path": state_path, "steps": step, "finished": time.time()})
    OUT.mkdir(exist_ok=True)
    (OUT / "run.json").write_text(json.dumps(log, indent=2))
    print("\nSAMPLER_PATH =", sampler_path)
    print("STATE_PATH   =", state_path)
    print("Add to .env:  KHARCHA_SAMPLER_PATH=" + sampler_path)


if __name__ == "__main__":
    main()
