# Kharcha · ghar ka hisaab, phone se bahar nahi jaata

A tiny, private expense ledger for my parent. Paste any Indian bank/UPI SMS, or say
"aaj sabzi wale ko 80 diye", and a 4B open-weight model fine-tuned on
[Tinker](https://thinkingmachines.ai/tinker/) turns it into a categorised ledger row.
The model runs **offline on a laptop with Ollama**; bank SMS never leave the machine.
[Backboard](https://backboard.io) remembers the corrections ("Sharma Kirana is groceries",
"Rahul is my son") and answers Hinglish questions about the month. The demo is hosted on
[Render](https://render.com): **https://kharcha-4aax.onrender.com** (free tier, ~1 min cold start).

Built for the DEV Hacktoberfest Weekend Challenge "Build for a Friend" (Oct 2026).

## How it works

```
SMS / voice note ─▶ Qwen3.5-4B + Kharcha LoRA ─▶ {"direction","amount","counterparty","channel","account_last4","date","category"}
                          ▲ hints                         │
                 Backboard memory  ◀── corrections ───────┘
```

* `data/` – program-labelled dataset generator (real HDFC/SBI/ICICI/Axis/Kotak/PNB/PhonePe/GPay/Paytm formats + Hinglish voice notes). Labels are exact because the program that writes the SMS also writes the label.
* `train/sft_tinker.py` – LoRA SFT on Tinker (`Qwen/Qwen3.5-4B`, rank 16, ~100 steps, < $1).
* `train/eval.py` – 150 held-out messages vs zero-shot Qwen3.5-4B and gpt-oss-120b. Produces `out/results.md`.
* `train/export_adapter.py` – download LoRA → PEFT → merge → GGUF → `ollama create kharcha`.
* `app/` – FastAPI + single-page UI. `PARSER_BACKEND=ollama` (private) or `tinker` (hosted demo).

## Run it yourself

```bash
brew install uv ollama
uv venv --python 3.12 && uv pip install -r requirements-train.txt
cp .env.example .env            # add TINKER_API_KEY, BACKBOARD_API_KEY

.venv/bin/python data/gen_dataset.py          # 1200/150/150
.venv/bin/python train/sft_tinker.py          # prints KHARCHA_SAMPLER_PATH -> put in .env
.venv/bin/python train/eval.py --systems base,big,lora
.venv/bin/python train/export_adapter.py      # -> ollama model "kharcha"
.venv/bin/python train/eval.py --systems ollama

ollama serve &
PARSER_BACKEND=ollama .venv/bin/uvicorn app.main:app --reload   # http://127.0.0.1:8000
```

## Results

See `out/results.md` after running the eval (table reproduced in the DEV post).

## Why open weights

* The adapter is ~150 MB and ours. Swap the base model by changing one string.
* Fine-tune cost under a dollar; inference cost at home is zero.
* A parent's bank SMS are the most private text they have. They stay on the laptop.
