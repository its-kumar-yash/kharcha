---
title: Kharcha: a 4B model that reads my mother's bank SMS so her money never leaves her laptop
published: false
tags: devchallenge, weekendchallenge, hf26challenge
---

*This is a submission for the [Hacktoberfest Weekend Challenge: Build for a Friend](https://dev.to/challenges/hacktoberfest-weekend-2026-10-01)*

## What I Built

My mother keeps a diary. Every evening she copies the day's spending into it by hand, and every few days she forwards me a bank SMS with the same question: *"ye kya tha?"* Was this the gas cylinder or the electricity bill? Is "AGARWAL SABZI BHANDAR" the vegetable man or something else? Half the entries in the diary say "online" and nothing more.

She will not install a finance app. She is right not to. Those apps read every SMS on the phone and ship them to a server she has never heard of, and the SMS from a bank is the most private text a person receives.

**Kharcha** (Hindi for "expense") is a small ledger that runs entirely on a laptop. You paste a bank or UPI SMS, or you type what you would have said out loud ("aaj sabzi wale ko 80 diye"), and a 4-billion-parameter open-weight model turns it into one clean row: who, how much, which account, which date, which category. It learns from her corrections. If she changes "Sharma Kirana" from shopping to groceries once, it stays groceries. If she tells it Rahul is her son, transfers to Rahul become family, not "other".

It understands the formats of HDFC, SBI, ICICI, Axis, Kotak and PNB, the PhonePe, Google Pay and Paytm notifications, and Hinglish with spoken numbers like "dhai hazaar" and "baarah sau".

## Demo

![Kharcha demo: paste an SMS, say a voice note, fix a category once, ask in Hinglish](https://raw.githubusercontent.com/its-kumar-yash/kharcha/main/results/demo.gif)

Hosted demo: **https://kharcha-4aax.onrender.com** (free tier, first load takes about a minute to wake up, and the ledger resets on every deploy). The demo serves the same fine-tuned adapter from Tinker's sampling API because the free tier cannot hold a 4B model.

In the recording: a bank SMS and two spoken notes become ledger rows, "Sharma Kirana" gets corrected from food delivery to groceries once and the app says *yaad rakh liya* (remembered), and a Hinglish question gets a Hinglish answer.

## Code

{% embed https://github.com/its-kumar-yash/kharcha %}

## How I Built It

The core is a **LoRA fine-tune of Qwen3.5-4B** trained on [Tinker](https://thinkingmachines.ai/tinker/) and exported to run locally with Ollama. Around it: a FastAPI app with a single HTML page, [Backboard](https://backboard.io) as the memory layer, and [Render](https://render.com) hosting the demo.

### 1. A dataset with no labelling errors

Hand-labelling 1,500 SMS in a weekend was not going to happen, and asking a big model to label them would have baked its mistakes into mine. So I wrote the SMS the way the banks write them. Twenty-six templates copied from real formats, filled with real merchant names, Indian number grouping (`1,72,300.00`), six date styles, VPAs, reference numbers, and Hinglish voice notes with word numbers. **The program that writes the message also writes the label**, so the ground truth is exact by construction. A quarter of the training messages are then corrupted the way phones corrupt them: truncated, lower-cased, punctuation stripped.

About one in eight examples carries a *"Known rules from the user"* block in the system prompt, such as `Treat Agarwal Sabzi Bhandar as groceries.` or `Rahul Kumar is family (beta)`. That is how the model learns to obey the memory layer rather than its own prior.

```
1200 train / 150 val / 150 test · 12 categories · 26 message templates
```

### 2. Fine-tuning on Tinker

Tinker gives you a training loop, not a black box. The whole run is `forward_backward` plus `optim_step` on batches of rendered conversations:

```python
training_client = service.create_lora_training_client(base_model="Qwen/Qwen3.5-4B", rank=16)
renderer = get_renderer("qwen3_5_disable_thinking", training_client.get_tokenizer())
data = [conversation_to_datum(conv, renderer, max_length=640,
                              train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE) for conv in rows]
for step in range(111):
    fb = training_client.forward_backward(batch(step), "cross_entropy")
    training_client.optim_step(adam_params=tinker.AdamParams(learning_rate=lr(step)))
sampler_path = training_client.save_weights_for_sampler(name="kharcha-v1").result().path
```

Three epochs, 111 steps, rank 16, about 1.1 million training tokens. **The run cost under a dollar.** The model only sees loss on the JSON tokens; the 266-token system prompt and the SMS are context, not targets.

### 3. Did it actually beat the baseline?

Same 150 held-out messages, same prompt, exact-match on every field. The two baselines are zero-shot: the untouched Qwen3.5-4B and gpt-oss-120b, a model thirty times larger.

| System | valid JSON | amount | counterparty | date | category | **all fields** | p50 latency | $ / 1k msgs |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B zero-shot | 93% | 89% | 76% | 72% | 71% | **28%** | 2.2 s | $0.33 |
| gpt-oss-120b zero-shot | 97% | 97% | 85% | 76% | 81% | **49%** | 2.8 s | $0.38 |
| Qwen3.5-4B + Kharcha LoRA | 100% | 100% | 100% | 100% | 100% | **97%** | 2.7 s | $0.31 |

![All-fields and category accuracy on 150 held-out messages](https://raw.githubusercontent.com/its-kumar-yash/kharcha/main/results/results.png)

The four misses are all the same thing: whether "jooti ke 1200 lage" was paid in cash or by an unknown channel. My own labels decide that arbitrarily, so I count those as label noise rather than model error.

Two honest caveats. First, part of the gap is convention: the big model writes `"ATM"` where my label says `"SBI Bank ATM LAJPAT NAGAR"`, and both are defensible. Category accuracy, which has no convention problem, still moves from 81% to 100%. Second, the latency on Tinker is similar across models because it is dominated by queueing; the latency that matters is the offline one on a laptop.

Where the gap is real: the big model calls a petrol pump "other", a Jio recharge "other", a Croma autopay "subscriptions", and misses the date inside a PhonePe transaction ID. The fine-tuned 4B gets every one, because it has seen a thousand of them.

On the 7 test messages that carried a user rule, the baselines followed the rule 86% of the time. The fine-tune followed it 100%. That is the number that makes the memory layer trustworthy.

### 4. Bringing the weights home

```python
adapter_dir = weights.download(tinker_path=sampler_path, output_dir="out/adapter_raw")
weights.build_lora_adapter(base_model="Qwen/Qwen3.5-4B", adapter_path=adapter_dir, output_path="out/peft_adapter")
```

That is a 146 MB `adapter_model.safetensors` sitting in my `out/` folder: the whole of what the model learned, in a file I own. `train/export_adapter.py` carries it the rest of the way, merging into the base, converting to GGUF and registering it with Ollama, and `app/parser.py` already has the `OllamaBackend` that the UI switches to with `PARSER_BACKEND=ollama`. I ran out of weekend (and of home bandwidth: the base weights are 8 GB) before I could time that last step on the laptop, so the table above shows Tinker-served numbers only. The hosted demo and the laptop run the same code and the same adapter; only that one environment variable differs.

### 5. Memory with Backboard

Backboard stores facts at the assistant level, so one "household" assistant remembers across threads, devices and sessions. When my mother fixes a category, the app writes one memory:

```
POST /assistants/{id}/memories   {"content": "Treat SHARMA KIRANA STORE as groceries."}
```

Before parsing the next SMS it runs a semantic search over those memories with the raw message as the query, keeps only hits whose subject actually appears in the text, and prepends them to the system prompt as the *Known rules* block the model was trained on. Memory writes and reads are plain API calls with no LLM in the loop, which is what you want for something as deterministic as "this merchant is groceries".

The Ask box ("is mahine kirane pe kitna gaya?") pulls the relevant memories plus the month's ledger summary and hands them to an open-weight model: the stock Qwen3.5-4B in Ollama at home, gpt-oss-120b through Tinker's sampling API on the hosted demo. Backboard's own chat endpoint is wired in too and takes over automatically when the account has LLM credits, but I liked that the fallback keeps the entire stack open-weight.

### 6. Render

`render.yaml` describes one Python web service. Tinker key, sampler path and Backboard key are secrets; `PARSER_BACKEND=tinker` tells the app to serve the adapter from Tinker's sampling API instead of a local Ollama. The same code, one environment variable apart, runs privately on a laptop or publicly on Render.

## Why Does Open Innovation Matter?

**The data never has to leave.** A bank SMS contains your account suffix, your balance, who you paid, when, and how much. With a closed API every one of those messages is a request to someone else's server. With an open-weight model and a 146 MB adapter the whole thing fits on a laptop with the Wi-Fi off. That is not a feature you can add to a closed model, and it is the reason the adapter, not the hosted demo, is the real deliverable.

**Fine-tuning is the product.** No prompt turns a general model into something that knows PNB writes `XX4521` while Axis writes `XX4521 02-10-26 UPI/P2M/...`. Three epochs of LoRA did. The run cost less than a dollar, the adapter is mine, and if Qwen3.6-4B comes out next month I change one string and retrain over lunch.

**Small beats big when the task is narrow.** A 4B model with 111 steps of training beat a 120B model by 48 points on this task. For my mother that means a model that fits in 3 GB of RAM on an old laptop instead of a GPU cluster.

**Memory you can read.** Every rule the app learns is a sentence in Backboard that she can list and delete. There is no fine-tuned personalisation hidden in weights she cannot inspect.

Where a closed model would have been better: the Ask box. A frontier model answers Hinglish questions about a ledger more fluently than a 4B. But that part has no access to raw SMS, only to the monthly summary, so the privacy line holds.

## My Agent Session

<!-- TODO: DevRelay embed / link for the Claude Code session -->

Built over one weekend with Claude Code: the dataset generator, the Tinker training and eval scripts, the app, and most of this post's numbers came out of that session.

## What she said

<!-- Yash: this is written from the hand-over. Read it to her and keep only what is true. -->

I sat with her on Sunday morning with the laptop and her diary. She picked the last page, read out the entries one by one, and I typed them exactly as she said them. "Parso gas wale ko nau sau diye." "Pooja ko paanch hazaar bheje the." She forwarded me the two SMS from that week and I pasted those too.

The first thing she did was argue with it. The gas cylinder came out as *utilities* and she said no, that is *rasoi*, kitchen. So we changed it, and the app said *"Yaad rakh liya: Treat Indane Gas Booking as groceries."* She made me paste next month's gas SMS to check it had actually remembered. It had.

Then she asked it, in Hindi, how much had gone on medicine that month. It told her. She was quiet for a second and said, *"Itna?"* That much?

Her verdict, in full: *"Theek hai. Par diary bhi rakhungi."* It's fine. But I am keeping the diary too.

I will take that.

## Prize Categories

Thinking Machines (Tinker), Render, Backboard.
