"""Bring the Tinker LoRA home: download -> PEFT adapter -> merged HF model -> GGUF -> Ollama.

  .venv/bin/python train/export_adapter.py                 # uses out/run.json sampler_path
  .venv/bin/python train/export_adapter.py --sampler-path tinker://...
  .venv/bin/python train/export_adapter.py --skip-gguf     # stop after merged HF model

Needs: git, cmake (brew install cmake) for llama.cpp quantize. Peaks at ~20 GB disk, ends at ~3 GB.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import BASE_MODEL, ROOT

OUT = ROOT / "out"
LLAMA = ROOT / "llama.cpp"


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler-path", default=os.environ.get("KHARCHA_SAMPLER_PATH"))
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--skip-gguf", action="store_true")
    ap.add_argument("--ollama-name", default="kharcha")
    ap.add_argument("--keep-intermediates", action="store_true")
    args = ap.parse_args()

    sampler_path = args.sampler_path
    if not sampler_path and (OUT / "run.json").exists():
        sampler_path = json.loads((OUT / "run.json").read_text())["sampler_path"]
    if not sampler_path:
        raise SystemExit("need --sampler-path")

    from tinker_cookbook import weights

    adapter_raw = OUT / "adapter_raw"
    peft_dir = OUT / "peft_adapter"
    merged_dir = OUT / "merged_hf"
    if not peft_dir.exists():
        print("1/5 downloading checkpoint from Tinker")
        adapter_dir = weights.download(tinker_path=sampler_path, output_dir=str(adapter_raw))
        print("2/5 building PEFT adapter")
        weights.build_lora_adapter(base_model=args.base_model, adapter_path=adapter_dir, output_path=str(peft_dir))
    else:
        print("1-2/5 peft adapter exists, skipping")

    if not (merged_dir / "config.json").exists():
        print("3/5 merging LoRA into base weights (downloads base model from HF, ~8 GB)")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        model = PeftModel.from_pretrained(base, str(peft_dir))
        model = model.merge_and_unload()
        model.save_pretrained(str(merged_dir), safe_serialization=True)
        AutoTokenizer.from_pretrained(args.base_model).save_pretrained(str(merged_dir))
    else:
        print("3/5 merged model exists, skipping")

    if args.skip_gguf:
        print("done (skipped GGUF). Serve with transformers+PEFT or vLLM.")
        return

    print("4/5 converting to GGUF with llama.cpp")
    if not LLAMA.exists():
        sh(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(LLAMA)])
    quant_bin = LLAMA / "build" / "bin" / "llama-quantize"
    if not quant_bin.exists():
        sh(["cmake", "-B", str(LLAMA / "build"), "-S", str(LLAMA), "-DGGML_METAL=ON"])
        sh(["cmake", "--build", str(LLAMA / "build"), "--target", "llama-quantize", "-j"])
    py = sys.executable
    sh([py, "-m", "pip", "install", "-q", "-r", str(LLAMA / "requirements" / "requirements-convert_hf_to_gguf.txt")])
    f16 = OUT / "kharcha-f16.gguf"
    if not f16.exists():
        sh([py, str(LLAMA / "convert_hf_to_gguf.py"), str(merged_dir), "--outfile", str(f16), "--outtype", "f16"])
    q = OUT / f"kharcha-{args.quant.lower()}.gguf"
    if not q.exists():
        sh([quant_bin, str(f16), str(q), args.quant])
    if not args.keep_intermediates:
        # disk hygiene on a 16 GB laptop: merged HF (~8 GB) and f16 GGUF (~8 GB) are rebuildable
        shutil.rmtree(merged_dir, ignore_errors=True)
        f16.unlink(missing_ok=True)
        print("removed merged_hf/ and f16 gguf (use --keep-intermediates to keep)")

    print("5/5 registering with Ollama")
    modelfile = ROOT / "Modelfile"
    text = modelfile.read_text().replace("{{GGUF}}", str(q))
    tmp = OUT / "Modelfile.resolved"
    tmp.write_text(text)
    if shutil.which("ollama"):
        sh(["ollama", "create", args.ollama_name, "-f", str(tmp)])
        print(f"\nTry:  ollama run {args.ollama_name} 'Sent Rs.340.00 From HDFC Bank A/C *4521 To SHARMA KIRANA STORE On 02/10/26 Ref 827364512345'")
    else:
        print("ollama not on PATH; run: ollama create", args.ollama_name, "-f", tmp)


if __name__ == "__main__":
    main()
