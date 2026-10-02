"""Thin Tinker sampling client that needs only `tinker` + `transformers` (no torch, no cookbook).

The Hugging Face chat template for each base model produces exactly the same tokens as the
tinker_cookbook renderer used in training (verified for qwen3_5_disable_thinking and
gpt_oss_low_reasoning), so the server can stay light enough for Render's free tier.
"""
from __future__ import annotations

import os
import re

STOP_TOKENS = {
    "qwen": ["<|im_end|>", "<|endoftext|>"],
    "gpt-oss": ["<|return|>", "<|call|>"],
}


class TinkerSampler:
    def __init__(self, *, base_model: str | None = None, model_path: str | None = None,
                 max_tokens: int = 200, temperature: float = 0.0):
        import tinker
        from transformers import AutoTokenizer

        self.service = tinker.ServiceClient()
        self.client = self.service.create_sampling_client(base_model=base_model, model_path=model_path)
        self.base_model = base_model or self.client.get_base_model()
        self.tok = AutoTokenizer.from_pretrained(os.environ.get("TOKENIZER_OVERRIDE") or self.base_model)
        family = "gpt-oss" if "gpt-oss" in self.base_model.lower() else "qwen"
        stop_ids = [i for i in (self.tok.convert_tokens_to_ids(t) for t in STOP_TOKENS[family]) if isinstance(i, int) and i >= 0]
        self.template_kwargs = {"enable_thinking": False} if family == "qwen" else {"reasoning_effort": "low"}
        self.family = family
        self.params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, stop=stop_ids)
        self._ModelInput = tinker.ModelInput

    def complete(self, system: str, user: str) -> str:
        ids = self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=True, return_dict=False, **self.template_kwargs)
        prompt = self._ModelInput.from_ints(list(ids))
        res = self.client.sample(prompt=prompt, num_samples=1, sampling_params=self.params).result()
        text = self.tok.decode(res.sequences[0].tokens, skip_special_tokens=False)
        return self._clean(text)

    def _clean(self, text: str) -> str:
        if self.family == "gpt-oss":
            # harmony format: keep the final channel only
            if "<|channel|>final<|message|>" in text:
                text = text.split("<|channel|>final<|message|>")[-1]
            text = re.sub(r"<\|[a-z_]+\|>", "", text)
        else:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
            text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
        return text.strip()
