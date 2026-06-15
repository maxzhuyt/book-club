"""Dual-copy contrastive decoding for Qwen3-32B (bf16).

Two full model copies, one per GPU (positive on cuda:0, negative on cuda:1).
The two branches differ only in their system prompt; per step we combine

    z_cd = z_pos + alpha * (z_pos - z_neg)

and sample, advancing the same chosen token in both branches' KV caches.
alpha = 0 short-circuits to plain decoding (negative branch never runs).

Why two copies on two GPUs instead of one model with two prompts: the forward
passes launch async on independent CUDA devices and overlap, roughly halving
wall-clock. This is the configuration proven at ~17.7 tok/s on 2xH200 in the
book-club runs (contrastive_decoding/simulation/src/cd/decoder.py).

Why Qwen3-32B and not DeepSeek-V4-Flash: V4-Flash's experts are FP4-packed and
the FP4 expert kernel is gated to Blackwell/SM100 — on H200 (SM90) it falls back
to a Triton path that produced corrupt logits. bf16 needs no FP8/FP4 kernels at
all and is device-correct by construction. (CD on V4-Flash via its API is also
impossible in principle: the API never exposes two synchronized logit vectors.)

The runner passes full OpenAI-style message lists per branch because the v3 read
pipeline is two-turn (pass-1 answer is part of the pass-2 context).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1200
    temperature: float = 0.75
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    seed: Optional[int] = None


class ContrastiveDecoder:
    def __init__(self, model_pos, model_neg, tokenizer, eos_ids: set[int]):
        self.model_pos = model_pos
        self.model_neg = model_neg
        self.tok = tokenizer
        self.eos_ids = eos_ids
        self.device_pos = next(model_pos.parameters()).device
        self.device_neg = next(model_neg.parameters()).device

    def _encode(self, messages: list[dict], device) -> torch.Tensor:
        ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt",
        )
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        elif isinstance(ids, dict):
            ids = ids["input_ids"]
        return ids.to(device)

    @staticmethod
    def _sample(logits, temperature, top_p, generator=None):
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        cutoff = cum > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        full = torch.full_like(logits, float("-inf"))
        full.scatter_(-1, sorted_idx, sorted_logits)
        probs = F.softmax(full, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator)

    @staticmethod
    def _apply_repetition_penalty(logits, prev_tokens, penalty):
        if penalty == 1.0 or not prev_tokens:
            return logits
        uniq = torch.tensor(list(set(prev_tokens)), device=logits.device,
                            dtype=torch.long)
        scores = logits.index_select(-1, uniq)
        scores = torch.where(scores < 0, scores * penalty, scores / penalty)
        out = logits.clone()
        out.index_copy_(-1, uniq, scores)
        return out

    @torch.inference_mode()
    def generate(self, messages_pos: list[dict], messages_neg: Optional[list[dict]],
                 cfg: GenerationConfig, alpha: float) -> dict:
        t0 = time.time()
        use_neg = alpha != 0.0

        pos_ids = self._encode(messages_pos, self.device_pos)
        pos_out = self.model_pos(input_ids=pos_ids, past_key_values=None, use_cache=True)
        z_pos = pos_out.logits[:, -1, :].float()
        pos_cache = pos_out.past_key_values

        neg_cache = None
        z_neg = None
        if use_neg:
            neg_ids = self._encode(messages_neg, self.device_neg)
            neg_out = self.model_neg(input_ids=neg_ids, past_key_values=None, use_cache=True)
            z_neg = neg_out.logits[:, -1, :].float().to(self.device_pos)
            neg_cache = neg_out.past_key_values

        gen = None
        if cfg.seed is not None:
            gen = torch.Generator(device=self.device_pos)
            gen.manual_seed(int(cfg.seed))

        generated: list[int] = []
        finish = "length"
        for _ in range(cfg.max_new_tokens):
            z_cd = z_pos + alpha * (z_pos - z_neg) if use_neg else z_pos
            z_cd = self._apply_repetition_penalty(z_cd, generated, cfg.repetition_penalty)
            tok = self._sample(z_cd, cfg.temperature, cfg.top_p, generator=gen)
            tok_id = int(tok.item())
            if tok_id in self.eos_ids:
                finish = "stop"
                break
            generated.append(tok_id)

            tok_view = tok.view(1, 1)
            pos_step = self.model_pos(input_ids=tok_view.to(self.device_pos),
                                      past_key_values=pos_cache, use_cache=True)
            z_pos = pos_step.logits[:, -1, :].float()
            pos_cache = pos_step.past_key_values
            if use_neg:
                neg_step = self.model_neg(input_ids=tok_view.to(self.device_neg),
                                          past_key_values=neg_cache, use_cache=True)
                z_neg = neg_step.logits[:, -1, :].float().to(self.device_pos)
                neg_cache = neg_step.past_key_values

        text = self.tok.decode(generated, skip_special_tokens=True).strip()
        had_think = "<think>" in text
        if had_think:
            text = THINK_RE.sub("", text).strip()
        dt = time.time() - t0
        del pos_cache, neg_cache
        torch.cuda.empty_cache()
        return {
            "text": text,
            "n_tokens": len(generated),
            "alpha": alpha,
            "finish": finish,
            "had_think": had_think,
            "seconds": round(dt, 1),
            "tok_per_s": round(len(generated) / dt, 2) if dt > 0 else 0.0,
        }


def load_dual_qwen(model_path: str, device_pos: str = "cuda:0",
                   device_neg: str = "cuda:1") -> ContrastiveDecoder:
    """Load two bf16 copies of the model, one pinned to each GPU."""
    tok = AutoTokenizer.from_pretrained(model_path)
    model_pos = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": device_pos})
    model_neg = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": device_neg})
    model_pos.eval()
    model_neg.eval()
    eos_ids: set[int] = set()
    if tok.eos_token_id is not None:
        eos_ids.add(int(tok.eos_token_id))
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        eos_ids.add(int(im_end))
    return ContrastiveDecoder(model_pos, model_neg, tok, eos_ids)
