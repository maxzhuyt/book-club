"""
Single-model contrastive decoding on the system prompt.

Implements Dong, Hu, Hui & Collier (2026) Eq. 5 (positive-negative formulation):

    z_pos = model_logits(s_pos, u, x_<t)
    z_neg = model_logits(s_neg, u, x_<t)
    z_cd  = z_pos + alpha * (z_pos - z_neg)
    x_t   ~ softmax(z_cd / T)   with top-p truncation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


@dataclass
class GenerationConfig:
    max_new_tokens: int = 800
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    seed: Optional[int] = None


class ContrastiveDecoder:
    """
    Two-branch contrastive decoder.  If model_neg is given, the negative branch
    runs on a separate model (and typically a separate GPU) in parallel with
    the positive branch — kernels launched on different CUDA devices execute
    concurrently in PyTorch, so this roughly halves wall-clock vs running both
    passes on a single GPU.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        alpha: float = 1.0,
        model_neg: Optional[AutoModelForCausalLM] = None,
    ):
        self.model_pos = model
        self.model_neg = model_neg if model_neg is not None else model
        self.tok = tokenizer
        self.alpha = float(alpha)
        self.device_pos = next(self.model_pos.parameters()).device
        self.device_neg = next(self.model_neg.parameters()).device
        # convenience alias for callers / existing code
        self.model = self.model_pos
        self.device = self.device_pos
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id

    # ---------------------------------------------------------------- helpers

    def _encode_chat(self, system: str, user: str, device) -> torch.Tensor:
        """Build an open-ended chat sequence ending at the assistant cue."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs = dict(add_generation_prompt=True, return_tensors="pt")
        # Qwen3 chat template supports enable_thinking; we want pure persona
        # voice, not <think>...</think> reasoning chains.
        try:
            result = self.tok.apply_chat_template(
                messages, enable_thinking=False, **kwargs,
            )
        except (TypeError, ValueError):
            # Older / non-Qwen3 tokenizers reject the kwarg or the system role
            try:
                result = self.tok.apply_chat_template(messages, **kwargs)
            except (TypeError, ValueError):
                # Some templates (e.g. older Gemma) don't accept system role;
                # fold system into the user message.
                merged = (system + "\n\n" + user) if system else user
                result = self.tok.apply_chat_template(
                    [{"role": "user", "content": merged}], **kwargs,
                )
        # Different tokenizers return different shapes:
        # - bare tensor (Qwen2.5, Qwen3)
        # - BatchEncoding / dict with 'input_ids' (Gemma3 under transformers 5.x)
        if hasattr(result, "input_ids"):
            ids = result.input_ids
        elif isinstance(result, dict) and "input_ids" in result:
            ids = result["input_ids"]
        else:
            ids = result
        return ids.to(device)

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Temperature + nucleus sampling on a (1, V) tensor of logits."""
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / temperature
        # Nucleus
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        cutoff = cum > top_p
        # shift right: always keep the top-1
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        # Re-sort back into vocab order
        full = torch.full_like(logits, float("-inf"))
        full.scatter_(-1, sorted_idx, sorted_logits)
        probs = F.softmax(full, dim=-1)
        token = torch.multinomial(probs, num_samples=1, generator=generator)
        return token

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor, prev_tokens: list[int], penalty: float
    ) -> torch.Tensor:
        if penalty == 1.0 or not prev_tokens:
            return logits
        unique = torch.tensor(list(set(prev_tokens)), device=logits.device, dtype=torch.long)
        scores = logits.index_select(-1, unique)
        scores = torch.where(scores < 0, scores * penalty, scores / penalty)
        out = logits.clone()
        out.index_copy_(-1, unique, scores)
        return out

    # --------------------------------------------------------------- generate

    @torch.inference_mode()
    def generate(
        self,
        s_pos: str,
        s_neg: str,
        user_msg: str,
        cfg: Optional[GenerationConfig] = None,
        alpha: Optional[float] = None,
        stream_callback=None,
    ) -> dict:
        """
        Generate the assistant turn under contrastive decoding.

        Returns a dict with 'text' (decoded string), 'tokens' (list[int]) and
        'alpha' (the alpha used).
        """
        cfg = cfg or GenerationConfig()
        a = self.alpha if alpha is None else float(alpha)
        gen = None
        if cfg.seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(cfg.seed))

        pos_ids = self._encode_chat(s_pos, user_msg, self.device_pos)
        neg_ids = self._encode_chat(s_neg, user_msg, self.device_neg)

        # Prefill both branches.  When the two models live on different GPUs,
        # the two forward calls launch async kernels on independent devices
        # and overlap; otherwise they serialize on the same device.
        # Pass past_key_values=None so the model constructs the right cache
        # type (DynamicCache for standard attention, HybridCache for models
        # with mixed linear/standard layers like Qwen3.5).
        pos_out = self.model_pos(
            input_ids=pos_ids,
            past_key_values=None,
            use_cache=True,
        )
        neg_out = self.model_neg(
            input_ids=neg_ids,
            past_key_values=None,
            use_cache=True,
        )
        # Move neg logits to the pos device for the contrastive combination.
        z_pos = pos_out.logits[:, -1, :].float()
        z_neg = neg_out.logits[:, -1, :].float().to(self.device_pos)
        pos_cache = pos_out.past_key_values
        neg_cache = neg_out.past_key_values

        eos_ids = set()
        if self.tok.eos_token_id is not None:
            eos_ids.add(int(self.tok.eos_token_id))
        # Qwen-2.5 also uses <|im_end|>
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end >= 0:
            eos_ids.add(int(im_end))

        generated: list[int] = []
        for step in range(cfg.max_new_tokens):
            z_cd = z_pos + a * (z_pos - z_neg)
            z_cd = self._apply_repetition_penalty(
                z_cd, generated, cfg.repetition_penalty
            )

            tok = self._sample(z_cd, cfg.temperature, cfg.top_p, generator=gen)
            tok_id = int(tok.item())
            if tok_id in eos_ids:
                break
            generated.append(tok_id)
            if stream_callback is not None:
                stream_callback(tok_id)

            tok_view = tok.view(1, 1)
            tok_pos = tok_view.to(self.device_pos)
            tok_neg = tok_view.to(self.device_neg)
            # Launch both forwards back-to-back; they overlap when devices differ.
            pos_step = self.model_pos(
                input_ids=tok_pos,
                past_key_values=pos_cache,
                use_cache=True,
            )
            neg_step = self.model_neg(
                input_ids=tok_neg,
                past_key_values=neg_cache,
                use_cache=True,
            )
            z_pos = pos_step.logits[:, -1, :].float()
            z_neg = neg_step.logits[:, -1, :].float().to(self.device_pos)
            pos_cache = pos_step.past_key_values
            neg_cache = neg_step.past_key_values

        text = self.tok.decode(generated, skip_special_tokens=True).strip()
        # Free caches eagerly
        del pos_cache, neg_cache, pos_out, neg_out
        torch.cuda.empty_cache()
        return {"text": text, "tokens": generated, "alpha": a}


def load_model(model_id: str, dtype=torch.bfloat16, device_map: str = "auto"):
    """Convenience loader."""
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    return model, tok


def load_dual_models(
    model_id: str,
    dtype=torch.bfloat16,
    device_pos: str = "cuda:0",
    device_neg: str = "cuda:1",
):
    """
    Load two copies of the same model on two GPUs so the positive and negative
    branches of contrastive decoding run in parallel.
    """
    tok = AutoTokenizer.from_pretrained(model_id)
    model_pos = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map={"": device_pos},
    )
    model_neg = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map={"": device_neg},
    )
    model_pos.eval()
    model_neg.eval()
    return model_pos, model_neg, tok
