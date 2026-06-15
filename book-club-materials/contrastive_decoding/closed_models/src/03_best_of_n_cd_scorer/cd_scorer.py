"""
Plan 03 — CD-derived scorers for reranking closed-model candidates.

Two scorers:

* Score A — CD logprob delta.
    score_A(x) = mean_t [ log p_pos(x_t | x_<t) - log p_neg(x_t | x_<t) ]
  where logprobs come from the same dual-model setup the CD decoder uses
  (loaded via cd.decoder.load_dual_models).  This is the integral over the
  candidate of the per-token CD signal — exactly what z_pos+α(z_pos−z_neg)
  steers toward.

* Score B — embedding distance to a CD reference bank.
    score_B(x) = mean cosine similarity of x's embedding to the top-3
    nearest references for that persona / phase.
  Cheaper than Score A; requires a pre-built bank (we reuse Plan 01's bank).

CD decoder code in simulation/src/cd is NOT modified — we re-derive only the
chat-template encoding pattern used in ContrastiveDecoder._encode_chat so
that scoring sees the same tokenization the generator did.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------- encoding


def encode_chat_with_response(
    tok, system: str, user: str, response: str, device,
) -> tuple[torch.Tensor, int]:
    """
    Build [system, user, assistant(=response)] token sequence using the
    tokenizer's chat template. Returns (input_ids, prefix_len) where
    prefix_len is the number of tokens BEFORE the response begins.

    Matches the pattern used in ContrastiveDecoder._encode_chat for the
    system+user prefix; the response is appended as raw tokens (no special
    tokens) on the end. This keeps the candidate token stream identical to
    what CD generation would have produced.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs = dict(add_generation_prompt=True, return_tensors="pt")
    try:
        result = tok.apply_chat_template(
            messages, enable_thinking=False, **kwargs,
        )
    except (TypeError, ValueError):
        try:
            result = tok.apply_chat_template(messages, **kwargs)
        except (TypeError, ValueError):
            merged = (system + "\n\n" + user) if system else user
            result = tok.apply_chat_template(
                [{"role": "user", "content": merged}], **kwargs,
            )
    if hasattr(result, "input_ids"):
        prefix_ids = result.input_ids
    elif isinstance(result, dict) and "input_ids" in result:
        prefix_ids = result["input_ids"]
    else:
        prefix_ids = result
    prefix_ids = prefix_ids.to(device)

    response_ids = tok(
        response, add_special_tokens=False, return_tensors="pt",
    ).input_ids.to(device)

    full = torch.cat([prefix_ids, response_ids], dim=1)
    prefix_len = prefix_ids.shape[1]
    return full, prefix_len


@torch.inference_mode()
def teacher_force_logprobs(
    model, tok, system: str, user: str, response: str,
) -> torch.Tensor:
    """
    Per-token logprobs of `response` tokens conditioned on the [system, user]
    chat prefix. Returns a 1-D tensor of shape (response_len,) on the model's
    device.
    """
    device = next(model.parameters()).device
    full_ids, prefix_len = encode_chat_with_response(
        tok, system, user, response, device,
    )
    if full_ids.shape[1] <= prefix_len:
        # Empty response — return zero-length tensor
        return torch.zeros(0, device=device)
    out = model(input_ids=full_ids, use_cache=False)
    logits = out.logits  # (1, T, V)
    # logits at position t predict token t+1, so for response tokens at
    # positions [prefix_len, T), the predictive logits are at
    # [prefix_len-1, T-1).
    target_ids = full_ids[:, prefix_len:]                   # (1, R)
    pred_logits = logits[:, prefix_len - 1 : -1, :].float() # (1, R, V)
    logp = F.log_softmax(pred_logits, dim=-1)
    token_logp = logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # (1, R)
    return token_logp.squeeze(0)


# ----------------------------------------------------------------- Score A


@dataclass
class ScoreAResult:
    delta_mean: float       # mean per-token (log p_pos - log p_neg)
    delta_sum: float        # sum over response
    n_tokens: int
    lp_pos_mean: float
    lp_neg_mean: float


class CDLogprobScorer:
    """
    Score A — uses model_pos, model_neg loaded via
    cd.decoder.load_dual_models (or any equivalent). Same chat template.
    """

    def __init__(self, model_pos, model_neg, tok):
        self.model_pos = model_pos
        self.model_neg = model_neg
        self.tok = tok

    def score(
        self, *,
        s_pos: str, s_neg: str, user_msg: str, response: str,
    ) -> ScoreAResult:
        lp_pos = teacher_force_logprobs(
            self.model_pos, self.tok, s_pos, user_msg, response,
        )
        lp_neg = teacher_force_logprobs(
            self.model_neg, self.tok, s_neg, user_msg, response,
        )
        # Align lengths (encoders should produce identical response tokens,
        # but be defensive — fall back to common length).
        n = min(lp_pos.shape[0], lp_neg.shape[0])
        if n == 0:
            return ScoreAResult(
                delta_mean=0.0, delta_sum=0.0, n_tokens=0,
                lp_pos_mean=0.0, lp_neg_mean=0.0,
            )
        lp_pos_n = lp_pos[:n]
        lp_neg_n = lp_neg[:n].to(lp_pos_n.device)
        delta = lp_pos_n - lp_neg_n
        return ScoreAResult(
            delta_mean=float(delta.mean().item()),
            delta_sum=float(delta.sum().item()),
            n_tokens=int(n),
            lp_pos_mean=float(lp_pos_n.mean().item()),
            lp_neg_mean=float(lp_neg_n.mean().item()),
        )


def load_score_a(model_id: str, device_pos: str = "cuda:0",
                 device_neg: str = "cuda:1") -> CDLogprobScorer:
    """Load model_pos and model_neg on two GPUs via the CD loader."""
    import sys
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(REPO_ROOT / "simulation" / "src"))
    from cd.decoder import load_dual_models  # noqa: E402
    model_pos, model_neg, tok = load_dual_models(
        model_id, device_pos=device_pos, device_neg=device_neg,
    )
    return CDLogprobScorer(model_pos, model_neg, tok)


# ----------------------------------------------------------------- Score B


class EmbeddingScorer:
    """
    Score B — cosine similarity to a CD reference bank, using a small
    sentence-transformers embedder. Reuses the Plan 01 bank directly.
    """

    def __init__(self, bank_dir: Path,
                 embedder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cuda:0", top_k: int = 3):
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(embedder_id, device=device)
        self.top_k = top_k
        self.bank_dir = bank_dir
        self._cache: dict[tuple[str, str], torch.Tensor] = {}

    def _references_for(self, short_id: str, phase: str) -> list[str]:
        # Pull from the requested phase; if missing, use any phase as fallback.
        persona_dir = self.bank_dir / short_id
        if not persona_dir.exists():
            return []
        candidates: list[str] = []
        primary = persona_dir / f"{phase}.json"
        if primary.exists():
            entries = json.loads(primary.read_text())
            candidates.extend(e["text"] for e in entries)
        if not candidates:
            for sibling in persona_dir.glob("phase*.json"):
                entries = json.loads(sibling.read_text())
                candidates.extend(e["text"] for e in entries)
        return candidates

    def _ref_emb(self, short_id: str, phase: str) -> Optional[torch.Tensor]:
        key = (short_id, phase)
        if key in self._cache:
            return self._cache[key]
        refs = self._references_for(short_id, phase)
        if not refs:
            self._cache[key] = None  # type: ignore[assignment]
            return None
        emb = self.embedder.encode(
            refs, convert_to_tensor=True, normalize_embeddings=True,
        )
        self._cache[key] = emb
        return emb

    def score(self, *, short_id: str, phase: str, response: str) -> float:
        refs = self._ref_emb(short_id, phase)
        if refs is None or refs.shape[0] == 0:
            return 0.0
        cand = self.embedder.encode(
            [response], convert_to_tensor=True, normalize_embeddings=True,
        )
        sims = (cand @ refs.T).squeeze(0)  # (n_refs,)
        k = min(self.top_k, sims.shape[0])
        topk = torch.topk(sims, k).values
        return float(topk.mean().item())
