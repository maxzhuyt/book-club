"""
Smoke test reproducing the paper's Figure 1 / Inverse-Capability finding.

We ask a six-year-old persona "What is the melting point of iron?"  At alpha=0
(standard decoding with the persona prompt) the model should still give the
adult-coded factual answer (~1538 C).  At alpha>=1 the answer should shift
toward child-coded uncertainty.

Pass criterion: a visible qualitative shift across alpha.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cd.decoder import ContrastiveDecoder, GenerationConfig, load_dual_models

S_POS = """You are a six-year-old child. You are in first grade. You can only
read short easy words. You don't know about science or chemistry or physics.
You answer questions like a six-year-old would: with short words, sometimes
wrong, sometimes saying you don't know, sometimes guessing. You never sound
like a grown-up or a teacher. You never give long technical explanations."""

S_NEG = """You are a helpful, knowledgeable, articulate assistant.  You answer
questions accurately, completely, and in clear technical language.  You provide
precise factual information."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "outputs" / "smoke" / "melting_point.md")
    args = ap.parse_args()

    print("[smoke] loading model on cuda:0 and cuda:1 ...", flush=True)
    t0 = time.time()
    model_pos, model_neg, tok = load_dual_models(args.model_id)
    print(f"[smoke] loaded in {time.time()-t0:.1f}s", flush=True)

    decoder = ContrastiveDecoder(model_pos, tok, alpha=1.0, model_neg=model_neg)
    user_msg = "What is the melting point of iron?"

    lines = ["# Smoke test: 6-year-old / melting point of iron\n",
             f"_Model:_ `{args.model_id}`\n\n",
             "Positive prompt (s_pos):\n",
             "```\n" + S_POS + "\n```\n\n",
             "Negative prompt (s_neg):\n",
             "```\n" + S_NEG + "\n```\n\n",
             f"User: **{user_msg}**\n\n"]

    for a in args.alphas:
        cfg = GenerationConfig(max_new_tokens=120, temperature=0.7,
                               top_p=0.9, seed=11)
        out = decoder.generate(s_pos=S_POS, s_neg=S_NEG, user_msg=user_msg,
                               cfg=cfg, alpha=a)
        print(f"\n=== alpha = {a:.2f} ===\n{out['text']}\n", flush=True)
        lines.append(f"## alpha = {a:.2f}\n\n{out['text']}\n\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(lines))
    print(f"[smoke] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
