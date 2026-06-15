"""
Quick smoke test for the 3 weekend-run target models.

For each model: load both branches (one copy per GPU), generate ~80 tokens
with the paper's six-year-old / melting-point toy at alpha=0 and alpha=1,
and check for <think>...</think> leakage (Qwen3 / Qwen3.5 thinking mode).

Run all three (each in its own subprocess, to guarantee memory release):
    python3 scripts/smoke_models.py

Run just one (worker mode, used internally):
    python3 scripts/smoke_models.py --model-id Qwen/Qwen3.5-9B
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODELS = [
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3.5-9B",
    "google/gemma-3-12b-it",
]

S_POS = ("You are a six-year-old child. You only use small words. "
         "You don't know science.")
S_NEG = ("You are a knowledgeable, articulate assistant who gives precise "
         "factual answers.")
USER = "What is the melting point of iron?"


def run_one(model_id: str) -> None:
    """Worker: smoke-test a single model in this process."""
    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from cd.decoder import ContrastiveDecoder, GenerationConfig, load_dual_models

    print(f"\n========== {model_id} ==========", flush=True)
    m_pos, m_neg, tok = load_dual_models(model_id)
    dec = ContrastiveDecoder(m_pos, tok, alpha=1.0, model_neg=m_neg)
    for alpha in [0.0, 1.0]:
        out = dec.generate(
            s_pos=S_POS, s_neg=S_NEG, user_msg=USER,
            cfg=GenerationConfig(
                max_new_tokens=80, temperature=0.7,
                top_p=0.9, seed=11,
            ),
            alpha=alpha,
        )
        text = out["text"]
        has_think = "<think>" in text or "</think>" in text
        print(f"  alpha={alpha}: ({len(out['tokens'])} toks, "
              f"has_think={has_think})", flush=True)
        print(f"    {text[:300]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, default=None,
                    help="If set, smoke-test only this model in-process. "
                    "Otherwise dispatch all 3 via subprocesses.")
    args = ap.parse_args()

    if args.model_id is not None:
        run_one(args.model_id)
        return

    here = Path(__file__).resolve()
    for model_id in MODELS:
        result = subprocess.run(
            [sys.executable, str(here), "--model-id", model_id],
            check=False,
        )
        if result.returncode != 0:
            print(f"  (subprocess for {model_id} exited {result.returncode})",
                  flush=True)
    print("\n[smoke done]", flush=True)


if __name__ == "__main__":
    main()
