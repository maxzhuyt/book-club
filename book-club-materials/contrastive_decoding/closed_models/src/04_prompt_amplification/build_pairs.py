"""
Plan 04 — generate paired CD-vs-no-CD Phase-1 reviews for each persona.

For each persona, generate K samples on the same story:
  - text_cd     : ContrastiveDecoder.generate(..., alpha=alpha_persona)
  - text_no_cd  : ContrastiveDecoder.generate(..., alpha=0.0)
Both calls share the same seed so the only difference is the CD weight.
The alpha=0 path collapses CD to the positive-only baseline (no contrast),
which is what the post-training "assistant" prior produces.

Output layout:

    pairs/
    ├── manifest.json
    └── <short_id>/
        └── pair_<seed>.json
              {
                "seed": ..., "alpha_cd": ..., "alpha_no_cd": 0.0,
                "text_cd": "...", "text_no_cd": "..."
              }

CD code in simulation/src/cd is used as-is. No edits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "simulation" / "src"))

from bookclub.cast import load_cast, add_six_year_old, speaker_tag  # noqa: E402
from cd.decoder import (  # noqa: E402
    ContrastiveDecoder, GenerationConfig, load_dual_models,
)
from cd import prompts as prompt_lib  # noqa: E402
from cd.alpha import compute_alpha  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas-dir", type=Path,
                    default=REPO_ROOT / "personas")
    ap.add_argument("--story", type=str, required=True,
                    help="Held-out story filename inside personas-dir.")
    ap.add_argument("--story-title", type=str, required=True)
    ap.add_argument(
        "--pairs-dir", type=Path,
        default=Path(__file__).resolve().parent / "pairs",
    )
    ap.add_argument("--cd-model-id", type=str,
                    default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--num-pairs", type=int, default=3,
                    help="Pairs per persona.")
    ap.add_argument("--base-seed", type=int, default=20260515)
    ap.add_argument("--no-child", action="store_true")
    args = ap.parse_args()

    args.pairs_dir.mkdir(parents=True, exist_ok=True)
    story_path = args.personas_dir / args.story
    if not story_path.exists():
        raise SystemExit(f"Story not found: {story_path}")
    story_text = story_path.read_text()

    print(f"[plan04] loading {args.cd_model_id} on cuda:0 and cuda:1",
          flush=True)
    t0 = time.time()
    model_pos, model_neg, tok = load_dual_models(args.cd_model_id)
    print(f"[plan04] loaded in {time.time()-t0:.1f}s", flush=True)
    decoder = ContrastiveDecoder(model_pos, tok, alpha=1.0,
                                 model_neg=model_neg)

    cast = load_cast(args.personas_dir)
    if not args.no_child:
        cast = add_six_year_old(cast, prompt_lib.SIX_YEAR_OLD_POS)

    s_neg = prompt_lib.build_negative_prompt(phase=1)
    user_msg = prompt_lib.phase1_user_message(story_text, args.story_title)

    manifest = {
        "plan": "04_prompt_amplification",
        "step": "build_pairs",
        "cd_model_id": args.cd_model_id,
        "story": args.story,
        "story_title": args.story_title,
        "num_pairs": args.num_pairs,
        "base_seed": args.base_seed,
        "include_six_year_old": not args.no_child,
        "personas": [],
    }

    for p in cast:
        alpha = compute_alpha(p.summary(), phase=1)
        persona_dir = args.pairs_dir / p.short_id
        persona_dir.mkdir(exist_ok=True)
        target_tokens = max(120, min(900, int(p.avg_review_words * 1.4)))

        for k in range(args.num_pairs):
            pair_seed = args.base_seed + 7919 * k + hash(p.user_id) % 10_000
            cfg_cd = GenerationConfig(
                max_new_tokens=target_tokens,
                temperature=0.75, top_p=0.9, seed=pair_seed,
            )
            cfg_no = GenerationConfig(
                max_new_tokens=target_tokens,
                temperature=0.75, top_p=0.9, seed=pair_seed,
            )
            print(f"\n[plan04] {speaker_tag(p)} pair {k+1}/{args.num_pairs}  "
                  f"alpha_cd={alpha:.2f}", flush=True)
            t0 = time.time()
            out_cd = decoder.generate(
                s_pos=p.s_pos, s_neg=s_neg, user_msg=user_msg,
                cfg=cfg_cd, alpha=alpha,
            )
            t_cd = time.time() - t0
            t0 = time.time()
            out_no = decoder.generate(
                s_pos=p.s_pos, s_neg=s_neg, user_msg=user_msg,
                cfg=cfg_no, alpha=0.0,
            )
            t_no = time.time() - t0
            print(f"  cd  ({len(out_cd['tokens'])} tok / {t_cd:.1f}s); "
                  f"no  ({len(out_no['tokens'])} tok / {t_no:.1f}s)",
                  flush=True)

            (persona_dir / f"pair_{pair_seed}.json").write_text(json.dumps({
                "seed": pair_seed,
                "alpha_cd": alpha,
                "alpha_no_cd": 0.0,
                "text_cd": out_cd["text"],
                "text_no_cd": out_no["text"],
            }, indent=2, ensure_ascii=False))

        manifest["personas"].append({
            "short_id": p.short_id, "user_id": p.user_id,
            "label": p.label, "alpha_cd": alpha,
            "six_year_old": p.six_year_old,
        })

    (args.pairs_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(f"\n[plan04] wrote pairs to {args.pairs_dir}", flush=True)


if __name__ == "__main__":
    main()
