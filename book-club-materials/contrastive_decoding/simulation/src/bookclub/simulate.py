"""
Four-phase book-club simulation runner.

Invariant: every persona's Phase-1 private review is generated BEFORE any
persona is shown any peer review.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cd.decoder import ContrastiveDecoder, GenerationConfig, load_dual_models
from cd import prompts as prompt_lib
from cd.alpha import compute_alpha
from bookclub.cast import Persona, load_cast, add_six_year_old, speaker_tag


# Phase-1 generation budgets are persona-aware: terse personas get fewer
# tokens, verbose personas get more.
def phase1_max_new_tokens(p: Persona) -> int:
    target = p.avg_review_words
    # ~1.4 tokens/word; clamp to a sensible window.
    return max(120, min(900, int(target * 1.4)))


def discussion_max_new_tokens(p: Persona) -> int:
    target = max(60, min(400, p.avg_review_words // 4))
    return int(target * 1.4) + 40


def reflection_max_new_tokens(p: Persona) -> int:
    return 220


# -----------------------------------------------------------------------------


def run_simulation(
    *,
    personas_dir: Path,
    output_dir: Path,
    model_id: str,
    story_path: Path,
    story_title: str,
    seed: int = 7,
    n_discussion_rounds: int = 2,
    include_six_year_old: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)

    # ---- load
    print(f"[load] model: {model_id} on cuda:0 and cuda:1", flush=True)
    t0 = time.time()
    model_pos, model_neg, tok = load_dual_models(model_id)
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)

    cast = load_cast(personas_dir)
    if include_six_year_old:
        cast = add_six_year_old(cast, prompt_lib.SIX_YEAR_OLD_POS)

    story_text = story_path.read_text()
    decoder = ContrastiveDecoder(model_pos, tok, alpha=1.0, model_neg=model_neg)

    # Manifest
    manifest = {
        "model_id": model_id,
        "seed": seed,
        "story_title": story_title,
        "personas": [
            {
                "short_id": p.short_id,
                "user_id": p.user_id,
                "label": p.label,
                "alpha": {
                    "phase1": compute_alpha(p.summary(), 1),
                    "phase3": compute_alpha(p.summary(), 3),
                    "phase4": compute_alpha(p.summary(), 4),
                },
                "six_year_old": p.six_year_old,
                "n_books": p.n_books,
                "avg_review_words": p.avg_review_words,
            }
            for p in cast
        ],
        "n_discussion_rounds": n_discussion_rounds,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ---- Phase 1: private stance (INDEPENDENT, no peer info) ----
    print("\n========== PHASE 1: PRIVATE STANCE ==========", flush=True)
    s_neg_default = prompt_lib.build_negative_prompt(phase=1)
    phase1_msg = prompt_lib.phase1_user_message(story_text, story_title)

    private_reviews: dict[str, str] = {}    # short_id -> text
    for p in cast:
        alpha = compute_alpha(p.summary(), phase=1)
        cfg = GenerationConfig(
            max_new_tokens=phase1_max_new_tokens(p),
            temperature=0.75,
            top_p=0.9,
            seed=seed + hash(p.user_id) % 10_000,
        )
        print(f"\n[Phase 1] {speaker_tag(p)}  alpha={alpha:.2f}  "
              f"max_new={cfg.max_new_tokens}", flush=True)
        t0 = time.time()
        out = decoder.generate(
            s_pos=p.s_pos,
            s_neg=s_neg_default,
            user_msg=phase1_msg,
            cfg=cfg,
            alpha=alpha,
        )
        dt = time.time() - t0
        tps = len(out["tokens"]) / max(dt, 1e-6)
        print(f"  -> {len(out['tokens'])} toks in {dt:.1f}s  ({tps:.1f} tok/s)", flush=True)
        private_reviews[p.short_id] = out["text"]
        print(out["text"][:400] + ("..." if len(out["text"]) > 400 else ""), flush=True)

    # Save Phase-1
    p1_payload = {p.short_id: {"speaker": speaker_tag(p),
                                "review": private_reviews[p.short_id]}
                  for p in cast}
    (transcripts_dir / "phase1_private.json").write_text(
        json.dumps(p1_payload, indent=2, ensure_ascii=False)
    )

    # ---- Phase 2: stance broadcast + reactions ----
    print("\n========== PHASE 2: BROADCAST & REACTIONS ==========", flush=True)
    labeled_peers = {speaker_tag(p): private_reviews[p.short_id] for p in cast}

    phase2_reactions: dict[str, str] = {}
    for p in cast:
        # exclude self from the peer block
        peers = {k: v for k, v in labeled_peers.items() if not k.startswith(p.short_id + " ")}
        msg = prompt_lib.phase2_user_message(story_title, peers)
        alpha = compute_alpha(p.summary(), phase=2)
        cfg = GenerationConfig(
            max_new_tokens=discussion_max_new_tokens(p),
            temperature=0.75,
            top_p=0.9,
            seed=seed + 1000 + hash(p.user_id) % 10_000,
        )
        print(f"\n[Phase 2] {speaker_tag(p)}  alpha={alpha:.2f}", flush=True)
        t0 = time.time()
        out = decoder.generate(
            s_pos=p.s_pos,
            s_neg=prompt_lib.build_negative_prompt(phase=2),
            user_msg=msg,
            cfg=cfg,
            alpha=alpha,
        )
        dt = time.time() - t0
        print(f"  -> {len(out['tokens'])} toks in {dt:.1f}s", flush=True)
        phase2_reactions[p.short_id] = out["text"]
        print(out["text"][:300] + ("..." if len(out["text"]) > 300 else ""), flush=True)

    p2_payload = {p.short_id: {"speaker": speaker_tag(p),
                                "reaction": phase2_reactions[p.short_id]}
                  for p in cast}
    (transcripts_dir / "phase2_reactions.json").write_text(
        json.dumps(p2_payload, indent=2, ensure_ascii=False)
    )

    # Build accumulated discussion history (used in Phase 3)
    history: list[dict] = []
    for p in cast:
        history.append({
            "speaker": speaker_tag(p) + " [Phase 1 private review]",
            "text": private_reviews[p.short_id],
        })
    for p in cast:
        history.append({
            "speaker": speaker_tag(p) + " [Phase 2 reaction]",
            "text": phase2_reactions[p.short_id],
        })

    # ---- Phase 3: moderated discussion ----
    print("\n========== PHASE 3: MODERATED DISCUSSION ==========", flush=True)
    moderator_lines = [
        # Round 1: focus on disagreement & specificity
        ("The moderator (an outside facilitator) says: I notice the group has "
         "shared first impressions and reactions. Now I want each of you, in "
         "turn, to engage DIRECTLY with the readers whose views differ most "
         "from your own. Quote them. Tell us where you think they are wrong, "
         "or right, or interestingly partial. Do NOT be polite for politeness' "
         "sake."),
        # Round 2: anti-conformity check
        ("The moderator says: We are nearing the end. Are you genuinely "
         "agreeing with the emerging tone of this discussion, or going along? "
         "If you secretly disagree, this is the moment to say it. If your "
         "Phase-1 private review is at odds with what you have just been "
         "saying in the group, name the gap and pick a side."),
    ]
    rounds: list[list[dict]] = []
    for r in range(n_discussion_rounds):
        addendum = moderator_lines[min(r, len(moderator_lines) - 1)]
        this_round: list[dict] = []
        # Rotate speaker order each round so no one is always first.
        order = cast[r % len(cast):] + cast[:r % len(cast)]
        for p in order:
            msg = prompt_lib.phase3_user_message(
                story_title, round_idx=r + 1, history=history,
                moderator_addendum=addendum,
            )
            alpha = compute_alpha(p.summary(), phase=3)
            cfg = GenerationConfig(
                max_new_tokens=discussion_max_new_tokens(p),
                temperature=0.8,
                top_p=0.9,
                seed=seed + 5000 + r * 100 + hash(p.user_id) % 10_000,
            )
            print(f"\n[Phase 3 R{r+1}] {speaker_tag(p)}  alpha={alpha:.2f}",
                  flush=True)
            t0 = time.time()
            out = decoder.generate(
                s_pos=p.s_pos,
                s_neg=prompt_lib.build_negative_prompt(phase=3),
                user_msg=msg,
                cfg=cfg,
                alpha=alpha,
            )
            dt = time.time() - t0
            print(f"  -> {len(out['tokens'])} toks in {dt:.1f}s", flush=True)
            print(out["text"][:300] + ("..." if len(out["text"]) > 300 else ""),
                  flush=True)
            entry = {
                "speaker": speaker_tag(p) + f" [Phase 3 R{r+1}]",
                "text": out["text"],
            }
            this_round.append(entry)
            history.append(entry)
        rounds.append(this_round)

    (transcripts_dir / "phase3_discussion.json").write_text(
        json.dumps([
            [{"speaker": e["speaker"], "text": e["text"]} for e in rr]
            for rr in rounds
        ], indent=2, ensure_ascii=False)
    )

    # ---- Phase 4: reflection ----
    print("\n========== PHASE 4: REFLECTION ==========", flush=True)
    reflections: dict[str, str] = {}
    for p in cast:
        msg = prompt_lib.phase4_user_message(story_title, history)
        alpha = compute_alpha(p.summary(), phase=4)
        cfg = GenerationConfig(
            max_new_tokens=reflection_max_new_tokens(p),
            temperature=0.7,
            top_p=0.9,
            seed=seed + 9000 + hash(p.user_id) % 10_000,
        )
        print(f"\n[Phase 4] {speaker_tag(p)}  alpha={alpha:.2f}", flush=True)
        t0 = time.time()
        out = decoder.generate(
            s_pos=p.s_pos,
            s_neg=prompt_lib.build_negative_prompt(phase=4),
            user_msg=msg,
            cfg=cfg,
            alpha=alpha,
        )
        dt = time.time() - t0
        print(f"  -> {len(out['tokens'])} toks in {dt:.1f}s", flush=True)
        print(out["text"][:300] + ("..." if len(out["text"]) > 300 else ""),
              flush=True)
        reflections[p.short_id] = out["text"]

    p4_payload = {p.short_id: {"speaker": speaker_tag(p),
                                "reflection": reflections[p.short_id]}
                  for p in cast}
    (transcripts_dir / "phase4_reflections.json").write_text(
        json.dumps(p4_payload, indent=2, ensure_ascii=False)
    )

    # ---- Write the consolidated Markdown transcript ----
    md = []
    md.append(f"# Book Club Discussion — {story_title}\n")
    md.append(f"_Base model:_ `{model_id}` (contrastive decoding "
              "on the system prompt, Dong et al. 2026)\n\n")
    md.append("## Cast\n\n")
    md.append("| Short ID | Persona | n_books | Avg review words | α phase 1/3/4 |\n")
    md.append("|---|---|---|---|---|\n")
    for p in cast:
        a1 = compute_alpha(p.summary(), 1)
        a3 = compute_alpha(p.summary(), 3)
        a4 = compute_alpha(p.summary(), 4)
        md.append(f"| {p.short_id} | {p.label} (user_{p.user_id}) | "
                  f"{p.n_books} | {p.avg_review_words} | "
                  f"{a1:.2f} / {a3:.2f} / {a4:.2f} |\n")

    md.append("\n---\n\n## Phase 1 — Private Reviews\n\n")
    for p in cast:
        md.append(f"### {speaker_tag(p)}\n\n{private_reviews[p.short_id]}\n\n")

    md.append("\n---\n\n## Phase 2 — Broadcast Reactions\n\n")
    for p in cast:
        md.append(f"### {speaker_tag(p)}\n\n{phase2_reactions[p.short_id]}\n\n")

    md.append("\n---\n\n## Phase 3 — Moderated Discussion\n\n")
    for r, rr in enumerate(rounds, start=1):
        md.append(f"### Round {r}\n\n")
        md.append(f"_Moderator:_ {moderator_lines[min(r-1, len(moderator_lines)-1)]}\n\n")
        for e in rr:
            md.append(f"**{e['speaker']}**\n\n{e['text']}\n\n")

    md.append("\n---\n\n## Phase 4 — Reflections\n\n")
    for p in cast:
        md.append(f"### {speaker_tag(p)}\n\n{reflections[p.short_id]}\n\n")

    (output_dir / "discussion_transcript.md").write_text("".join(md))

    print(f"\n[done] outputs written to {output_dir}", flush=True)
    return {
        "private_reviews": private_reviews,
        "phase2_reactions": phase2_reactions,
        "phase3_rounds": rounds,
        "phase4_reflections": reflections,
        "cast": [
            {"short_id": p.short_id, "label": p.label,
             "user_id": p.user_id, "six_year_old": p.six_year_old}
            for p in cast
        ],
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--personas-dir", type=Path,
                    default=Path(__file__).resolve().parents[3] / "personas")
    ap.add_argument("--output-dir", type=Path,
                    default=Path(__file__).resolve().parents[2] / "outputs")
    ap.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--story", type=str, default="story_1.md")
    ap.add_argument("--story-title", type=str, default="Love in the Limelight")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--no-child", action="store_true",
                    help="omit the 6-year-old persona")
    args = ap.parse_args()

    story_path = args.personas_dir / args.story
    run_simulation(
        personas_dir=args.personas_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        story_path=story_path,
        story_title=args.story_title,
        seed=args.seed,
        n_discussion_rounds=args.rounds,
        include_six_year_old=not args.no_child,
    )
