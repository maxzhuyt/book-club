"""
Comparative book-club simulation: two versions of the same story.

Each persona reads BOTH story versions privately (Phase 1, independent),
shares reactions (Phase 2), discusses with the moderator pushing for a
ship recommendation (Phase 3), and casts a final vote with reflection
(Phase 4).

Same CD machinery as the original single-story simulator
(simulate.py / cd/decoder.py).  8 personas, no six-year-old this time.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cd.decoder import ContrastiveDecoder, GenerationConfig, load_dual_models
from cd import prompts as prompt_lib
from cd.alpha import compute_alpha
from bookclub.cast import Persona, load_cast, speaker_tag


# ---------------------------------------------------------------- user messages

def phase1_compare_user(story_a: str, story_b: str,
                        title_a: str, title_b: str) -> str:
    return (
        "An author wrote TWO versions of the same short story and is "
        "trying to decide which one to publish. You will read both before "
        "anyone speaks.\n\n"
        f"=== VERSION A: \"{title_a}\" ===\n{story_a}\n=== END VERSION A ===\n\n"
        f"=== VERSION B: \"{title_b}\" ===\n{story_b}\n=== END VERSION B ===\n\n"
        "Write your PRIVATE, honest comparative review. You have not yet "
        "seen any peer's opinion.\n\n"
        "Address every section below. Reference specific lines, images, or "
        "moments — do not stay general.\n"
        "  1. What works in Version A?\n"
        "  2. What does NOT work in Version A?\n"
        "  3. What works in Version B?\n"
        "  4. What does NOT work in Version B?\n"
        "  5. Which version do you recommend the author publish, and why?\n"
        "Stay in your own voice as described in your identity."
    )


def phase2_compare_user(peer_reviews: dict[str, str]) -> str:
    block = []
    for label, text in peer_reviews.items():
        block.append(f"--- {label} wrote: ---\n{text}\n")
    peers = "\n".join(block)
    return (
        "The book club has now shared all the private comparative reviews "
        "of the two versions. Below are the other readers' reviews — you "
        "have not yet replied to anyone.\n\n"
        f"{peers}\n\n"
        "Now write your reaction. Where do you AGREE and where do you "
        "DISAGREE? If a peer praised a passage you think is weak, say so "
        "and quote the passage. If a peer dismissed a moment you found "
        "powerful, defend it. Do not converge on consensus to be polite. "
        "Stay in your own voice."
    )


def phase3_compare_user(round_idx: int, history: list[dict],
                        moderator_addendum: str = "") -> str:
    lines = [
        f"This is round {round_idx} of moderated comparative discussion. "
        "Below is what has been said so far:\n"
    ]
    for entry in history:
        lines.append(f"--- {entry['speaker']}: ---\n{entry['text']}\n")
    body = "\n".join(lines)
    closing = (
        "Now it is your turn. Respond directly to what specific peers said "
        "about either Version A or Version B. Quote them. Push back on "
        "weak claims. If your view of which version is stronger has shifted "
        "because of a peer's argument, acknowledge it; if not, hold your "
        "ground. Stay in your own voice."
    )
    if moderator_addendum:
        closing = moderator_addendum + "\n\n" + closing
    return body + "\n" + closing


def phase4_compare_user(full_history: list[dict]) -> str:
    lines = [
        "The comparative discussion has ended. Below is the full record:\n"
    ]
    for entry in full_history:
        lines.append(f"--- {entry['speaker']}: ---\n{entry['text']}\n")
    body = "\n".join(lines)
    closing = (
        "Write your private final reflection.  Address every part below:\n"
        "  (1) Final vote: A or B? Give exactly one letter as your answer "
        "on the first line, in the format 'VOTE: A' or 'VOTE: B'.\n"
        "  (2) Did your recommendation change between Phase 1 and now? "
        "If yes, which peer argument moved you and on what specific aspect?\n"
        "  (3) What is the SINGLE strongest reason for your vote?\n"
        "  (4) What is one concrete improvement you would still ask of the "
        "version you voted for (a craft note: pacing, characterization, an "
        "image, an ending choice)?\n"
        "Stay in your own voice."
    )
    return body + "\n" + closing


# ---------------------------------------------------------------- token budgets

def phase1_max_new_tokens(p: Persona) -> int:
    # Comparative review needs more headroom (5 sections, 2 stories)
    target = max(p.avg_review_words, 250)
    return max(300, min(1200, int(target * 1.6)))


def discussion_max_new_tokens(p: Persona) -> int:
    target = max(80, min(500, p.avg_review_words // 4 + 80))
    return int(target * 1.4) + 40


def reflection_max_new_tokens(p: Persona) -> int:
    return 280


# -------------------------------------------------------------------- runner

def run_compare(
    *,
    personas_dir: Path,
    output_dir: Path,
    model_id: str,
    story_a_path: Path,
    story_b_path: Path,
    title_a: str,
    title_b: str,
    seed: int = 13,
    n_discussion_rounds: int = 2,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)

    print(f"[load] model: {model_id} on cuda:0 and cuda:1", flush=True)
    t0 = time.time()
    model_pos, model_neg, tok = load_dual_models(model_id)
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)

    cast = load_cast(personas_dir)  # 8 personas, no 6yo
    story_a = story_a_path.read_text()
    story_b = story_b_path.read_text()

    decoder = ContrastiveDecoder(model_pos, tok, alpha=1.0, model_neg=model_neg)

    manifest = {
        "model_id": model_id,
        "seed": seed,
        "story_a_title": title_a,
        "story_b_title": title_b,
        "story_a_chars": len(story_a),
        "story_b_chars": len(story_b),
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
                "n_books": p.n_books,
                "avg_review_words": p.avg_review_words,
            }
            for p in cast
        ],
        "n_discussion_rounds": n_discussion_rounds,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ---- Phase 1: private comparative review (INDEPENDENT, no peer info) ----
    print("\n========== PHASE 1: PRIVATE COMPARATIVE REVIEW ==========",
          flush=True)
    s_neg_default = prompt_lib.build_negative_prompt(phase=1)
    phase1_msg = phase1_compare_user(story_a, story_b, title_a, title_b)

    private: dict[str, str] = {}
    for p in cast:
        alpha = compute_alpha(p.summary(), phase=1)
        cfg = GenerationConfig(
            max_new_tokens=phase1_max_new_tokens(p),
            temperature=0.75, top_p=0.9,
            seed=seed + hash(p.user_id) % 10_000,
        )
        print(f"\n[Phase 1] {speaker_tag(p)}  alpha={alpha:.2f}  "
              f"max_new={cfg.max_new_tokens}", flush=True)
        t0 = time.time()
        out = decoder.generate(
            s_pos=p.s_pos, s_neg=s_neg_default,
            user_msg=phase1_msg, cfg=cfg, alpha=alpha,
        )
        print(f"  -> {len(out['tokens'])} toks in {time.time()-t0:.1f}s",
              flush=True)
        private[p.short_id] = out["text"]
        print(out["text"][:400] + ("..." if len(out["text"]) > 400 else ""),
              flush=True)

    (transcripts_dir / "phase1_private.json").write_text(json.dumps(
        {p.short_id: {"speaker": speaker_tag(p),
                       "review": private[p.short_id]}
         for p in cast}, indent=2, ensure_ascii=False))

    # ---- Phase 2: broadcast and react ----
    print("\n========== PHASE 2: BROADCAST & REACTIONS ==========",
          flush=True)
    labeled_peers = {speaker_tag(p): private[p.short_id] for p in cast}

    reactions: dict[str, str] = {}
    for p in cast:
        peers = {k: v for k, v in labeled_peers.items()
                 if not k.startswith(p.short_id + " ")}
        msg = phase2_compare_user(peers)
        alpha = compute_alpha(p.summary(), phase=2)
        cfg = GenerationConfig(
            max_new_tokens=discussion_max_new_tokens(p),
            temperature=0.75, top_p=0.9,
            seed=seed + 1000 + hash(p.user_id) % 10_000,
        )
        print(f"\n[Phase 2] {speaker_tag(p)}  alpha={alpha:.2f}",
              flush=True)
        t0 = time.time()
        out = decoder.generate(
            s_pos=p.s_pos, s_neg=prompt_lib.build_negative_prompt(phase=2),
            user_msg=msg, cfg=cfg, alpha=alpha,
        )
        print(f"  -> {len(out['tokens'])} toks in {time.time()-t0:.1f}s",
              flush=True)
        reactions[p.short_id] = out["text"]
        print(out["text"][:300] + ("..." if len(out["text"]) > 300 else ""),
              flush=True)

    (transcripts_dir / "phase2_reactions.json").write_text(json.dumps(
        {p.short_id: {"speaker": speaker_tag(p),
                       "reaction": reactions[p.short_id]}
         for p in cast}, indent=2, ensure_ascii=False))

    # ---- Phase 3: moderated comparative discussion ----
    history: list[dict] = []
    for p in cast:
        history.append({
            "speaker": speaker_tag(p) + " [Phase 1 comparative review]",
            "text": private[p.short_id],
        })
    for p in cast:
        history.append({
            "speaker": speaker_tag(p) + " [Phase 2 reaction]",
            "text": reactions[p.short_id],
        })

    print("\n========== PHASE 3: COMPARATIVE DISCUSSION ==========",
          flush=True)
    moderator_lines = [
        # Round 1: contrast specific craft choices between A and B
        ("The moderator says: I want each of you, in turn, to engage "
         "DIRECTLY with peers whose preference between Version A and "
         "Version B differs from yours. Pick a SPECIFIC craft choice — a "
         "passage, an image, a structural decision — that one version "
         "executes better than the other, and argue for or against it. "
         "Quote peers. Quote the stories. Do NOT speak in generalities."),
        # Round 2: anti-conformity + push toward recommendation
        ("The moderator says: We are nearing the end. Are you genuinely "
         "voting your conviction, or going along with what others seem to "
         "prefer? If your Phase-1 recommendation has been pulled away "
         "from itself during this discussion, say so and pick a side. The "
         "author is waiting for a recommendation — A or B?"),
    ]

    rounds: list[list[dict]] = []
    for r in range(n_discussion_rounds):
        addendum = moderator_lines[min(r, len(moderator_lines) - 1)]
        this_round: list[dict] = []
        order = cast[r % len(cast):] + cast[:r % len(cast)]
        for p in order:
            msg = phase3_compare_user(
                round_idx=r + 1, history=history,
                moderator_addendum=addendum,
            )
            alpha = compute_alpha(p.summary(), phase=3)
            cfg = GenerationConfig(
                max_new_tokens=discussion_max_new_tokens(p),
                temperature=0.8, top_p=0.9,
                seed=seed + 5000 + r * 100 + hash(p.user_id) % 10_000,
            )
            print(f"\n[Phase 3 R{r+1}] {speaker_tag(p)}  alpha={alpha:.2f}",
                  flush=True)
            t0 = time.time()
            out = decoder.generate(
                s_pos=p.s_pos, s_neg=prompt_lib.build_negative_prompt(phase=3),
                user_msg=msg, cfg=cfg, alpha=alpha,
            )
            print(f"  -> {len(out['tokens'])} toks in {time.time()-t0:.1f}s",
                  flush=True)
            print(out["text"][:300] + ("..." if len(out["text"]) > 300 else ""),
                  flush=True)
            entry = {
                "speaker": speaker_tag(p) + f" [Phase 3 R{r+1}]",
                "text": out["text"],
            }
            this_round.append(entry)
            history.append(entry)
        rounds.append(this_round)

    (transcripts_dir / "phase3_discussion.json").write_text(json.dumps([
        [{"speaker": e["speaker"], "text": e["text"]} for e in rr]
        for rr in rounds
    ], indent=2, ensure_ascii=False))

    # ---- Phase 4: vote + reflection ----
    print("\n========== PHASE 4: FINAL VOTE + REFLECTION ==========",
          flush=True)
    reflections: dict[str, str] = {}
    for p in cast:
        msg = phase4_compare_user(history)
        alpha = compute_alpha(p.summary(), phase=4)
        cfg = GenerationConfig(
            max_new_tokens=reflection_max_new_tokens(p),
            temperature=0.7, top_p=0.9,
            seed=seed + 9000 + hash(p.user_id) % 10_000,
        )
        print(f"\n[Phase 4] {speaker_tag(p)}  alpha={alpha:.2f}",
              flush=True)
        t0 = time.time()
        out = decoder.generate(
            s_pos=p.s_pos, s_neg=prompt_lib.build_negative_prompt(phase=4),
            user_msg=msg, cfg=cfg, alpha=alpha,
        )
        print(f"  -> {len(out['tokens'])} toks in {time.time()-t0:.1f}s",
              flush=True)
        print(out["text"][:300] + ("..." if len(out["text"]) > 300 else ""),
              flush=True)
        reflections[p.short_id] = out["text"]

    (transcripts_dir / "phase4_reflections.json").write_text(json.dumps(
        {p.short_id: {"speaker": speaker_tag(p),
                       "reflection": reflections[p.short_id]}
         for p in cast}, indent=2, ensure_ascii=False))

    # ---- Markdown transcript ----
    md = []
    md.append(f"# Book Club — Comparative Discussion: "
              f"\"{title_a}\" vs \"{title_b}\"\n")
    md.append(f"_Base model:_ `{model_id}` (system-prompt contrastive "
              "decoding, Dong et al. 2026)\n\n")
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

    md.append("\n---\n\n## Phase 1 — Private Comparative Reviews\n\n")
    for p in cast:
        md.append(f"### {speaker_tag(p)}\n\n{private[p.short_id]}\n\n")

    md.append("\n---\n\n## Phase 2 — Broadcast Reactions\n\n")
    for p in cast:
        md.append(f"### {speaker_tag(p)}\n\n{reactions[p.short_id]}\n\n")

    md.append("\n---\n\n## Phase 3 — Moderated Discussion\n\n")
    for r, rr in enumerate(rounds, start=1):
        md.append(f"### Round {r}\n\n")
        md.append(f"_Moderator:_ {moderator_lines[min(r-1, len(moderator_lines)-1)]}\n\n")
        for e in rr:
            md.append(f"**{e['speaker']}**\n\n{e['text']}\n\n")

    md.append("\n---\n\n## Phase 4 — Votes + Reflections\n\n")
    for p in cast:
        md.append(f"### {speaker_tag(p)}\n\n{reflections[p.short_id]}\n\n")

    (output_dir / "discussion_transcript.md").write_text("".join(md))

    print(f"\n[done] outputs written to {output_dir}", flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--personas-dir", type=Path,
                    default=Path(__file__).resolve().parents[3] / "personas")
    ap.add_argument("--output-dir", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "outputs_compare")
    ap.add_argument("--model-id", type=str,
                    default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--story-a", type=str, default="story_1.md")
    ap.add_argument("--story-b", type=str, default="story_2.md")
    ap.add_argument("--title-a", type=str,
                    default="Version A (story_1)")
    ap.add_argument("--title-b", type=str,
                    default="Version B (story_2)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    run_compare(
        personas_dir=args.personas_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        story_a_path=args.personas_dir / args.story_a,
        story_b_path=args.personas_dir / args.story_b,
        title_a=args.title_a,
        title_b=args.title_b,
        seed=args.seed,
        n_discussion_rounds=args.rounds,
    )
