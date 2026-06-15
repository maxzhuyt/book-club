"""
Plan 01 — closed-model 4-phase book-club simulator with CD-generated few-shot
exemplars injected into each persona's system prompt.

Mirrors simulation/src/bookclub/simulate.py turn-for-turn, but every
decoder.generate() call is replaced with an OpenRouter chat-completion call
and the persona's system prompt is augmented with 2 CD-generated voice
exemplars from the bank produced by build_exemplars.py.

The CD code (simulation/src/cd/) is NOT modified or imported at runtime
here — only build_exemplars.py touches it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "simulation" / "src"))
sys.path.insert(0, str(REPO_ROOT / "closed_models" / "src"))

from bookclub.cast import Persona, load_cast, add_six_year_old, speaker_tag  # noqa: E402
from cd import prompts as prompt_lib  # noqa: E402

from common.openrouter import (  # noqa: E402
    DEFAULT_KEY_NAME, load_key, call_openrouter, extract_text,
)


# ---------------------------------------------------------------- exemplars

PHASE_KEYS = ("phase1", "phase2", "phase3_r1", "phase3_r2", "phase4")

# Per-phase preference list for which bank phases to draw exemplars from
# when the current generation phase has no/insufficient exemplars.
FALLBACK_ORDER = {
    "phase1": ["phase1", "phase2", "phase3_r1"],
    "phase2": ["phase2", "phase3_r1", "phase3_r2", "phase1"],
    "phase3_r1": ["phase3_r1", "phase3_r2", "phase2"],
    "phase3_r2": ["phase3_r2", "phase3_r1", "phase2"],
    "phase4": ["phase4", "phase3_r2", "phase3_r1", "phase2"],
}

MAX_EXEMPLAR_CHARS = 3200   # ~800 tokens at ~4 chars/tok


def load_bank(bank_dir: Path) -> tuple[dict, dict]:
    manifest = json.loads((bank_dir / "manifest.json").read_text())
    bank: dict[str, dict[str, list[dict]]] = {}
    for entry in manifest["cast"]:
        sid = entry["short_id"]
        persona_dir = bank_dir / sid
        if not persona_dir.exists():
            continue
        bank[sid] = {}
        for phase in PHASE_KEYS:
            p_path = persona_dir / f"{phase}.json"
            if p_path.exists():
                bank[sid][phase] = json.loads(p_path.read_text())
    return manifest, bank


def pick_exemplars(
    bank: dict[str, dict[str, list[dict]]],
    short_id: str,
    target_phase: str,
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Pick `n` exemplars for the given target phase, falling back as needed."""
    chosen: list[dict] = []
    persona_bank = bank.get(short_id, {})
    seen_seeds: set[int] = set()

    for phase in FALLBACK_ORDER.get(target_phase, [target_phase]):
        candidates = persona_bank.get(phase, [])
        # Filter length and dedupe by seed
        candidates = [
            c for c in candidates
            if len(c.get("text", "")) <= MAX_EXEMPLAR_CHARS
            and c.get("seed") not in seen_seeds
        ]
        rng.shuffle(candidates)
        for c in candidates:
            chosen.append({"phase": phase, **c})
            seen_seeds.add(c["seed"])
            if len(chosen) >= n:
                return chosen
    return chosen


def render_system_prompt(
    persona: Persona, exemplars: list[dict],
) -> str:
    if not exemplars:
        return persona.s_pos
    block = [
        "\n\n--- BOOK-CLUB VOICE EXEMPLARS ---",
        "Below are examples of how YOU specifically write in a book-club "
        "discussion. Match the diction, sentence rhythm, hedging style, "
        "opinion strength, and length. Do NOT imitate the example's topic "
        "or stance; imitate its voice.",
    ]
    for i, ex in enumerate(exemplars, start=1):
        block.append(f"\nEXAMPLE {i} (your {ex['phase']} style):")
        block.append("<<<")
        block.append(ex["text"].strip())
        block.append(">>>")
    block.append("--- END EXEMPLARS ---")
    return persona.s_pos + "\n".join(block)


# ------------------------------------------------------ token budgets (mirrors simulate.py)

def phase1_max_new_tokens(p: Persona) -> int:
    target = p.avg_review_words
    return max(120, min(900, int(target * 1.4)))


def discussion_max_new_tokens(p: Persona) -> int:
    target = max(60, min(400, p.avg_review_words // 4))
    return int(target * 1.4) + 40


def reflection_max_new_tokens(p: Persona) -> int:
    return 220


# ---------------------------------------------------------------- runner


def run(
    *,
    personas_dir: Path,
    bank_dir: Path,
    output_dir: Path,
    model_id: str,
    story_path: Path,
    story_title: str,
    api_key: str,
    seed: int = 17,
    n_discussion_rounds: int = 2,
    n_exemplars: int = 2,
    include_six_year_old: bool = False,
    temperature: float = 0.75,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)

    manifest_bank, bank = load_bank(bank_dir)
    print(f"[plan01] bank story = {manifest_bank.get('story')}", flush=True)
    print(f"[plan01] closed model = {model_id}", flush=True)

    cast = load_cast(personas_dir)
    if include_six_year_old:
        cast = add_six_year_old(cast, prompt_lib.SIX_YEAR_OLD_POS)
    story_text = story_path.read_text()

    rng = random.Random(seed)

    manifest = {
        "plan": "01_few_shot_distillation",
        "closed_model_id": model_id,
        "cd_model_id": manifest_bank.get("cd_model_id"),
        "bank_story": manifest_bank.get("story"),
        "eval_story": story_path.name,
        "story_title": story_title,
        "seed": seed,
        "n_discussion_rounds": n_discussion_rounds,
        "n_exemplars_per_turn": n_exemplars,
        "temperature": temperature,
        "include_six_year_old": include_six_year_old,
        "personas": [
            {
                "short_id": p.short_id, "user_id": p.user_id,
                "label": p.label, "n_books": p.n_books,
                "avg_review_words": p.avg_review_words,
                "six_year_old": p.six_year_old,
            }
            for p in cast
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )

    def call(p: Persona, target_phase: str, user_msg: str,
             max_tokens: int, call_seed: int) -> tuple[str, list[dict]]:
        exemplars = pick_exemplars(bank, p.short_id, target_phase,
                                   n=n_exemplars, rng=rng)
        sys_prompt = render_system_prompt(p, exemplars)
        resp = call_openrouter(
            model_id=model_id, system=sys_prompt, user=user_msg,
            api_key=api_key, max_tokens=max_tokens,
            temperature=temperature, seed=call_seed,
        )
        return extract_text(resp), exemplars

    # ---- Phase 1 ----
    print("\n========== PHASE 1: PRIVATE STANCE ==========", flush=True)
    phase1_msg = prompt_lib.phase1_user_message(story_text, story_title)
    private_reviews: dict[str, str] = {}
    exemplar_trace: dict[str, dict] = {p.short_id: {} for p in cast}
    for p in cast:
        call_seed = seed + hash(p.user_id) % 10_000
        print(f"\n[Phase 1] {speaker_tag(p)}", flush=True)
        t0 = time.time()
        text, used = call(p, "phase1", phase1_msg,
                          phase1_max_new_tokens(p), call_seed)
        print(f"  -> {len(text)} chars in {time.time()-t0:.1f}s "
              f"({len(used)} exemplars)", flush=True)
        private_reviews[p.short_id] = text
        exemplar_trace[p.short_id]["phase1"] = [
            {"phase": e["phase"], "seed": e["seed"]} for e in used
        ]
        print(text[:300] + ("..." if len(text) > 300 else ""), flush=True)

    (transcripts_dir / "phase1_private.json").write_text(json.dumps(
        {p.short_id: {"speaker": speaker_tag(p),
                       "review": private_reviews[p.short_id]}
         for p in cast}, indent=2, ensure_ascii=False))

    # ---- Phase 2 ----
    print("\n========== PHASE 2: BROADCAST & REACTIONS ==========",
          flush=True)
    labeled_peers = {speaker_tag(p): private_reviews[p.short_id] for p in cast}
    reactions: dict[str, str] = {}
    for p in cast:
        peers = {k: v for k, v in labeled_peers.items()
                 if not k.startswith(p.short_id + " ")}
        msg = prompt_lib.phase2_user_message(story_title, peers)
        call_seed = seed + 1000 + hash(p.user_id) % 10_000
        print(f"\n[Phase 2] {speaker_tag(p)}", flush=True)
        t0 = time.time()
        text, used = call(p, "phase2", msg,
                          discussion_max_new_tokens(p), call_seed)
        print(f"  -> {len(text)} chars in {time.time()-t0:.1f}s "
              f"({len(used)} exemplars)", flush=True)
        reactions[p.short_id] = text
        exemplar_trace[p.short_id]["phase2"] = [
            {"phase": e["phase"], "seed": e["seed"]} for e in used
        ]
        print(text[:300] + ("..." if len(text) > 300 else ""), flush=True)

    (transcripts_dir / "phase2_reactions.json").write_text(json.dumps(
        {p.short_id: {"speaker": speaker_tag(p),
                       "reaction": reactions[p.short_id]}
         for p in cast}, indent=2, ensure_ascii=False))

    # ---- Phase 3 ----
    history: list[dict] = []
    for p in cast:
        history.append({
            "speaker": speaker_tag(p) + " [Phase 1 private review]",
            "text": private_reviews[p.short_id],
        })
    for p in cast:
        history.append({
            "speaker": speaker_tag(p) + " [Phase 2 reaction]",
            "text": reactions[p.short_id],
        })

    print("\n========== PHASE 3: MODERATED DISCUSSION ==========",
          flush=True)
    moderator_lines = [
        ("The moderator (an outside facilitator) says: I notice the group has "
         "shared first impressions and reactions. Now I want each of you, in "
         "turn, to engage DIRECTLY with the readers whose views differ most "
         "from your own. Quote them. Tell us where you think they are wrong, "
         "or right, or interestingly partial. Do NOT be polite for politeness' "
         "sake."),
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
        order = cast[r % len(cast):] + cast[:r % len(cast)]
        target_phase = f"phase3_r{r+1}"
        for p in order:
            msg = prompt_lib.phase3_user_message(
                story_title, round_idx=r + 1, history=history,
                moderator_addendum=addendum,
            )
            call_seed = seed + 5000 + r * 100 + hash(p.user_id) % 10_000
            print(f"\n[Phase 3 R{r+1}] {speaker_tag(p)}", flush=True)
            t0 = time.time()
            text, used = call(p, target_phase, msg,
                              discussion_max_new_tokens(p), call_seed)
            print(f"  -> {len(text)} chars in {time.time()-t0:.1f}s "
                  f"({len(used)} exemplars)", flush=True)
            print(text[:300] + ("..." if len(text) > 300 else ""),
                  flush=True)
            exemplar_trace[p.short_id].setdefault(target_phase, [])
            exemplar_trace[p.short_id][target_phase].append(
                [{"phase": e["phase"], "seed": e["seed"]} for e in used]
            )
            entry = {
                "speaker": speaker_tag(p) + f" [Phase 3 R{r+1}]",
                "text": text,
            }
            this_round.append(entry)
            history.append(entry)
        rounds.append(this_round)

    (transcripts_dir / "phase3_discussion.json").write_text(json.dumps([
        [{"speaker": e["speaker"], "text": e["text"]} for e in rr]
        for rr in rounds
    ], indent=2, ensure_ascii=False))

    # ---- Phase 4 ----
    print("\n========== PHASE 4: REFLECTION ==========", flush=True)
    reflections: dict[str, str] = {}
    for p in cast:
        msg = prompt_lib.phase4_user_message(story_title, history)
        call_seed = seed + 9000 + hash(p.user_id) % 10_000
        print(f"\n[Phase 4] {speaker_tag(p)}", flush=True)
        t0 = time.time()
        text, used = call(p, "phase4", msg,
                          reflection_max_new_tokens(p), call_seed)
        print(f"  -> {len(text)} chars in {time.time()-t0:.1f}s "
              f"({len(used)} exemplars)", flush=True)
        print(text[:300] + ("..." if len(text) > 300 else ""), flush=True)
        reflections[p.short_id] = text
        exemplar_trace[p.short_id]["phase4"] = [
            {"phase": e["phase"], "seed": e["seed"]} for e in used
        ]

    (transcripts_dir / "phase4_reflections.json").write_text(json.dumps(
        {p.short_id: {"speaker": speaker_tag(p),
                       "reflection": reflections[p.short_id]}
         for p in cast}, indent=2, ensure_ascii=False))

    (output_dir / "exemplar_trace.json").write_text(
        json.dumps(exemplar_trace, indent=2, ensure_ascii=False)
    )

    # ---- Markdown transcript ----
    md = [f"# Book Club Discussion — {story_title}\n",
          f"_Closed model:_ `{model_id}` with CD few-shot exemplars from "
          f"`{manifest_bank.get('cd_model_id')}` "
          f"(Plan 01 — few-shot distillation).\n\n",
          "## Cast\n\n",
          "| Short ID | Persona | n_books | Avg review words |\n",
          "|---|---|---|---|\n"]
    for p in cast:
        md.append(f"| {p.short_id} | {p.label} (user_{p.user_id}) | "
                  f"{p.n_books} | {p.avg_review_words} |\n")

    md.append("\n---\n\n## Phase 1 — Private Reviews\n\n")
    for p in cast:
        md.append(f"### {speaker_tag(p)}\n\n{private_reviews[p.short_id]}\n\n")

    md.append("\n---\n\n## Phase 2 — Broadcast Reactions\n\n")
    for p in cast:
        md.append(f"### {speaker_tag(p)}\n\n{reactions[p.short_id]}\n\n")

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
    print(f"\n[plan01] outputs written to {output_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-id", type=str, required=True,
        help="OpenRouter model id, e.g. anthropic/claude-opus-4.6, "
             "openai/gpt-4.1, google/gemini-2.5-pro.",
    )
    ap.add_argument("--key-name", type=str, default=DEFAULT_KEY_NAME)
    ap.add_argument(
        "--personas-dir", type=Path,
        default=REPO_ROOT / "personas",
    )
    ap.add_argument(
        "--bank-dir", type=Path,
        default=Path(__file__).resolve().parent / "bank",
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "simulation" / "outputs_closed_fewshot",
    )
    ap.add_argument("--story", type=str, default="story_1.md",
                    help="Eval story (MUST differ from bank story).")
    ap.add_argument("--story-title", type=str, default="Love in the Limelight")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-exemplars", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--include-child", action="store_true",
                    help="Include the 6-year-old (default: omit; see "
                         "closed_models/00_meta.md for reasoning).")
    args = ap.parse_args()

    api_key = load_key(args.key_name)
    story_path = args.personas_dir / args.story
    if not story_path.exists():
        raise SystemExit(f"Story not found: {story_path}")

    # Defensive: refuse to run if bank story equals eval story.
    bank_manifest = json.loads((args.bank_dir / "manifest.json").read_text())
    if bank_manifest.get("story") == args.story:
        raise SystemExit(
            f"Bank story ({bank_manifest.get('story')}) == eval story "
            f"({args.story}). Exemplars would contaminate eval. "
            "Re-run build_exemplars.py with a different --story."
        )

    run(
        personas_dir=args.personas_dir,
        bank_dir=args.bank_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        story_path=story_path,
        story_title=args.story_title,
        api_key=api_key,
        seed=args.seed,
        n_discussion_rounds=args.rounds,
        n_exemplars=args.n_exemplars,
        include_six_year_old=args.include_child,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
