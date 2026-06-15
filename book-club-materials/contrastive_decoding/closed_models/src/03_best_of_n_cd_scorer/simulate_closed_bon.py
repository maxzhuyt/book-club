"""
Plan 03 — closed-model 4-phase book-club simulator with best-of-N sampling
and CD-derived reranking.

At each turn:
  1. Sample N candidates from the closed model in parallel via OpenRouter
     (different seeds, fixed temperature/top_p).
  2. Optionally pre-filter to top-M with Score B (embedding distance to
     CD reference bank from Plan 01).
  3. Pick argmax with Score A (CD logprob delta) if --use-score-a is set,
     otherwise pick argmax with Score B.

The CD code in simulation/src/cd is NOT modified — Score A only calls
load_dual_models() and the resulting models' forward() for teacher forcing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "simulation" / "src"))
sys.path.insert(0, str(REPO_ROOT / "closed_models" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bookclub.cast import Persona, load_cast, add_six_year_old, speaker_tag  # noqa: E402
from cd import prompts as prompt_lib  # noqa: E402

from common.openrouter import (  # noqa: E402
    DEFAULT_KEY_NAME, load_key, call_openrouter_n_parallel, extract_text,
)
from cd_scorer import (  # noqa: E402
    CDLogprobScorer, EmbeddingScorer, load_score_a,
)


# ------------------------------------------------------ token budgets

def phase1_max_new_tokens(p: Persona) -> int:
    target = p.avg_review_words
    return max(120, min(900, int(target * 1.4)))


def discussion_max_new_tokens(p: Persona) -> int:
    target = max(60, min(400, p.avg_review_words // 4))
    return int(target * 1.4) + 40


def reflection_max_new_tokens(p: Persona) -> int:
    return 220


# ------------------------------------------------------ reranking

def rerank(
    candidates: list[str],
    *,
    short_id: str,
    target_phase: str,
    persona: Persona,
    user_msg: str,
    n_neg: str,
    score_a: Optional[CDLogprobScorer],
    score_b: Optional[EmbeddingScorer],
    m_survivors: int,
) -> tuple[int, list[dict]]:
    """
    Pick the best candidate. Returns (winner_idx, per_candidate_records).

    - If score_b is given, filter to top-M by Score B first.
    - If score_a is given, pick argmax over Score A within the survivors.
    - If only one scorer is given, use it alone.
    - If neither is given, return candidate 0 (caller should avoid this).
    """
    records: list[dict] = []
    for i, c in enumerate(candidates):
        rec: dict = {"index": i, "text": c, "n_chars": len(c)}
        if score_b is not None:
            rec["score_b"] = score_b.score(
                short_id=short_id, phase=target_phase, response=c,
            )
        records.append(rec)

    survivors_idx = list(range(len(candidates)))
    if score_b is not None and m_survivors < len(candidates):
        survivors_idx = sorted(
            range(len(candidates)),
            key=lambda i: records[i]["score_b"], reverse=True,
        )[:m_survivors]

    if score_a is not None:
        for i in survivors_idx:
            r = score_a.score(
                s_pos=persona.s_pos, s_neg=n_neg,
                user_msg=user_msg, response=candidates[i],
            )
            records[i]["score_a"] = {
                "delta_mean": r.delta_mean,
                "delta_sum": r.delta_sum,
                "n_tokens": r.n_tokens,
                "lp_pos_mean": r.lp_pos_mean,
                "lp_neg_mean": r.lp_neg_mean,
            }
        winner = max(survivors_idx,
                     key=lambda i: records[i]["score_a"]["delta_mean"])
    elif score_b is not None:
        winner = max(survivors_idx, key=lambda i: records[i]["score_b"])
    else:
        winner = 0
    return winner, records


# ------------------------------------------------------ runner

def run(
    *,
    personas_dir: Path,
    output_dir: Path,
    model_id: str,
    story_path: Path,
    story_title: str,
    api_key: str,
    seed: int,
    n_discussion_rounds: int,
    n_candidates: int,
    m_survivors: int,
    temperature: float,
    use_score_a: bool,
    use_score_b: bool,
    bank_dir: Optional[Path],
    cd_model_id: str,
    include_six_year_old: bool,
    embedder_id: str,
    device_pos: str,
    device_neg: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(exist_ok=True)

    cast = load_cast(personas_dir)
    if include_six_year_old:
        cast = add_six_year_old(cast, prompt_lib.SIX_YEAR_OLD_POS)
    story_text = story_path.read_text()

    # ---- load scorers
    score_a: Optional[CDLogprobScorer] = None
    score_b: Optional[EmbeddingScorer] = None
    if use_score_a:
        print(f"[plan03] loading Score A dual models: {cd_model_id} on "
              f"{device_pos}/{device_neg}", flush=True)
        t0 = time.time()
        score_a = load_score_a(cd_model_id, device_pos=device_pos,
                               device_neg=device_neg)
        print(f"[plan03] Score A ready in {time.time()-t0:.1f}s", flush=True)
    if use_score_b:
        if bank_dir is None or not (bank_dir / "manifest.json").exists():
            raise SystemExit(
                f"Score B requires a Plan-01 bank at --bank-dir; "
                f"got: {bank_dir}"
            )
        print(f"[plan03] loading Score B embedder: {embedder_id}", flush=True)
        score_b = EmbeddingScorer(
            bank_dir=bank_dir, embedder_id=embedder_id, device=device_pos,
        )

    manifest = {
        "plan": "03_best_of_n_cd_scorer",
        "closed_model_id": model_id,
        "story": story_path.name,
        "story_title": story_title,
        "seed": seed,
        "n_candidates": n_candidates,
        "m_survivors": m_survivors,
        "temperature": temperature,
        "use_score_a": use_score_a,
        "use_score_b": use_score_b,
        "cd_model_id": cd_model_id if use_score_a else None,
        "bank_dir": str(bank_dir) if use_score_b else None,
        "n_discussion_rounds": n_discussion_rounds,
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

    def bon_turn(p: Persona, target_phase: str, phase_label: str,
                 user_msg: str, max_tokens: int, base_seed: int) -> str:
        s_neg_default = prompt_lib.build_negative_prompt(
            phase=int(phase_label) if phase_label.isdigit() else 3
        )
        # Build N candidates in parallel
        resps = call_openrouter_n_parallel(
            model_id=model_id, system=p.s_pos, user=user_msg,
            api_key=api_key, n=n_candidates, base_seed=base_seed,
            max_tokens=max_tokens, temperature=temperature,
        )
        candidates = [extract_text(r) for r in resps]
        winner_idx, records = rerank(
            candidates,
            short_id=p.short_id, target_phase=target_phase, persona=p,
            user_msg=user_msg, n_neg=s_neg_default,
            score_a=score_a, score_b=score_b, m_survivors=m_survivors,
        )
        # Persist all candidates for ablation / debugging
        out_file = candidates_dir / f"{p.short_id}_{target_phase}.json"
        out_file.write_text(json.dumps({
            "winner_index": winner_idx,
            "records": records,
        }, indent=2, ensure_ascii=False))
        return candidates[winner_idx]

    # ---- Phase 1 ----
    print("\n========== PHASE 1: PRIVATE STANCE ==========", flush=True)
    phase1_msg = prompt_lib.phase1_user_message(story_text, story_title)
    private_reviews: dict[str, str] = {}
    for p in cast:
        print(f"\n[Phase 1] {speaker_tag(p)}  N={n_candidates}", flush=True)
        t0 = time.time()
        chosen = bon_turn(p, "phase1", "1", phase1_msg,
                          phase1_max_new_tokens(p),
                          seed + hash(p.user_id) % 10_000)
        print(f"  -> {len(chosen)} chars in {time.time()-t0:.1f}s",
              flush=True)
        private_reviews[p.short_id] = chosen
        print(chosen[:300] + ("..." if len(chosen) > 300 else ""), flush=True)

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
        print(f"\n[Phase 2] {speaker_tag(p)}", flush=True)
        t0 = time.time()
        chosen = bon_turn(p, "phase2", "2", msg,
                          discussion_max_new_tokens(p),
                          seed + 1000 + hash(p.user_id) % 10_000)
        print(f"  -> {len(chosen)} chars in {time.time()-t0:.1f}s",
              flush=True)
        reactions[p.short_id] = chosen
        print(chosen[:300] + ("..." if len(chosen) > 300 else ""), flush=True)

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
            print(f"\n[Phase 3 R{r+1}] {speaker_tag(p)}", flush=True)
            t0 = time.time()
            chosen = bon_turn(p, target_phase, "3", msg,
                              discussion_max_new_tokens(p),
                              seed + 5000 + r * 100 + hash(p.user_id) % 10_000)
            print(f"  -> {len(chosen)} chars in {time.time()-t0:.1f}s",
                  flush=True)
            print(chosen[:300] + ("..." if len(chosen) > 300 else ""),
                  flush=True)
            entry = {
                "speaker": speaker_tag(p) + f" [Phase 3 R{r+1}]",
                "text": chosen,
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
        print(f"\n[Phase 4] {speaker_tag(p)}", flush=True)
        t0 = time.time()
        chosen = bon_turn(p, "phase4", "4", msg,
                          reflection_max_new_tokens(p),
                          seed + 9000 + hash(p.user_id) % 10_000)
        print(f"  -> {len(chosen)} chars in {time.time()-t0:.1f}s",
              flush=True)
        print(chosen[:300] + ("..." if len(chosen) > 300 else ""), flush=True)
        reflections[p.short_id] = chosen

    (transcripts_dir / "phase4_reflections.json").write_text(json.dumps(
        {p.short_id: {"speaker": speaker_tag(p),
                       "reflection": reflections[p.short_id]}
         for p in cast}, indent=2, ensure_ascii=False))

    # ---- Markdown
    md = [f"# Book Club Discussion — {story_title}\n",
          f"_Closed model:_ `{model_id}` with best-of-N (N={n_candidates}) "
          f"reranked by "
          f"{'Score A' if use_score_a else ''}"
          f"{'+' if use_score_a and use_score_b else ''}"
          f"{'Score B' if use_score_b else ''} "
          f"(Plan 03).\n\n",
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
    print(f"\n[plan03] outputs written to {output_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-id", type=str, required=True,
        help="OpenRouter closed-model id.",
    )
    ap.add_argument("--key-name", type=str, default=DEFAULT_KEY_NAME)
    ap.add_argument("--personas-dir", type=Path,
                    default=REPO_ROOT / "personas")
    ap.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "simulation" / "outputs_closed_bon",
    )
    ap.add_argument("--story", type=str, default="story_1.md")
    ap.add_argument("--story-title", type=str, default="Love in the Limelight")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-candidates", type=int, default=6,
                    help="N for best-of-N sampling.")
    ap.add_argument("--m-survivors", type=int, default=3,
                    help="Top-M from Score B that get Score A. Only used "
                         "when both scorers are enabled.")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--use-score-a", action="store_true",
                    help="Enable CD logprob delta scoring (requires 2 GPUs).")
    ap.add_argument("--use-score-b", action="store_true",
                    help="Enable embedding-distance scoring "
                         "(requires --bank-dir).")
    ap.add_argument(
        "--bank-dir", type=Path,
        default=REPO_ROOT / "closed_models" / "src"
                / "01_few_shot_distillation" / "bank",
        help="Plan 01 exemplar bank (used as Score B reference set).",
    )
    ap.add_argument("--cd-model-id", type=str,
                    default="Qwen/Qwen2.5-14B-Instruct",
                    help="Open model for Score A (must match CD setup).")
    ap.add_argument("--embedder-id", type=str,
                    default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device-pos", type=str, default="cuda:0")
    ap.add_argument("--device-neg", type=str, default="cuda:1")
    ap.add_argument("--include-child", action="store_true")
    args = ap.parse_args()

    if not args.use_score_a and not args.use_score_b:
        raise SystemExit(
            "Need at least one of --use-score-a / --use-score-b; "
            "otherwise reranking is a no-op."
        )

    api_key = load_key(args.key_name)
    story_path = args.personas_dir / args.story
    if not story_path.exists():
        raise SystemExit(f"Story not found: {story_path}")

    run(
        personas_dir=args.personas_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        story_path=story_path,
        story_title=args.story_title,
        api_key=api_key,
        seed=args.seed,
        n_discussion_rounds=args.rounds,
        n_candidates=args.n_candidates,
        m_survivors=args.m_survivors,
        temperature=args.temperature,
        use_score_a=args.use_score_a,
        use_score_b=args.use_score_b,
        bank_dir=args.bank_dir,
        cd_model_id=args.cd_model_id,
        include_six_year_old=args.include_child,
        embedder_id=args.embedder_id,
        device_pos=args.device_pos,
        device_neg=args.device_neg,
    )


if __name__ == "__main__":
    main()
