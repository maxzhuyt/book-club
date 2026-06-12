"""
Per-condition discussion pipelines.

Each pipeline takes:
  - story_text, story_title, cell_context
  - cast (list[Persona]) — the panel for the condition
  - gen(persona, system_prompt, user_msg, max_tokens, temperature) -> str

and returns:
  - transcript: {"p1": {pid: text}, "p2": {pid: text}, "p3": {pid: text}, "p4": {pid: text}}
                (empty dicts for phases that did not run)
  - suggestions: {pid: [str]}                (empty for C1)
"""

from __future__ import annotations
import asyncio
import re
from typing import Callable, Awaitable

from prompts_exp import (
    c0_solo_review,
    c1_silent_read, c1_reveal_react,
    c2_workshop_private, c2_workshop_react, c2_workshop_round,
    c3_workshop_private, c3_workshop_react, c3_workshop_round,
    C2_MODERATOR_LINES, C3_MODERATOR_LINES,
    workshop_reflection, workshop_suggestions,
)

# Type alias
GenFn = Callable[[object, str, int, float], Awaitable[str]]
# gen(persona, user_msg, max_tokens, temperature) -> str  — uses persona.system_prompt


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def extract_rating(text: str) -> float | None:
    m = re.search(r"RATING:\s*([1-5](?:\.\d)?)", text)
    return round(float(m.group(1)), 1) if m else None


def extract_suggestions(text: str) -> list[str]:
    lines = [
        l[len("SUGGESTION:"):].strip()
        for l in text.splitlines()
        if l.strip().upper().startswith("SUGGESTION:")
    ]
    if not lines:
        lines = [l.strip() for l in text.strip().splitlines() if len(l.strip()) > 20][:4]
    return lines[:4]


def phase1_tokens(p):  return max(400, min(1500, int(p.avg_review_words * 1.6)))
def reaction_tokens(p): return int(max(150, min(700, p.avg_review_words // 3)) * 1.5) + 60
def reflection_tokens(_): return 500
def suggestion_tokens(_): return 600
def c1_response_tokens(p): return int(max(250, min(900, p.avg_review_words // 2 + 200)))


# ────────────────────────────────────────────────────────────────────────────
# C0 — Single reader
# ────────────────────────────────────────────────────────────────────────────

async def pipeline_C0(story_text: str, story_title: str, cell_context: str,
                      cast: list, gen: GenFn) -> tuple[dict, dict]:
    assert len(cast) == 1, "C0 must be a single persona"
    p = cast[0]
    msg = c0_solo_review(story_text, story_title, cell_context)
    text = await gen(p, msg, phase1_tokens(p), 0.70)
    transcript = {"p1": {p.frontend_id: text}, "p2": {}, "p3": {}, "p4": {}}
    suggestions = {p.frontend_id: extract_suggestions(text)}
    return transcript, suggestions


# ────────────────────────────────────────────────────────────────────────────
# C1 — Book-Club Probe (silent read → reveal/react). 2 phases only.
# ────────────────────────────────────────────────────────────────────────────

async def pipeline_C1(story_text: str, story_title: str, cell_context: str,
                      cast: list, gen: GenFn, probe_text: str) -> tuple[dict, dict]:
    # Phase 1: silent read with probe (parallel)
    msg1 = c1_silent_read(story_text, story_title, probe_text, cell_context)
    p1_texts = await asyncio.gather(*[
        gen(p, msg1, c1_response_tokens(p), 0.75) for p in cast
    ])
    p1 = {p.frontend_id: t for p, t in zip(cast, p1_texts)}

    # Phase 2: reveal & react (parallel; each persona sees all peer responses
    # except their own)
    labeled = {p.speaker_tag(): p1[p.frontend_id] for p in cast}
    p2_msgs = [
        c1_reveal_react({k: v for k, v in labeled.items()
                         if not k.startswith(p.frontend_id + " ")})
        for p in cast
    ]
    p2_texts = await asyncio.gather(*[
        gen(p, m, reaction_tokens(p), 0.75) for p, m in zip(cast, p2_msgs)
    ])
    p2 = {p.frontend_id: t for p, t in zip(cast, p2_texts)}

    transcript = {"p1": p1, "p2": p2, "p3": {}, "p4": {}}
    suggestions = {p.frontend_id: [] for p in cast}  # summarizer infers in C1
    return transcript, suggestions


# ────────────────────────────────────────────────────────────────────────────
# Shared workshop pipeline body (C2 / C3)
# ────────────────────────────────────────────────────────────────────────────

async def _workshop_pipeline(
    story_text: str, story_title: str, cell_context: str,
    cast: list, gen: GenFn,
    private_fn, react_fn, round_fn, moderator_lines,
    n_rounds: int = 2,
    voice_hold: bool = False,
) -> tuple[dict, dict]:
    # Phase 1: private workshop notes (parallel)
    msg1 = private_fn(story_text, story_title, cell_context)
    p1_texts = await asyncio.gather(*[
        gen(p, msg1, phase1_tokens(p), 0.75) for p in cast
    ])
    p1 = {p.frontend_id: t for p, t in zip(cast, p1_texts)}

    # Phase 2: react (parallel)
    labeled = {p.speaker_tag(): p1[p.frontend_id] for p in cast}
    p2_msgs = [
        react_fn({k: v for k, v in labeled.items()
                  if not k.startswith(p.frontend_id + " ")})
        for p in cast
    ]
    p2_texts = await asyncio.gather(*[
        gen(p, m, reaction_tokens(p), 0.75) for p, m in zip(cast, p2_msgs)
    ])
    p2 = {p.frontend_id: t for p, t in zip(cast, p2_texts)}

    # Build rolling history (full record for the summarizer; pipeline uses tail only)
    history: list[dict] = []
    for p in cast:
        history.append({"speaker": p.speaker_tag() + " [Phase 1 private]",
                        "text": p1[p.frontend_id]})
    for p in cast:
        history.append({"speaker": p.speaker_tag() + " [Phase 2 reaction]",
                        "text": p2[p.frontend_id]})

    # Phase 3: sequential rounds, each turn sees only the recent window
    p3_combined = {p.frontend_id: "" for p in cast}
    for r in range(n_rounds):
        mod = moderator_lines[min(r, len(moderator_lines) - 1)]
        order = cast[r % len(cast):] + cast[:r % len(cast)]
        for p in order:
            msg = round_fn(story_title, r + 1, history, mod)
            turn = await gen(p, msg, reaction_tokens(p), 0.80)
            history.append({"speaker": p.speaker_tag() + f" [Phase 3 R{r+1}]", "text": turn})
            sep = "\n\n[Round 2]\n" if r == 1 and p3_combined[p.frontend_id] else ""
            p3_combined[p.frontend_id] += sep + turn

    # Phase 4: reflection (parallel; each persona sees own Phase 1 + tail of history)
    p4_msgs = [
        workshop_reflection(story_title, p1[p.frontend_id], history,
                            voice_hold=voice_hold)
        for p in cast
    ]
    p4_texts = await asyncio.gather(*[
        gen(p, m, reflection_tokens(p), 0.70) for p, m in zip(cast, p4_msgs)
    ])
    p4 = {p.frontend_id: t for p, t in zip(cast, p4_texts)}

    # Suggestions: parallel; each persona sees story + own Phase 1 (no full transcript)
    # We compute a brief recap from the history for the directive prompt.
    recap = "Phase 1 & Phase 2 surfaced specific notes; Phase 3 rounds engaged " \
            "across positions; Phase 4 captured shifts. See peer notes."
    sugg_msgs = [
        workshop_suggestions(story_title, story_text, p1[p.frontend_id], recap,
                             voice_hold=voice_hold)
        for p in cast
    ]
    sugg_texts = await asyncio.gather(*[
        gen(p, m, suggestion_tokens(p), 0.70) for p, m in zip(cast, sugg_msgs)
    ])
    suggestions = {p.frontend_id: extract_suggestions(t) for p, t in zip(cast, sugg_texts)}

    transcript = {"p1": p1, "p2": p2, "p3": p3_combined, "p4": p4}
    return transcript, suggestions


# ────────────────────────────────────────────────────────────────────────────
# C2 — Workshop · Consensus
# ────────────────────────────────────────────────────────────────────────────

async def pipeline_C2(story_text: str, story_title: str, cell_context: str,
                      cast: list, gen: GenFn) -> tuple[dict, dict]:
    return await _workshop_pipeline(
        story_text, story_title, cell_context, cast, gen,
        private_fn=c2_workshop_private,
        react_fn=c2_workshop_react,
        round_fn=c2_workshop_round,
        moderator_lines=C2_MODERATOR_LINES,
        n_rounds=2,
    )


# ────────────────────────────────────────────────────────────────────────────
# C3 — Workshop · Adversarial
# ────────────────────────────────────────────────────────────────────────────

async def pipeline_C3(story_text: str, story_title: str, cell_context: str,
                      cast: list, gen: GenFn) -> tuple[dict, dict]:
    return await _workshop_pipeline(
        story_text, story_title, cell_context, cast, gen,
        private_fn=c3_workshop_private,
        react_fn=c3_workshop_react,
        round_fn=c3_workshop_round,
        moderator_lines=C3_MODERATOR_LINES,
        n_rounds=2,
        voice_hold=True,
    )
