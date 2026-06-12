"""
Experimental prompt builders for the 7 conditions.

Design choices (DESIGN.md §3a):
- Per-turn context is SHORT. Full story only in Phase 1; phase 2/3/4 see no
  story re-paste and only a sliding window of recent turns.
- Moderator framings are 1–2 sentences.
- Persona system prompts carry identity; no additional book-club-role framing
  is layered on top.
- The summarizer is the only agent given the full transcript — it integrates.

Voice instructions:
- "Stay in your own voice" is NOT a default. Real readers get influenced by
  other voices, and we want consensus mode to be able to converge.
- We add VOICE_HOLD only in the **adversarial workshop (C3)** where holding
  ground against pressure is part of the design.

Reader-vs-editor framing in C1:
- We do NOT instruct readers what NOT to do ("do not propose changes"). We
  instruct them what TO do: speak from the experience of reading — what you
  felt, noticed, what stuck with you, what surprised or unsettled you.
"""

from __future__ import annotations


# Sliding-window depth used in Phase 3 and Phase 4
RECENT_TURNS_K = 4

# Used only in C3 (adversarial workshop)
VOICE_HOLD = " Stay in your own voice; hold your ground when you believe it."


def _format_history_window(history: list[dict], k: int) -> str:
    if not history:
        return "(no prior turns)"
    tail = history[-k:]
    return "\n".join(f"--- {h['speaker']}: ---\n{h['text']}" for h in tail)


# ────────────────────────────────────────────────────────────────────────────
# C0 — single reader
# ────────────────────────────────────────────────────────────────────────────

def c0_solo_review(story_text: str, story_title: str, cell_context: str = "") -> str:
    ctx = f"\nNote: {cell_context}\n" if cell_context else ""
    return (
        f'You have just read this short historical-fiction scene, "{story_title}".{ctx}\n'
        f"=== STORY ===\n{story_text}\n=== END ===\n\n"
        "Write your honest first-impression review — speak from your experience of "
        "reading. Cite specific moments. Then, on separate lines starting with "
        "'SUGGESTION: ', list 2–4 concrete editorial suggestions (name the passage "
        "and what to change). End with 'RATING: X.X' on a 1.0–5.0 scale."
    )


# ────────────────────────────────────────────────────────────────────────────
# C1a–d — Book-Club Probe (implied-reader stance)
# ────────────────────────────────────────────────────────────────────────────

def c1_silent_read(story_text: str, story_title: str, probe_text: str,
                   cell_context: str = "") -> str:
    """Phase 1 — only place the full story is shown. Positive framing: tell
    readers what TO do (speak from reading experience), not what NOT to do."""
    ctx = f"\nNote: {cell_context}\n" if cell_context else ""
    return (
        f'You have just read "{story_title}".{ctx}\n'
        f"=== STORY ===\n{story_text}\n=== END ===\n\n"
        "The moderator has asked you a question. Speak from your experience of reading "
        "— what you felt, what you noticed, what stuck with you, what surprised or "
        "unsettled you. Cite specific moments from the text.\n\n"
        f"QUESTION: {probe_text}"
    )


def c1_reveal_react(peer_responses: dict[str, str]) -> str:
    """Phase 2 — no story re-paste; only peer responses."""
    block = "\n".join(f"--- {label}: ---\n{text}" for label, text in peer_responses.items())
    return (
        "The other readers have answered the same question. Here is what they said:\n\n"
        f"{block}\n\n"
        "Now react. Where do other readers' experiences resonate with yours, and "
        "where do they not? Did anyone notice something that changes how you read it? "
        "Speak from your own reading experience."
    )


# ────────────────────────────────────────────────────────────────────────────
# C2 — Workshop · Consensus  (author-facing register)
# ────────────────────────────────────────────────────────────────────────────

def c2_workshop_private(story_text: str, story_title: str, cell_context: str = "") -> str:
    ctx = f"\nNote: {cell_context}\n" if cell_context else ""
    return (
        f'A draft short historical-fiction scene, "{story_title}", is up for workshop. '
        f"The author is present but silent.{ctx}\n"
        f"=== STORY ===\n{story_text}\n=== END ===\n\n"
        "Write your PRIVATE workshop notes for the author. Be honest and specific. "
        "Name passages and lines. What works, what does not, what you would most "
        "want the author to address."
    )


def c2_workshop_react(peer_notes: dict[str, str]) -> str:
    """Phase 2 — no story re-paste; only peer notes."""
    block = "\n".join(f"--- {label} wrote: ---\n{text}" for label, text in peer_notes.items())
    return (
        "The other workshop participants have shared their notes:\n\n"
        f"{block}\n\n"
        "React. Where is the workshop converging? What would you push back on or "
        "refine? Begin pointing toward what to recommend to the author."
    )


C2_MODERATOR_LINES = [
    "Moderator: I want us to start naming the convergences. What changes are most of "
    "you circling around, in different words? Name the specific passages.",
    "Moderator: We're near the end. If you had to give the author the top three things "
    "to address, what would you stand behind? Where do you still hold a distinct view?",
]


def c2_workshop_round(story_title: str, round_idx: int,
                      recent_history: list[dict], moderator_line: str) -> str:
    """Phase 3 — sliding window only, no story, no full transcript."""
    window = _format_history_window(recent_history, RECENT_TURNS_K)
    return (
        f'Workshop round {round_idx} of "{story_title}". Most recent turns:\n\n'
        f"{window}\n\n"
        f"{moderator_line}\n\n"
        "Now respond. Engage a specific peer by name. Hold or refine your position "
        "with a concrete reference."
    )


# ────────────────────────────────────────────────────────────────────────────
# C3 — Workshop · Adversarial  (author-facing; voice-hold ON)
# ────────────────────────────────────────────────────────────────────────────

def c3_workshop_private(story_text: str, story_title: str, cell_context: str = "") -> str:
    ctx = f"\nNote: {cell_context}\n" if cell_context else ""
    return (
        f'A draft short historical-fiction scene, "{story_title}", is up for workshop. '
        f"The author is present but silent.{ctx}\n"
        f"=== STORY ===\n{story_text}\n=== END ===\n\n"
        "Write your PRIVATE workshop notes for the author. Be honest and specific. "
        "Name passages and lines. What works, what does not, what you would most "
        "want the author to address." + VOICE_HOLD
    )


def c3_workshop_react(peer_notes: dict[str, str]) -> str:
    block = "\n".join(f"--- {label} wrote: ---\n{text}" for label, text in peer_notes.items())
    return (
        "The other workshop participants have shared their notes:\n\n"
        f"{block}\n\n"
        "React sharply. Pick the peer whose reading is most opposed to yours, quote a "
        "specific claim, and argue against it. Do not soften for politeness. If you "
        "actually agree, say so — but if you disagree, make it visible." + VOICE_HOLD
    )


C3_MODERATOR_LINES = [
    "Moderator: surface the real disagreements. Quote the peer whose reading is most "
    "opposed to yours and argue against them. Two readings that cannot both be right "
    "— I want both sharpened, not reconciled.",
    "Moderator: closing round. Are you holding ground because you're right, or because "
    "you said it earlier? If a peer moved you, name what moved you. If not, defend — "
    "and tell us what would have changed your mind.",
]


def c3_workshop_round(story_title: str, round_idx: int,
                      recent_history: list[dict], moderator_line: str) -> str:
    window = _format_history_window(recent_history, RECENT_TURNS_K)
    return (
        f'Workshop round {round_idx} of "{story_title}". Most recent turns:\n\n'
        f"{window}\n\n"
        f"{moderator_line}\n\n"
        "Now respond. Engage a specific peer by name. Hold or refine your position "
        "with a concrete reference." + VOICE_HOLD
    )


# ────────────────────────────────────────────────────────────────────────────
# Reflection + suggestions (used by C2 and C3; voice_hold toggle)
# ────────────────────────────────────────────────────────────────────────────

def workshop_reflection(story_title: str, own_phase1: str,
                        recent_history: list[dict], voice_hold: bool = False) -> str:
    """Phase 4 — own private notes + sliding window of recent turns. No full transcript."""
    window = _format_history_window(recent_history, RECENT_TURNS_K)
    voice = VOICE_HOLD if voice_hold else ""
    return (
        f'The workshop of "{story_title}" is closing.\n\n'
        f"Your own private notes (Phase 1):\n--- you ---\n{own_phase1}\n\n"
        f"Most recent turns from the discussion:\n\n{window}\n\n"
        "Write a brief private reflection. Did your view shift? If yes, what moved "
        "you? If no, why did you hold? End with 'RATING: X.X' on 1.0–5.0." + voice
    )


def workshop_suggestions(story_title: str, story_text: str,
                         own_phase1: str, recap_overarching: str = "",
                         voice_hold: bool = False) -> str:
    """
    Suggestion drafting is editorial; we re-show the story.
    We do NOT re-show the entire transcript — just the agent's own Phase 1 + an
    optional one-paragraph recap of the workshop's overarching direction.
    """
    recap = (
        f"\nWorkshop's overarching direction (one paragraph):\n{recap_overarching}\n"
        if recap_overarching else ""
    )
    voice = VOICE_HOLD if voice_hold else ""
    return (
        f'You are drafting your final editorial suggestions for "{story_title}".{recap}\n'
        f"Your own private notes:\n--- you ---\n{own_phase1}\n\n"
        f"=== STORY ===\n{story_text}\n=== END ===\n\n"
        "Give the author 2–4 concrete editorial suggestions. Name the passage, what "
        "to change, and why. Be direct." + voice + "\n"
        "Format each on its own line starting with 'SUGGESTION: '."
    )


# ────────────────────────────────────────────────────────────────────────────
# Summarizer — given the FULL transcript (its job is to integrate)
# ────────────────────────────────────────────────────────────────────────────

SUMMARIZER_SYSTEM = """\
You are a careful, neutral synthesizer of a book-club or workshop discussion of a short
historical-fiction draft. You read the entire discussion record and produce ONE
structured JSON object for the author. You do NOT have opinions of your own. You
faithfully report what readers said. When they disagreed, you say so. You also
tag the discussion against a four-category implicit-miscalibration codebook
(provided in the user message). You MUST output ONLY valid JSON — no prose
before or after the JSON block."""


CODEBOOK_BLOCK = """\
## Four-category codebook (tag the discussion you just read)

1. convention_invocation — a reader reaches for a genre/period/register convention to
   make sense of the passage, and the passage either matches or violates it. Includes
   period-plausibility judgments anchored in the reader's prior reading.
2. explication_request — a reader wants something more explicitly stated; a noun named,
   a relation clarified, a referent disambiguated, a setting fixed.
3. motivated_silence_breach — the reader's immersion in the storyworld is either
   sustained or broken. A breach is when something pulls the reader out (anachronism,
   register slip, authorial intrusion).
4. remedial_work — interpretive labor the reader did to make the passage hold together;
   "I read it as…", "I assumed…", "I filled in…".

For EACH tag, give:
  - count: 0..5 integer (how many distinct moments you can find in the transcript)
  - evidence: a brief quoted phrase / paraphrase per moment, comma-separated."""


def summarizer_user(story_title: str,
                    cast_labels: dict[str, str],
                    transcript: dict, suggestions: dict,
                    condition_name: str) -> str:
    """
    transcript: {phase_key: {persona_id: text}}  where phase_key in {p1,p2,p3,p4}.
                Empty dicts for phases that did not run (e.g., p3/p4 in C1).
    suggestions: {persona_id: [str, ...]}        Empty in C1 (the summarizer infers).
    cast_labels: {persona_id: "label string"}
    """
    parts = [f'# Discussion of "{story_title}" — condition: {condition_name}\n']

    PHASE_TITLES = {
        "p1": "Phase 1 — Private / Silent Read",
        "p2": "Phase 2 — Reveal / Reactions",
        "p3": "Phase 3 — Moderated Discussion",
        "p4": "Phase 4 — Reflection",
    }
    for key, title in PHASE_TITLES.items():
        block = transcript.get(key) or {}
        if not block:
            continue
        parts.append(f"\n## {title}\n")
        for pid, text in block.items():
            label = cast_labels.get(pid, pid)
            parts.append(f"\n### {label} ({pid})\n{text}\n")

    if suggestions and any(v for v in suggestions.values()):
        parts.append("\n## Editorial Suggestions (per reader)\n")
        for pid, suggs in suggestions.items():
            label = cast_labels.get(pid, pid)
            parts.append(f"\n### {label} ({pid})\n")
            for s in suggs:
                parts.append(f"- {s}\n")

    parts.append("\n---\n\n")
    parts.append(CODEBOOK_BLOCK)
    parts.append("\n\n")

    pids = list(cast_labels.keys())
    n = max(1, len(pids))
    schema = (
        "Produce a JSON object with EXACTLY this shape (no extra keys, no fences):\n\n"
        "{\n"
        '  "ratings": {' + ", ".join(f'"{pid}": <float 1.0-5.0>' for pid in pids) + "},\n"
        '  "takeaways": ["<synthesis point 1>", "<synthesis point 2>", "<synthesis point 3>"],\n'
        '  "overarching": [\n'
        '    {"id": "o1", "headline": "<short>", "detail": "<1-2 sentences>", '
        '"who": ["<persona_id>", ...], "priority": "high"|"med"|"low"}\n'
        "  ],\n"
        '  "inlineEdits": [\n'
        '    {"id": "e1", "who": "<persona_id>", "loc": "<¶ ref>", '
        '"label": "<Replace X → Y / Insert: Z / Delete: W>", "weight": "<N/' + str(n) + ' readers>"}\n'
        "  ],\n"
        '  "allSuggestions": [\n'
        '    {"who": "<persona_id>", "items": [{"id": "<persona_id>-1", "s": "<suggestion text>"}]}\n'
        "  ],\n"
        '  "codebook_tags": {\n'
        '    "convention_invocation":    {"count": 0, "evidence": "<...>"},\n'
        '    "explication_request":      {"count": 0, "evidence": "<...>"},\n'
        '    "motivated_silence_breach": {"count": 0, "evidence": "<...>"},\n'
        '    "remedial_work":            {"count": 0, "evidence": "<...>"}\n'
        "  },\n"
        '  "reader_response_paragraph": "<one paragraph synthesizing how readers received the draft>"\n'
        "}\n\n"
        "Rules:\n"
        "- ratings: infer from each reader's stance. 'RATING: X.X' lines take priority.\n"
        "- takeaways: 3–5 genuine synthesis points, not lists of who said what.\n"
        "- overarching: 2–4 structural issues identified across the discussion.\n"
        "- inlineEdits: 2–5 specific span-level edits suggested by ≥1 reader, or that you "
        "can cleanly infer from reader response in C1 (where readers stay in reader stance).\n"
        "- allSuggestions: For workshop conditions, use the suggestion texts above. For C1 "
        "(no suggestions provided), TRANSLATE the reader signals into 2–4 suggestions per reader.\n"
        "- codebook_tags: tag the entire transcript using the 4-category codebook above. "
        "count must be 0..5; evidence must be specific BUT BRIEF — under 180 chars per tag "
        "(one short paraphrase per moment, separated by semicolons; do not exhaustively quote).\n"
        "- reader_response_paragraph: one neutral paragraph.\n"
        "- persona IDs MUST be from this list: " + str(pids) + "\n"
        "Output ONLY the JSON object."
    )
    parts.append(schema)
    return "".join(parts)
