"""
Per-persona alpha schedule.

Default alpha = 1.0.  Add offsets based on identity-card signals that mark
conflict with the post-training assistant prior (knowledgeable, articulate,
moderate-length, balanced).  Stronger conflict -> higher alpha, capped at 2.0.
"""

from __future__ import annotations

from typing import Optional


# Story-1 features used for genre-conflict signal.
STORY_1_GENRE = "literary contemporary romance / character study"
STORY_1_KEYWORDS = {
    "welsh", "wales", "poetry", "r.s. thomas", "gwenallt",
    "celebrity", "paparazzi", "bookshop", "literary",
}


def compute_alpha(
    persona_summary: dict,
    phase: int,
    base_alpha: float = 1.0,
    cap: float = 2.0,
) -> float:
    """
    persona_summary: dict from cast_summary.json (one cast entry), or a
        dict with keys: n_books, avg_review_words, analyticity, emotionality,
        era, label, avoids, six_year_old.
    phase: 1, 2, 3, or 4. Phase 4 uses 0.5x of the Phase-1 alpha.
    """
    a = base_alpha

    # Six-year-old: always max alpha — strongest anti-prior persona.
    if persona_summary.get("six_year_old"):
        a = cap
    else:
        # Low book count -> low-proficiency-ish persona (conflicts with prior).
        n_books = persona_summary.get("n_books", 500)
        if n_books <= 250:
            a += 0.4
        elif n_books <= 500:
            a += 0.2

        # Very short review style -> conflicts with "elaborate articulate" prior.
        avg_words = persona_summary.get("avg_review_words", 500)
        if avg_words <= 100:
            a += 0.4
        elif avg_words <= 300:
            a += 0.15

        # Very long review style -> still conflicts (prior is moderate),
        # but in the other direction; smaller offset.
        if avg_words >= 1800:
            a += 0.2

        # Strong genre mismatch with the target story.
        avoids = [s.lower() for s in persona_summary.get("avoids", [])]
        if any(k in " ".join(avoids) for k in ("classics", "nonfiction", "history", "literary")):
            a += 0.2

        # Extreme tone — pure analytic or pure emotional.
        emo = persona_summary.get("emotionality", 3.0)
        ana = persona_summary.get("analyticity", 3.0)
        if emo >= 8.0 or ana >= 6.0:
            a += 0.15

    # Phase weighting.
    if phase == 4:
        a = max(0.0, 0.5 * a)
    # Phases 1, 2, 3 use the same alpha.

    return min(cap, a)
