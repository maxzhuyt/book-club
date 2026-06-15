"""
Cast definition for the book-club simulation.  Loads the 8 Goodreads-grounded
personas from cast_summary.json and adds a 9th six-year-old persona used to
test contrastive decoding's ability to simulate non-default knowledge levels.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Persona:
    user_id: str
    label: str          # display label for the transcript ("The Emotional Reader", "Six-year-old")
    role: str
    short_id: str       # short tag for prompts, e.g. "Reader-A"
    s_pos: str          # full positive system prompt (3-layer scaffold)
    n_books: int
    avg_review_words: int
    analyticity: float
    emotionality: float
    era: str
    avoids: list[str]
    six_year_old: bool = False

    def summary(self) -> dict:
        return {
            "n_books": self.n_books,
            "avg_review_words": self.avg_review_words,
            "analyticity": self.analyticity,
            "emotionality": self.emotionality,
            "era": self.era,
            "label": self.label,
            "avoids": self.avoids,
            "six_year_old": self.six_year_old,
        }


def _parse_avoids(s_pos: str) -> list[str]:
    m = re.search(r"You tend to avoid:\s*(.+?)\.\n", s_pos)
    if not m:
        return []
    raw = m.group(1)
    if raw.lower().strip().startswith("nothing"):
        return []
    return [w.strip() for w in raw.split(",")]


def load_cast(personas_dir: Path) -> list[Persona]:
    cast_summary = json.loads((personas_dir / "cast_summary.json").read_text())
    personas: list[Persona] = []
    for i, entry in enumerate(cast_summary["cast"]):
        uid = entry["user_id"]
        s_pos_path = personas_dir / "system_prompt" / f"user_{uid}_system_prompt.txt"
        s_pos = s_pos_path.read_text()
        # disambiguating short id, e.g. "Reader-1: The Emotional Reader (#69106439)"
        short = f"Reader-{chr(ord('A') + i)}"
        personas.append(
            Persona(
                user_id=uid,
                label=entry["label"],
                role=entry["role"],
                short_id=short,
                s_pos=s_pos,
                n_books=int(entry["n_books"]),
                avg_review_words=int(entry["avg_review_words"]),
                analyticity=float(entry["analyticity"]),
                emotionality=float(entry["emotionality"]),
                era=entry["era"],
                avoids=_parse_avoids(s_pos),
                six_year_old=False,
            )
        )
    return personas


def add_six_year_old(cast: list[Persona], s_pos_six: str) -> list[Persona]:
    p = Persona(
        user_id="child001",
        label="The Six-Year-Old",
        role="Six-year-old in first grade. Just learning to read.",
        short_id=f"Reader-{chr(ord('A') + len(cast))}",
        s_pos=s_pos_six,
        n_books=20,
        avg_review_words=30,
        analyticity=0.5,
        emotionality=5.0,
        era="picture books",
        avoids=["nonfiction", "classics", "literary"],
        six_year_old=True,
    )
    return cast + [p]


def speaker_tag(p: Persona) -> str:
    return f"{p.short_id} ({p.label})"
