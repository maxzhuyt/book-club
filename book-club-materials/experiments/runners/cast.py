"""
Self-contained persona loader for the experiments folder.

Mirrors the structure of book-club-app/backend/bookclub/cast.py but loads from
experiments/personas/ so the test does not depend on the running app.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PERSONAS_DIR = Path(__file__).resolve().parent.parent / "personas"

FRONTEND_TO_USER_ID: dict[str, str] = {
    "mara":  "69106439",
    "theo":  "3427339",
    "iris":  "7313087",
    "june":  "5431458",
    "sam":   "1375548",
    "rohan": "5149179",
    "hugo":  "84023",
    "elena": "22227336",
    "pete":  "62478339",
    "nadia": "89672274",
}


@dataclass
class Persona:
    frontend_id: str
    user_id: str
    label: str
    role: str
    system_prompt: str
    avg_review_words: int
    analyticity: float
    emotionality: float

    def speaker_tag(self) -> str:
        return f"{self.frontend_id} ({self.label})"


def _load_cast_summary() -> dict:
    return json.loads((PERSONAS_DIR / "cast_summary.json").read_text())


def load_persona(frontend_id: str) -> Persona:
    user_id = FRONTEND_TO_USER_ID[frontend_id]
    cast = _load_cast_summary()
    entry = {e["user_id"]: e for e in cast["cast"]}[user_id]
    sp_path = PERSONAS_DIR / "system_prompt" / f"user_{user_id}_system_prompt.txt"
    return Persona(
        frontend_id=frontend_id,
        user_id=user_id,
        label=entry["label"],
        role=entry["role"],
        system_prompt=sp_path.read_text(),
        avg_review_words=int(entry["avg_review_words"]),
        analyticity=float(entry["analyticity"]),
        emotionality=float(entry["emotionality"]),
    )


def load_cast(persona_ids: list[str]) -> list[Persona]:
    return [load_persona(pid) for pid in persona_ids]
