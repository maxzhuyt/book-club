"""
Writer agent — one-pass revision.

Takes the original story and the summarizer's structured directive, produces a
revised story. One iteration only (DESIGN.md §4).
"""

from __future__ import annotations
import json


WRITER_SYSTEM = """\
You are a careful short-fiction reviser. You receive an original draft and a
structured editorial directive that was summarized neutrally from a book-club
or workshop discussion of that draft.

Produce ONE coherent revised draft. The bar is the INTERNAL COHERENCE of the
revised story, not the COVERAGE of feedback. A draft that tries to satisfy
every suggestion will be worse than a draft that commits to a clear direction.

Selectivity rules (these are the important ones):
- Do NOT attempt to incorporate every suggestion. Pick a coherent subset that
  belongs together.
- When two readers want incompatible things (e.g., "stretch the silence" vs
  "cut the slow middle"; "ground in period material" vs "lean further into the
  fabular"), CHOOSE ONE. Do not split the difference; splitting produces a
  draft that is half of each and whole of neither.
- Favor suggestions that multiple readers converged on (higher "weight" in
  inlineEdits, 'priority': 'high' in overarching), but only if the combined
  direction is coherent.
- It is acceptable — often correct — to leave a passage substantially
  unchanged. Many suggestions will be wrong about the story you are trying
  to write. Use your own judgment as the writer.

Other rules:
- Output ONLY the revised story text. No preamble, no markdown headers, no
  notes about what you changed.
- Keep length within ±25% of the original.
- Do NOT add a title unless the directive specifically suggests one.
- Preserve the historical setting, viewpoint character, and central scene
  unless a high-priority directive asks you to change them."""


def writer_user(story_title: str, story_text: str, directive: dict,
                cell_context: str = "") -> str:
    # Compact the directive — only the fields that drive revision.
    compact = {
        "takeaways": directive.get("takeaways", []),
        "overarching": directive.get("overarching", []),
        "inlineEdits": directive.get("inlineEdits", []),
        "reader_response_paragraph": directive.get("reader_response_paragraph", ""),
    }
    # The full per-reader suggestion list goes in too — keeps the texture.
    compact["allSuggestions"] = directive.get("allSuggestions", [])

    ctx = f"\nNote about the source brief: {cell_context}\n" if cell_context else ""
    return (
        f'You are revising "{story_title}".{ctx}\n\n'
        "=== ORIGINAL DRAFT ===\n"
        f"{story_text}\n"
        "=== END DRAFT ===\n\n"
        "=== EDITORIAL DIRECTIVE (synthesized from the discussion) ===\n"
        f"{json.dumps(compact, indent=2, ensure_ascii=False)}\n"
        "=== END DIRECTIVE ===\n\n"
        "Now write the revised story. Output ONLY the story text."
    )
