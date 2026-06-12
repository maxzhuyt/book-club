"""
Judge agent — BLIND paired A/B comparator.

The judge is given two versions of the same scene labeled only "Version A" and
"Version B" — it does not know which is the original draft and which is the
revision produced by the book club / workshop pipeline. The orchestrator
randomizes the (A, B) assignment per (cell, condition) and records the mapping
so we can compute "did the revision win on marker X?" without label bias.

Output: per-marker / per-craft-dimension preference (A | B | tied), one-paragraph
synthesis, overall winner. Evidence required for every preference call.

The CRAFT_GUIDE.md is embedded into the system prompt verbatim so the framework
is identical to a human judge using that guide.
"""

from __future__ import annotations
import os
from pathlib import Path


# Defaults to <repo-root>/canons/CRAFT_GUIDE.md derived from this file's location
# (this file lives at <repo>/experiments/runners/judge.py). Set CRAFT_GUIDE_PATH
# in the env to override.
_DEFAULT_CRAFT_GUIDE = (Path(__file__).resolve().parent.parent.parent
                        / "canons" / "CRAFT_GUIDE.md")
CRAFT_GUIDE_PATH = Path(os.environ.get("CRAFT_GUIDE_PATH", _DEFAULT_CRAFT_GUIDE))


def _craft_guide_text() -> str:
    return CRAFT_GUIDE_PATH.read_text()


JUDGE_SYSTEM_TEMPLATE = """\
You are a careful judge of short fiction in a BLIND PAIRED COMPARISON.

You will be shown two versions of the same short historical-fiction scene,
labeled only "Version A" and "Version B". One is an earlier draft and the other
is a revision; you are NOT told which is which and must not guess. Your job
is to compare them along specific dimensions of craft and report which version
is stronger on each — or note that they are tied.

Your evaluation framework is the "Judge's Guide to Great Short Fiction"
reproduced in full below. Apply its markers, false-marker checks, and craft
dimensions exactly as written. Cite specific passages from A or B as evidence
for every preference call.

You MUST output ONLY valid JSON — no prose before or after the JSON block.
Use exactly the codes from the guide. Do not infer authorship, intent, or
which version is "newer." Compare what is on the page.

================ JUDGE'S GUIDE ================
{guide}
================ END OF GUIDE ================
"""


def judge_system() -> str:
    return JUDGE_SYSTEM_TEMPLATE.format(guide=_craft_guide_text())


JUDGE_OUTPUT_SCHEMA = """\
Output a JSON object with EXACTLY this shape (no extra keys, no fences). For
every preference field, "winner" is one of "A" | "B" | "tied" and "evidence"
is a brief quoted phrase or close paraphrase from the relevant version(s).

{
  "mode_identified": {
    "A": "period immersion | recovery | fictionalized biography | invented past | counterfactual | secret history",
    "B": "same set"
  },
  "stratum": {
    "A": "competent | good | great",
    "B": "competent | good | great"
  },
  "markers": {
    "M1_multiple_functions_per_element": {"winner": "A|B|tied", "evidence": "..."},
    "M2_necessary_specificity":          {"winner": "A|B|tied", "evidence": "..."},
    "M3_trained_reading":                {"winner": "A|B|tied", "evidence": "..."},
    "M4_stable_contradiction":           {"winner": "A|B|tied", "evidence": "..."},
    "M5_surplus_meaning":                {"winner": "A|B|tied", "evidence": "..."},
    "M6_earned_risk":                    {"winner": "A|B|tied", "evidence": "..."},
    "M7_sentence_level_necessity":       {"winner": "A|B|tied", "evidence": "..."},
    "M8_defamiliarization":              {"winner": "A|B|tied", "evidence": "..."},
    "M9_implication_exceeds_assertion":  {"winner": "A|B|tied", "evidence": "..."},
    "M10_reread_test":                   {"winner": "A|B|tied", "evidence": "..."},
    "H1_period_mind_not_costume":        {"winner": "A|B|tied", "evidence": "..."},
    "H2_verifiable_specific_anchors":    {"winner": "A|B|tied", "evidence": "..."},
    "H3_material_culture_texture":       {"winner": "A|B|tied", "evidence": "..."},
    "H4_historiographical_question":     {"winner": "A|B|tied", "evidence": "..."},
    "H5_no_present_tense_moralizing":    {"winner": "A|B|tied", "evidence": "..."},
    "H6_selective_compression":          {"winner": "A|B|tied", "evidence": "..."},
    "H7_right_linguistic_distance":      {"winner": "A|B|tied", "evidence": "..."},
    "H8_real_persons_boundary_respected":{"winner": "A|B|tied", "evidence": "..."},
    "H9_estranging_effect_on_present":   {"winner": "A|B|tied", "evidence": "..."},
    "H10_counterfactual_discipline":     {"winner": "A|B|tied", "evidence": "..."}
  },
  "false_markers_flagged": {
    "A": [{"code": "F1|F2|F3|F4|F5|F6|F7|FH1|FH2|FH3|FH4|FH5|FH6", "evidence": "..."}],
    "B": [{"code": "F1|F2|F3|F4|F5|F6|F7|FH1|FH2|FH3|FH4|FH5|FH6", "evidence": "..."}]
  },
  "craft_dimensions": {
    "pov":           {"winner": "A|B|tied", "evidence": "..."},
    "voice_tone":    {"winner": "A|B|tied", "evidence": "..."},
    "structure":     {"winner": "A|B|tied", "evidence": "..."},
    "dialogue":      {"winner": "A|B|tied", "evidence": "..."},
    "image_detail":  {"winner": "A|B|tied", "evidence": "..."},
    "character":     {"winner": "A|B|tied", "evidence": "..."},
    "time":          {"winner": "A|B|tied", "evidence": "..."},
    "ending":        {"winner": "A|B|tied", "evidence": "..."},
    "length_scale":  {"winner": "A|B|tied", "evidence": "..."},
    "earned_effect": {"winner": "A|B|tied", "evidence": "..."}
  },
  "decisive_tests": {
    "would_teach":             {"winner": "A|B|tied", "reason": "..."},
    "would_give_to_loved_one": {"winner": "A|B|tied", "reason": "..."},
    "would_reread_in_5_years": {"winner": "A|B|tied", "reason": "..."}
  },
  "overall_winner": "A|B|tied",
  "overall_margin": "decisive | clear | narrow | tied",
  "regressions_in_winner": "<short note: if the winner has a passage that is WORSE than the other version's analogous passage, name it. Empty string if none.>",
  "qualitative_observations": {
    "version_A_felt_experience": "<2-4 sentences in your own voice: what was it like to read A? where did it land, where did it sag, what stayed with you?>",
    "version_B_felt_experience": "<2-4 sentences in your own voice: what was it like to read B?>",
    "where_the_difference_lives": "<2-4 sentences: locate the felt difference between A and B — is it in the prose, the structure, the texture, the ending? Cite specific passages.>",
    "what_the_rubric_misses": "<1-3 sentences: anything striking about either version that the marker grid does not capture — a quality, a flaw, a quiet effect, a pattern. Empty string if nothing.>"
  },
  "one_paragraph_judgment": "<1 paragraph synthesizing the comparison. Refer to A and B by letter only. Do not speculate about authorship or revision history.>"
}

Notes:
- "tied" is a valid call when the two versions are genuinely close on a dimension.
  Use it; do not force differentiation.
- For markers that do not apply (e.g., H4 historiographical question for a story
  without an archival angle), set winner to "tied" and evidence to "n/a — marker
  does not apply to either version".
- Do not let any aspect (length, vocabulary register, surface polish) bleed into
  judgments of other aspects."""


def judge_user(story_title: str,
               version_A: str, version_B: str,
               cell_context: str = "") -> str:
    ctx = (
        f"\nCell-level context (for grounding mode identification; the same brief "
        f"underlies BOTH versions):\n{cell_context}\n" if cell_context else ""
    )
    return (
        f'BLIND paired comparison of two versions of "{story_title}".{ctx}\n\n'
        "Read both versions twice — first immersively, then analytically. Then "
        "produce the structured comparison. Cite specific passages from A or B "
        "as evidence for every preference call.\n\n"
        f"=== VERSION A ===\n{version_A}\n=== END VERSION A ===\n\n"
        f"=== VERSION B ===\n{version_B}\n=== END VERSION B ===\n\n"
        f"{JUDGE_OUTPUT_SCHEMA}"
    )
