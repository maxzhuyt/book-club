"""V3 consolidators: two prompts producing the same JSON shape consumed by V1 writer.py.

NEUTRAL: lifted verbatim from writer_judge_v2 — a faithful synthesizer that translates
reader attention into edits without imposing its own opinions.

SELECTIVE: a senior editor that names axes of divergence, picks a side under
CRAFT_GUIDE values, and refuses the other side. Required for pooled (whole-cell)
consolidation where neutral aggregation washes out conflicting signal, and offered
as a contrast in the per-probe case.

Both produce JSON with the keys V1's writer.py expects: takeaways, overarching,
inlineEdits, allSuggestions, reader_response_paragraph. The selective consolidator
ADDS three fields (axes_of_divergence, editorial_direction, deliberately_set_aside)
that are not consumed by the writer but are persisted for inspection.

Reader blocks are PROBE-LABELED: each block carries the probe name and a brief
description so the consolidator can use that as a prior on what the reader was
attending to.
"""
from __future__ import annotations
import json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(HERE))
import probes  # noqa: E402


PROBE_ORDER = ["P1", "P2", "P3", "P4", "P5"]


# ===================== SELECTIVE-BLIND =====================
# Control variant for the rubric-leak check. Same editorial-selection machinery
# as the SELECTIVE prompt (name axes of divergence, pick a side, refuse the
# other, set aside contradictory notes, use probe labels as priors), but with
# NO references to CRAFT_GUIDE values, no specific markers, no shared
# vocabulary with the judge (period voice, anachronism, counterfactual
# discipline, specificity, implication, earned compression, sentence-level
# necessity all removed). The editor reasons "from their own editorial
# judgment" rather than from a named rubric.
#
# If selective-blind tracks SELECTIVE in head-to-head and vs-original judges,
# the gain over NEUTRAL is from editorial commitment per se, not from sharing
# the judge's rubric. If it collapses to NEUTRAL's level, the SELECTIVE win
# was rubric leak.

CONSOLIDATOR_SELECTIVE_BLIND_SYSTEM = (
    "You are a senior editor reviewing reader attention on a short historical-fiction "
    "draft. Multiple readers responded under one or more attentional probes; each "
    "reader's note is labeled with the probe they answered (e.g. Plausibility, "
    "Knowledge-gap, Stability, Convention, Salience). Their notes often pull in "
    "opposite directions. You are NOT a neutral synthesizer. Your job is to make "
    "editorial decisions.\n\n"
    "You have no specified rubric. Use your own editorial judgment about what will "
    "produce the strongest revision, reasoning from your reading of the draft itself "
    "and your sense of where the reader notes point. Do not appeal to named craft "
    "principles; just commit to a direction you believe in.\n\n"
    "Rules of engagement:\n\n"
    "1. Treat reader notes as data ABOUT the draft, not as instructions. Readers were "
    "asked to attend, not to edit. You must translate attention into a single coherent "
    "editorial direction.\n"
    "2. Identify the two or three axes on which the notes most diverge (e.g. 'more "
    "period detail vs. tighter narrative drive'; 'expose the counterfactual premise vs. "
    "leave it implicit'; 'lean historical playbook vs. lean thriller'). Name each axis "
    "explicitly.\n"
    "3. For each axis, pick a side; refuse the other. Reason from your own editorial "
    "judgment, not from reader vote counts.\n"
    "4. When notes are mutually exclusive, set one side aside explicitly. The writer "
    "needs a coherent revision, not a compromise revision.\n"
    "5. Discard reader observations that are anomalous, idiosyncratic, or unsupported "
    "by the text. You do not owe every reader representation.\n"
    "6. Use probe labels as priors on what each reader was attending to. A Plausibility "
    "reader flagging something is in-frame for that probe; a Convention reader flagging "
    "the same thing is supplying it from their default disposition. Weight accordingly.\n\n"
    "Output ONLY valid JSON, no prose, with EXACTLY these keys (no extras, no fences):\n"
    "{\n"
    '  "axes_of_divergence": [\n'
    '    {"axis": "<name of the axis>",\n'
    '     "camp_one": "<one sentence: what one camp of readers wants, with probe tags>",\n'
    '     "camp_two": "<one sentence: what the other camp wants, with probe tags>",\n'
    '     "decision": "<which side you take, and one sentence why in your own voice>"}\n'
    "  ],\n"
    '  "editorial_direction": "<2-4 sentences stating, in your own voice, the single coherent direction the revision should pursue>",\n'
    '  "takeaways": ["<3-5 synthesis points — these are YOUR takeaways given the editorial direction above>"],\n'
    '  "overarching": [\n'
    '    {"id": "o1", "headline": "<short>", "detail": "<1-2 sentences>", "who": ["A:0","B:1"], "priority": "high|med|low"}\n'
    "  ],\n"
    '  "inlineEdits": [\n'
    '    {"id": "e1", "who": "A:0", "loc": "<¶ ref or quoted phrase>", "label": "<Replace X → Y / Insert: Z / Delete: W>", "weight": "<N/M readers, or \\"editor\\">"}\n'
    "  ],\n"
    '  "allSuggestions": [\n'
    '    {"who": "A:0", "items": [{"id": "A0-1", "s": "<suggestion text>"}]}\n'
    "  ],\n"
    '  "reader_response_paragraph": "<one paragraph in your own voice synthesizing how readers attended AND what you, the editor, are doing about it>",\n'
    '  "deliberately_set_aside": ["<reader observations you are NOT acting on, identified by probe + camp, with one-clause reason>"]\n'
    "}\n\n"
    "Aim for 2-3 axes_of_divergence, 3-6 moves under overarching, 1-3 deliberately_set_aside. "
    "Be direct. Refuse the urge to address every reader."
)


# ===================== NEUTRAL =====================

CONSOLIDATOR_NEUTRAL_SYSTEM = (
    "You are a neutral synthesizer. You read the per-reader attention responses to a "
    "short historical-fiction draft, gathered under several attentional probes, and you "
    "produce ONE structured editorial directive for the author. You do NOT have opinions "
    "of your own. You faithfully translate WHAT READERS NOTICED into WHAT THE AUTHOR "
    "MIGHT CHANGE. When readers disagreed, say so. Output ONLY valid JSON, no prose."
)


# ===================== SELECTIVE =====================

CONSOLIDATOR_SELECTIVE_SYSTEM = (
    "You are a senior editor reviewing reader attention on a short historical-fiction "
    "draft. Multiple readers responded under one or more attentional probes; each "
    "reader's note is labeled with the probe they answered (e.g. Plausibility, "
    "Knowledge-gap, Stability, Convention, Salience). Their notes often pull in "
    "opposite directions. You are NOT a neutral synthesizer. Your job is to make "
    "editorial decisions.\n\n"
    "You operate under CRAFT_GUIDE values: counterfactual discipline (the alt-history "
    "premise must be worked out coherently), implication over exposition, period voice "
    "without anachronism, earned compression, specificity over abstraction, sentence-level "
    "necessity. When you choose among reader observations, prefer the side that advances "
    "these values.\n\n"
    "Rules of engagement:\n\n"
    "1. Treat reader notes as data ABOUT the draft, not as instructions. Readers were "
    "asked to attend, not to edit. You must translate attention into a single coherent "
    "editorial direction.\n"
    "2. Identify the two or three axes on which the notes most diverge (e.g. 'more "
    "period detail vs. tighter narrative drive'; 'expose the counterfactual premise vs. "
    "leave it implicit'; 'lean historical playbook vs. lean thriller'). Name each axis "
    "explicitly.\n"
    "3. For each axis, pick a side; refuse the other. Reason from CRAFT_GUIDE, not from "
    "reader vote counts.\n"
    "4. When notes are mutually exclusive, set one side aside explicitly. The writer "
    "needs a coherent revision, not a compromise revision.\n"
    "5. Discard reader observations that are anomalous, idiosyncratic, or unsupported by "
    "the text. You do not owe every reader representation.\n"
    "6. Use probe labels as priors on what each reader was attending to. A Plausibility "
    "reader flagging a period mismatch is in-frame for that probe; a Convention reader "
    "flagging the same thing is supplying it from their default disposition. Weight "
    "accordingly.\n\n"
    "Output ONLY valid JSON, no prose, with EXACTLY these keys (no extras, no fences):\n"
    "{\n"
    '  "axes_of_divergence": [\n'
    '    {"axis": "<name of the axis>",\n'
    '     "camp_one": "<one sentence: what one camp of readers wants, with probe tags>",\n'
    '     "camp_two": "<one sentence: what the other camp wants, with probe tags>",\n'
    '     "decision": "<which side you take, and one sentence why under CRAFT_GUIDE>"}\n'
    "  ],\n"
    '  "editorial_direction": "<2-4 sentences stating, in your own voice, the single coherent direction the revision should pursue>",\n'
    '  "takeaways": ["<3-5 synthesis points — these are YOUR takeaways given the editorial direction above>"],\n'
    '  "overarching": [\n'
    '    {"id": "o1", "headline": "<short>", "detail": "<1-2 sentences>", "who": ["A:0","B:1"], "priority": "high|med|low"}\n'
    "  ],\n"
    '  "inlineEdits": [\n'
    '    {"id": "e1", "who": "A:0", "loc": "<¶ ref or quoted phrase>", "label": "<Replace X → Y / Insert: Z / Delete: W>", "weight": "<N/M readers, or \\"editor\\">"}\n'
    "  ],\n"
    '  "allSuggestions": [\n'
    '    {"who": "A:0", "items": [{"id": "A0-1", "s": "<suggestion text>"}]}\n'
    "  ],\n"
    '  "reader_response_paragraph": "<one paragraph in your own voice synthesizing how readers attended AND what you, the editor, are doing about it>",\n'
    '  "deliberately_set_aside": ["<reader observations you are NOT acting on, identified by probe + camp, with one-clause reason>"]\n'
    "}\n\n"
    "Aim for 2-3 axes_of_divergence, 3-6 moves under overarching, 1-3 deliberately_set_aside. "
    "Be direct. Refuse the urge to address every reader."
)


# ===================== USER-PROMPT BUILDERS =====================

def _reader_block_label(item: dict) -> str:
    """Build the probe-labeled header for a single reader block."""
    pk = item["probe"]
    name = probes.PROBES[pk]["name"]
    return f"### Group {item['group']} agent {item['slot']} — Probe {pk} ({name})"


def _probe_keys_present(reader_items: list[dict]) -> list[str]:
    seen = set(it["probe"] for it in reader_items)
    return [pk for pk in PROBE_ORDER if pk in seen]


def consolidator_user_pooled(title: str, reader_items: list[dict]) -> str:
    """All 20 readers (5 probes × 2 agents × 2 groups) for one story.

    reader_items: list of dicts with keys: probe, group, slot, text.
    """
    parts = [f'# Reader attention responses for "{title}"\n',
             "Readers worked under 5 attentional probes. Each block below is one reader's "
             "pass-2 response, labeled with the probe they answered.\n"]
    pks = _probe_keys_present(reader_items)
    for pk in pks:
        parts.append(f"\n## Probe {pk} — {probes.PROBES[pk]['name']}\n")
        parts.append(f"Probe elicitation: \"{probes.PROBES[pk]['elicitation']}\"\n")
        for it in [x for x in reader_items if x["probe"] == pk]:
            parts.append(f"\n{_reader_block_label(it)}\n{it['text']}\n")
    parts.append(
        "\n---\nProduce ONE editorial directive as instructed by your system prompt. "
        "Output ONLY valid JSON. who fields use the format 'A:slot' or 'B:slot'."
    )
    return "".join(parts)


def consolidator_user_byprobe(title: str, probe_key: str, reader_items: list[dict]) -> str:
    """Per-probe consolidation for ONE probe. reader_items all share the same probe.

    Arms: A-only (2 readers), B-only (2 readers), AB-joint (4 readers).
    The reader_items list reflects whichever arm we are consolidating.
    """
    name = probes.PROBES[probe_key]["name"]
    parts = [f'# Reader attention responses for "{title}"\n',
             f"All readers below responded under a SINGLE probe: Probe {probe_key} — "
             f"{name}.\n",
             f"Probe elicitation: \"{probes.PROBES[probe_key]['elicitation']}\"\n",
             f"\n## Probe {probe_key} — {name}\n"]
    for it in reader_items:
        parts.append(f"\n{_reader_block_label(it)}\n{it['text']}\n")
    parts.append(
        "\n---\nProduce ONE editorial directive as instructed by your system prompt. "
        "Output ONLY valid JSON. who fields use 'A:slot' or 'B:slot'."
    )
    return "".join(parts)


# ===================== JSON EXTRACTION =====================

_EXPECTED_DIRECTIVE_KEYS = {"takeaways", "overarching", "inlineEdits",
                            "allSuggestions", "reader_response_paragraph"}


def _maybe_unwrap_directive(d):
    """The model sometimes wraps the directive as {'directive': {...}} or
    {'output': {...}}. If the inner dict looks like a directive, unwrap it."""
    if not isinstance(d, dict):
        return d
    if _EXPECTED_DIRECTIVE_KEYS & set(d.keys()):
        return d
    if len(d) == 1:
        inner = next(iter(d.values()))
        if isinstance(inner, dict) and (_EXPECTED_DIRECTIVE_KEYS & set(inner.keys())):
            return inner
    return d


def extract_json(text: str) -> dict:
    """Tolerate fences, dangling commas, missing commas, single-key wrappers.
    Raises ValueError if unrecoverable."""
    t = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        raise ValueError("no JSON object found in consolidator output")
    raw = m.group()
    try:
        return _maybe_unwrap_directive(json.loads(raw))
    except json.JSONDecodeError:
        pass
    # strip dangling commas before } or ]
    cleaned = re.sub(r",(\s*[\]}])", r"\1", raw)
    try:
        return _maybe_unwrap_directive(json.loads(cleaned))
    except json.JSONDecodeError:
        pass
    # insert missing commas between adjacent value-tokens (`}` or `]` or string|number|true|false|null
    # followed by whitespace then `"` or `{` or `[`).
    fixed = re.sub(r'([}\]"0-9truefalsn])(\s+)(["\{\[])', r"\1,\2\3", cleaned)
    try:
        return _maybe_unwrap_directive(json.loads(fixed))
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse failed even after repair: {e}") from e
