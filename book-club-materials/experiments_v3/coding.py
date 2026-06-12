"""Attention coder: scores a single reader response on attention dimensions.
Assesses reader ATTENTION, not story quality."""
from __future__ import annotations
import json, re
import client_v2

CODER_SYSTEM = (
    "You are a careful coder of reader responses. You score how a reader attended to a "
    "historical-fiction passage, NOT whether the story is good. You output ONLY one JSON "
    "object, no prose. The graded dimensions are integers 0-5; use the FULL range — "
    "reserve 5 for genuinely rich, expert-level attention and 0 for none. The two count "
    "fields are non-negative integers; convention_type is one of the four allowed labels."
)

DIMENSIONS = """\
Score this reader's response on these dimensions:
- period_specificity (0-5): does the reader invoke specific period knowledge (dates, technologies, social/material facts)? 0 none, 3 some general period sense, 5 rich and precise expert detail.
- concreteness (0-5): 0 = vague impressions only; 3 = some named observations; 5 = consistently named, locatable, specific.
- knowledge_invoked (0-5): how much outside/background knowledge the reader brings to bear, beyond what the passage states. 0 none, 5 extensive.
- locations_cited (integer count): how many specific places in the passage the reader points to.
- anchors (integer count): how many concrete historical anchors (objects, terms, events) the reader names.
- convention_type (label): if the reader invokes genre/period conventions or expectations, are they HISTORICAL-fiction-specific (period accuracy, how the past is rendered) = "historical"; generic-genre (thriller/romance/noir/detective beats, pacing, structure) = "generic-genre"; both = "mixed"; or the reader invokes no conventions = "none".
Also: note (one short clause of evidence).
Output exactly: {"period_specificity": int, "concreteness": int, "knowledge_invoked": int, "locations_cited": int, "anchors": int, "convention_type": "historical"|"generic-genre"|"mixed"|"none", "note": str}"""


def parse_coding(text: str) -> dict:
    t = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return {"_parse_error": "no json", "_raw": text[:300]}
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        try:
            obj = json.loads(re.sub(r",(\s*[}\]])", r"\1", m.group()))
        except Exception as e:
            return {"_parse_error": str(e), "_raw": text[:300]}
    for k in ("period_specificity", "concreteness", "knowledge_invoked",
              "locations_cited", "anchors"):
        obj.setdefault(k, 0)
    if obj.get("convention_type") not in ("historical", "generic-genre", "mixed", "none"):
        obj["convention_type"] = "none"
    return obj


async def code_response(response_text: str) -> dict:
    user = f"{DIMENSIONS}\n\n=== READER RESPONSE ===\n{response_text}\n=== END ==="
    raw = await client_v2.chat(
        [{"role": "system", "content": CODER_SYSTEM}, {"role": "user", "content": user}],
        max_tokens=400, temperature=0.2)
    return parse_coding(raw)
