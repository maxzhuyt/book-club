"""Blind group comparator: given the two groups' pass-2 answers to the SAME probe on
the SAME story, labeled only Group 1 / Group 2 (randomized), report what one group
noticed that the other did not. Does not know which is the HF group."""
from __future__ import annotations
import json, re, random
import client_v2

COMPARATOR_SYSTEM = (
    "You compare two sets of reader responses to the same question about the same "
    "historical-fiction passage. You are told only 'Group 1' and 'Group 2'; you do NOT "
    "know anything about who they are. Report differences in what each group ATTENDED "
    "to (specificity, period knowledge, what they noticed), not story quality. Output "
    "ONLY one JSON object."
)


def assign_blind(a_answers: list[str], b_answers: list[str], rng: random.Random):
    """Randomly map real groups A/B to display labels Group 1/Group 2."""
    flip = rng.random() < 0.5
    if flip:
        blind = {"Group 1": b_answers, "Group 2": a_answers}
        label_to_group = {"Group 1": "B", "Group 2": "A"}
    else:
        blind = {"Group 1": a_answers, "Group 2": b_answers}
        label_to_group = {"Group 1": "A", "Group 2": "B"}
    return blind, label_to_group


def parse_comparison(text: str) -> dict:
    t = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return {"_parse_error": "no json", "_raw": text[:300]}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        try:
            return json.loads(re.sub(r",(\s*[}\]])", r"\1", m.group()))
        except Exception as e:
            return {"_parse_error": str(e), "_raw": text[:300]}


async def compare(elicitation: str, a_answers: list[str], b_answers: list[str],
                  rng: random.Random) -> dict:
    blind, label_to_group = assign_blind(a_answers, b_answers, rng)

    def fmt(ans):
        return "\n".join(f"- {x}" for x in ans)

    user = (f"The question asked of every reader was: \"{elicitation}\"\n\n"
            f"=== GROUP 1 ANSWERS ===\n{fmt(blind['Group 1'])}\n\n"
            f"=== GROUP 2 ANSWERS ===\n{fmt(blind['Group 2'])}\n\n"
            "Output exactly this JSON: {\"stronger_on_specificity\": \"Group 1\"|\"Group 2\"|\"tied\", "
            "\"what_group1_saw\": str, \"what_group2_saw\": str, "
            "\"key_difference\": str, \"summary\": str}")
    raw = await client_v2.chat(
        [{"role": "system", "content": COMPARATOR_SYSTEM}, {"role": "user", "content": user}],
        max_tokens=700, temperature=0.2)
    return {"blinding": label_to_group, "comparison": parse_comparison(raw)}
