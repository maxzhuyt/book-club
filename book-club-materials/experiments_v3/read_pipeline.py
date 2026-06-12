"""Two-phase read: system = persona + probe prime; user1 = story + generic question;
assistant1 = pass-1; user2 = probe elicitation; assistant2 = pass-2."""
from __future__ import annotations
import client_v2
import probes as _p


def build_pass1_messages(persona_prompt: str, probe: dict, story: str) -> list[dict]:
    system = persona_prompt.rstrip() + "\n\n--- ATTENTIONAL FOCUS ---\n" + probe["prime"]
    user1 = f"=== PASSAGE ===\n{story}\n=== END PASSAGE ===\n\n{_p.GENERIC_PASS1}"
    return [{"role": "system", "content": system},
            {"role": "user", "content": user1}]


def build_pass2_messages(pass1_messages: list[dict], pass1_answer: str, probe: dict) -> list[dict]:
    return pass1_messages + [
        {"role": "assistant", "content": pass1_answer},
        {"role": "user", "content": probe["elicitation"]},
    ]


async def run_two_phase(persona_prompt: str, probe: dict, story: str,
                        max_tokens: int = 3000, temperature: float = 0.75) -> dict:
    m1 = build_pass1_messages(persona_prompt, probe, story)
    pass1 = await client_v2.chat(m1, max_tokens, temperature)
    m2 = build_pass2_messages(m1, pass1, probe)
    pass2 = await client_v2.chat(m2, max_tokens, temperature)
    return {"pass1": pass1, "pass2": pass2}
