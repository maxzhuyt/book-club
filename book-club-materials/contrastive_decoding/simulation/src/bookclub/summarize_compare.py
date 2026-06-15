"""
Produce a comparative group summary from a finished two-version run:
- What the group enjoyed about each version (with specifics)
- What improvements they want for each version
- Vote tally (parsed from "VOTE: A" / "VOTE: B" lines in Phase 4)
- Overall recommendation with the single strongest reason
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cd.decoder import ContrastiveDecoder, GenerationConfig, load_dual_models


SUMMARIZER_SYSTEM = """You are a careful, neutral note-taker who has been
asked to summarize a book-club comparative discussion of two versions
("A" and "B") of the same story.  You do NOT have opinions of your own.
You FAITHFULLY report what the readers said — never invent details that
are not in the transcript. You quote or attribute every claim to a
specific reader by their short ID (e.g. "Reader-C felt..."). When
readers disagreed, say so explicitly."""


VOTE_RE = re.compile(r"VOTE\s*:?\s*([AB])", re.IGNORECASE)


def parse_votes(p4: dict) -> dict[str, str]:
    """Return short_id -> 'A' | 'B' | 'unclear'."""
    votes: dict[str, str] = {}
    for sid, entry in p4.items():
        text = entry["reflection"]
        # search only the first ~120 chars, where the VOTE line is meant to be
        head = text[:300]
        m = VOTE_RE.search(head)
        if not m:
            m = VOTE_RE.search(text)
        votes[sid] = m.group(1).upper() if m else "unclear"
    return votes


def build_user_msg(transcripts_dir: Path, manifest: dict,
                   votes: dict[str, str]) -> str:
    title_a = manifest.get("story_a_title", "Version A")
    title_b = manifest.get("story_b_title", "Version B")
    p1 = json.loads((transcripts_dir / "phase1_private.json").read_text())
    p2 = json.loads((transcripts_dir / "phase2_reactions.json").read_text())
    p3 = json.loads((transcripts_dir / "phase3_discussion.json").read_text())
    p4 = json.loads((transcripts_dir / "phase4_reflections.json").read_text())

    parts: list[str] = []
    parts.append(f"# Comparative book-club discussion: "
                 f"\"{title_a}\" vs \"{title_b}\"\n")

    parts.append("\n## Cast and parsed Phase-4 votes\n")
    for p in manifest["personas"]:
        v = votes.get(p["short_id"], "unclear")
        parts.append(
            f"- {p['short_id']}: {p['label']} (user_{p['user_id']}), "
            f"alpha P1/P3/P4 = "
            f"{p['alpha']['phase1']:.2f}/{p['alpha']['phase3']:.2f}/"
            f"{p['alpha']['phase4']:.2f} — VOTE: {v}"
        )

    parts.append("\n## Phase 1 — Private Comparative Reviews\n")
    for sid, body in p1.items():
        parts.append(f"\n### {body['speaker']}\n{body['review']}\n")

    parts.append("\n## Phase 2 — Broadcast Reactions\n")
    for sid, body in p2.items():
        parts.append(f"\n### {body['speaker']}\n{body['reaction']}\n")

    parts.append("\n## Phase 3 — Moderated Discussion\n")
    for i, rr in enumerate(p3, start=1):
        parts.append(f"\n### Round {i}\n")
        for e in rr:
            parts.append(f"\n**{e['speaker']}**\n{e['text']}\n")

    parts.append("\n## Phase 4 — Votes + Reflections\n")
    for sid, body in p4.items():
        parts.append(f"\n### {body['speaker']}\n{body['reflection']}\n")

    transcript = "\n".join(parts)

    # Vote tally string
    tally_a = sum(1 for v in votes.values() if v == "A")
    tally_b = sum(1 for v in votes.values() if v == "B")
    tally_unclear = sum(1 for v in votes.values() if v == "unclear")
    tally_line = f"Phase-4 vote tally: A={tally_a}, B={tally_b}, unclear={tally_unclear}"

    instructions = (
        "\n\n---\n\n"
        f"{tally_line}\n\n"
        "Read the full transcript above carefully. Produce a concise "
        "summary in Markdown with EXACTLY the following sections, in this "
        "order. Use ONLY information present in the transcript. Do NOT "
        "invent details. Keep the whole summary under 900 words.\n\n"
        f"## What the group enjoyed about Version A (\"{title_a}\")\n"
        "  3-6 bullets. Each bullet names (a) a specific element of the "
        "story (a line, character moment, image, structural choice) and "
        "(b) WHO appreciated it (by short ID).\n\n"
        f"## What the group enjoyed about Version B (\"{title_b}\")\n"
        "  Same format. 3-6 bullets.\n\n"
        f"## Improvements the group wants for Version A\n"
        "  2-5 bullets. Concrete craft-level requests, attributed.\n\n"
        f"## Improvements the group wants for Version B\n"
        "  Same format. 2-5 bullets.\n\n"
        "## Where readers disagreed about which version is stronger\n"
        "  2-4 bullets, each naming the disagreement and who took which "
        "side.\n\n"
        "## Vote tally\n"
        f"  Restate: {tally_line}. Then list each reader's vote.\n\n"
        "## Recommendation to the author\n"
        "  State the panel's overall recommendation (A or B). Give the "
        "single strongest collective reason. Then give ONE caveat: the "
        "best argument the minority made against this choice.\n"
    )

    return transcript + instructions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str,
                    default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--outputs-dir", type=Path,
                    default=REPO_ROOT / "outputs_compare")
    args = ap.parse_args()

    manifest = json.loads((args.outputs_dir / "manifest.json").read_text())
    p4 = json.loads((args.outputs_dir / "transcripts"
                     / "phase4_reflections.json").read_text())
    votes = parse_votes(p4)
    (args.outputs_dir / "votes.json").write_text(
        json.dumps(votes, indent=2)
    )

    user_msg = build_user_msg(
        args.outputs_dir / "transcripts", manifest, votes,
    )
    (args.outputs_dir / "summary_input.md").write_text(user_msg)

    print("[summarize] loading model...", flush=True)
    model_pos, model_neg, tok = load_dual_models(args.model_id)
    decoder = ContrastiveDecoder(model_pos, tok, alpha=0.0,
                                 model_neg=model_neg)
    cfg = GenerationConfig(
        max_new_tokens=1500, temperature=0.3, top_p=0.9, seed=42,
    )
    out = decoder.generate(
        s_pos=SUMMARIZER_SYSTEM, s_neg=SUMMARIZER_SYSTEM,
        user_msg=user_msg, cfg=cfg, alpha=0.0,
    )
    summary_path = args.outputs_dir / "group_summary.md"
    summary_path.write_text(out["text"])
    print(f"[summarize] wrote {summary_path}", flush=True)
    print("\n----- SUMMARY -----\n" + out["text"], flush=True)
    print(f"\n----- VOTES -----\n{json.dumps(votes, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
