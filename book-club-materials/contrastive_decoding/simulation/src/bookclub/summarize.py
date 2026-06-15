"""
Produce a concise group summary from a finished simulation:
- What the group enjoyed (specifics)
- What they want improved
- Anti-conformity check: who shifted opinion and why
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cd.decoder import ContrastiveDecoder, GenerationConfig, load_dual_models


SUMMARIZER_SYSTEM = """You are a careful, neutral note-taker who has been
asked to summarize a book-club discussion. You do NOT have opinions of your
own.  You FAITHFULLY report what the readers said — never invent details that
are not in the transcript. You quote or attribute every claim to a specific
reader by their short ID (e.g. "Reader-C felt..."). When readers disagreed,
say so explicitly. When opinions shifted between Phase 1 and Phase 4, name
the shift and who shifted."""


def build_user_msg(transcripts_dir: Path, manifest: dict, story_title: str) -> str:
    p1 = json.loads((transcripts_dir / "phase1_private.json").read_text())
    p2 = json.loads((transcripts_dir / "phase2_reactions.json").read_text())
    p3 = json.loads((transcripts_dir / "phase3_discussion.json").read_text())
    p4 = json.loads((transcripts_dir / "phase4_reflections.json").read_text())

    parts: list[str] = []
    parts.append(f"# Book club discussion of \"{story_title}\"\n")

    parts.append("\n## Cast\n")
    for p in manifest["personas"]:
        marker = " [6yo]" if p.get("six_year_old") else ""
        parts.append(
            f"- {p['short_id']}: {p['label']} (user_{p['user_id']}){marker}, "
            f"alpha P1/P3/P4 = "
            f"{p['alpha']['phase1']:.2f}/{p['alpha']['phase3']:.2f}/{p['alpha']['phase4']:.2f}"
        )

    parts.append("\n## Phase 1 — Private Reviews (independent, no peer info)\n")
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

    parts.append("\n## Phase 4 — Reflections\n")
    for sid, body in p4.items():
        parts.append(f"\n### {body['speaker']}\n{body['reflection']}\n")

    transcript = "\n".join(parts)

    instructions = (
        "\n\n---\n\n"
        "Read the full transcript above carefully. Now produce a concise summary "
        "in Markdown with the following sections:\n\n"
        "## What the group enjoyed (with specifics)\n"
        "  - 4-7 bullets. Each bullet must name (a) a specific element of the "
        "story (a line, character moment, image, structural choice) and (b) "
        "WHO appreciated it (by short ID).\n\n"
        "## Where readers disagreed\n"
        "  - 2-5 bullets. Each bullet captures a genuine disagreement, who "
        "took which side, and what specific aspect of the story it was about.\n\n"
        "## Improvements the group would like to see\n"
        "  - 3-6 bullets. Concrete craft-level requests (clarity, pacing, "
        "characterization, ending, etc.), attributed to specific readers.\n\n"
        "## Whose opinion shifted, and why\n"
        "  - For each reader whose Phase-4 reflection indicates a shift from "
        "their Phase-1 private review, write one line: who shifted, in which "
        "direction, and which specific peer argument moved them. For readers "
        "who held firm, also note who and why.\n\n"
        "Use ONLY information present in the transcript. Do NOT invent details. "
        "Keep the summary under 700 words."
    )

    return transcript + instructions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--outputs-dir", type=Path,
                    default=REPO_ROOT / "outputs")
    ap.add_argument("--story-title", type=str, default="Love in the Limelight")
    args = ap.parse_args()

    manifest = json.loads((args.outputs_dir / "manifest.json").read_text())
    user_msg = build_user_msg(args.outputs_dir / "transcripts",
                              manifest, args.story_title)
    (args.outputs_dir / "summary_input.md").write_text(user_msg)

    # For the summarizer we want NO contrastive steering — just the neutral
    # assistant prior. alpha=0 with s_pos == s_neg gives standard decoding.
    print("[summarize] loading model...", flush=True)
    model_pos, model_neg, tok = load_dual_models(args.model_id)
    decoder = ContrastiveDecoder(model_pos, tok, alpha=0.0,
                                 model_neg=model_neg)

    cfg = GenerationConfig(
        max_new_tokens=1200,
        temperature=0.3,
        top_p=0.9,
        seed=42,
    )
    out = decoder.generate(
        s_pos=SUMMARIZER_SYSTEM,
        s_neg=SUMMARIZER_SYSTEM,
        user_msg=user_msg,
        cfg=cfg,
        alpha=0.0,
    )
    summary_path = args.outputs_dir / "group_summary.md"
    summary_path.write_text(out["text"])
    print(f"[summarize] wrote {summary_path}", flush=True)
    print("\n----- SUMMARY -----\n" + out["text"], flush=True)


if __name__ == "__main__":
    main()
