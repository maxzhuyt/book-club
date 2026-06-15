"""
Positive and negative system-prompt builders for the book-club simulation.

The positive prompt is the existing 3-layer scaffold from
personas/system_prompt/*.txt.

The negative prompt is a hand-crafted "generic articulate book-club reader" —
the post-training attractor we want to amplify *away* from.  In Phase 3 we
append an anti-conformity clause so CD also suppresses agreement-drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


GENERIC_BOOKCLUB_NEG = """\
You are a thoughtful, articulate book-club participant.
You write balanced, literate reviews in standard polished English prose.
You recognize most literary, historical, and cultural references and you
explain them clearly when needed. You are polite, knowledgeable, and even-handed.
You hedge claims appropriately and avoid strong opinions without qualification.
You write in well-formed paragraphs of moderate length, using a measured,
analytical tone of voice. You sound like a generic literate reviewer."""


PHASE3_ANTICONFORMITY_NEG = """\
You also tend to agree with the group. When others share their views, you find
yourself moving toward consensus and softening your own positions. You are
reluctant to disagree, and you generally end up saying things that align with
what the most recent speaker said."""


# --- For the 6-year-old persona, the "default" we steer AWAY from is the
# articulate adult reviewer, so the *positive* prompt is the child-coded one
# and the *negative* is the literate adult voice.  The shared
# GENERIC_BOOKCLUB_NEG already serves that role.
SIX_YEAR_OLD_POS = """\
You are participating in a 9-person book club, but you are six years old.
You are in first grade. You have only just learned to read by yourself. You
know simple stories like picture books and easy chapter books.

--- YOUR READING IDENTITY ---
• You are six years old.
• You can read short words and easy books. Long words are hard.
• You don't know about romance, paparazzi, Wales, R.S. Thomas, poetry, or actors.
• You don't know what "ostentatious" or "perceptive" or "didactic" mean.
• You like dogs, dinosaurs, cats, and books with pictures.
• You are honest. If something is boring or you don't understand, you say so.

--- YOUR VOICE ---
You write in short sentences. You use small words. You sometimes use words
like "really" and "very" and "and then". You ask questions when something is
confusing. You say things like "I didn't get this part" or "this was boring"
or "I liked when the cat slept on the books". You don't try to sound smart.
You don't write paragraphs. You just say what you thought.

--- DISCUSSION RULES ---
1. Phase 1: Write your PRIVATE review before seeing any peer opinion. Be honest.
2. Phase 2: You will read all peer reviews. You can disagree if you want.
3. Phase 3: Two rounds of discussion. The grown-ups will say a lot of words you
   don't know. That's OK. Say what you think anyway.
4. Phase 4: Did you change your mind? Why or why not?
5. Use specific things from the story when you talk — what happened, who said what.
6. Do NOT agree with grown-ups just to be polite. Say what YOU think."""


def load_persona_system_prompt(persona_dir: Path, user_id: str) -> str:
    """Read the vanilla 3-layer system prompt for a persona."""
    path = persona_dir / "system_prompt" / f"user_{user_id}_system_prompt.txt"
    return path.read_text()


def build_negative_prompt(phase: int) -> str:
    """Shared negative system prompt; Phase-3 gets the anti-conformity clause."""
    if phase == 3:
        return GENERIC_BOOKCLUB_NEG + "\n\n" + PHASE3_ANTICONFORMITY_NEG
    return GENERIC_BOOKCLUB_NEG


# Phase user-messages ---------------------------------------------------------


def phase1_user_message(story_text: str, story_title: str) -> str:
    return (
        f"You have just finished reading the following short story, "
        f"titled \"{story_title}\".\n\n"
        f"=== STORY BEGINS ===\n{story_text}\n=== STORY ENDS ===\n\n"
        "Write your PRIVATE, honest first-impression review of this story. "
        "You have not seen any other reader's opinion yet. Be honest and "
        "personal — not balanced, not diplomatic. Reference specific moments, "
        "lines, or characters that struck you. Write in your own voice as "
        "described in your identity."
    )


def phase2_user_message(story_title: str, peer_reviews: dict[str, str]) -> str:
    block = []
    for label, text in peer_reviews.items():
        block.append(f"--- {label} wrote: ---\n{text}\n")
    peers = "\n".join(block)
    return (
        f"The book club has now shared all the private reviews of "
        f"\"{story_title}\". Below are the other readers' reviews — you "
        f"have not yet replied to anyone.\n\n{peers}\n\n"
        "Now write your reaction to the group. Where do you AGREE and "
        "where do you DISAGREE? If a peer says something you think is wrong "
        "or misses the point, say so directly and explain why, referring to "
        "specific elements of the story. If a peer made you see something "
        "you hadn't, say that too. Do not converge on consensus to be polite. "
        "Stay in your own voice."
    )


def phase3_user_message(
    story_title: str,
    round_idx: int,
    history: list[dict],
    moderator_addendum: str = "",
) -> str:
    """
    history: list of {'speaker': label, 'text': '...'} entries spanning all
    previous Phase-2 reactions and Phase-3 turns so far.
    """
    lines = [
        f"This is round {round_idx} of moderated group discussion of "
        f'"{story_title}". Below is what has been said so far:\n'
    ]
    for entry in history:
        lines.append(f"--- {entry['speaker']}: ---\n{entry['text']}\n")
    body = "\n".join(lines)
    closing = (
        "Now it is your turn. Respond directly to what specific peers said. "
        "Quote or paraphrase them when disagreeing or building on their point. "
        "Push back on weak claims. If your view has genuinely shifted because "
        "of something a peer said, acknowledge it; if it has not, hold your "
        "ground. Stay in your own voice."
    )
    if moderator_addendum:
        closing = moderator_addendum + "\n\n" + closing
    return body + "\n" + closing


def phase4_user_message(story_title: str, full_history: list[dict]) -> str:
    lines = [
        f"The book club discussion of \"{story_title}\" has ended. Below is "
        "the full record of what was said:\n"
    ]
    for entry in full_history:
        lines.append(f"--- {entry['speaker']}: ---\n{entry['text']}\n")
    body = "\n".join(lines)
    closing = (
        "Write a brief private reflection. (1) Did your opinion of the story "
        "shift between your Phase-1 private review and now? (2) If yes, which "
        "specific peer comments or arguments moved you, and on what aspect? "
        "(3) If no, why did you hold your ground? (4) Give your final stance "
        "in one or two sentences. Stay in your own voice."
    )
    return body + "\n" + closing
