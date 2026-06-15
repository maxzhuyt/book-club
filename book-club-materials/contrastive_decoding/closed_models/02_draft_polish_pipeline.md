# Plan 02 — Draft / polish pipeline

## Core idea

A two-pass generation where one model drafts and the other refines. Two
directions are possible; both keep the existing CD code untouched and only
add an orchestration layer.

### Direction A — Open+CD drafts, closed model polishes

```
[Qwen + CD] ──draft──► [Claude Opus] ──polished──► output
                            ▲
                            └── system prompt instructs:
                                "preserve voice, fix only X/Y/Z"
```

### Direction B — Closed model drafts, open+CD edits

```
[Claude Opus] ──draft──► [Qwen + CD as editor] ──edited──► output
                              ▲
                              └── user message contains the draft
                                  to be rewritten in voice
```

Direction A is the more defensible default; Direction B has serious problems
(see "Risks" below) and is recorded here only for completeness.

## What stays untouched

- `simulation/src/cd/decoder.py`, `prompts.py`, `alpha.py`.
- The per-persona system prompts in `personas/system_prompt/`.

## What gets added

```
closed_models/
└── runners/
    ├── simulate_closed_draft_polish.py     # Direction A
    └── simulate_open_cd_editor.py          # Direction B (optional)
```

## Direction A — implementation sketch

For each turn (Phase 1 review, Phase 2 reaction, each Phase 3 turn, Phase 4
reflection):

1. **Draft**: call `ContrastiveDecoder.generate(s_pos, s_neg, user_msg, ...)`
   exactly as `simulate_compare.py` does today. Save `draft = out["text"]`.
2. **Polish**: call OpenRouter (reuse `summarize_opus.py:call_opus`) with:

   ```
   SYSTEM:
   You are a copy-editor for a single book-club participant whose voice
   must be preserved exactly. Below is a draft of their book-club post.
   Your job:
     • Fix grammatical errors, awkward phrasing, broken sentences,
       and obvious word-substitution glitches.
     • Tighten only when the draft is clearly meandering.
     • PRESERVE: vocabulary register, sentence length distribution,
       hedging vs. assertive tone, opinion strength, idiosyncratic
       diction, and any reference to specific passages or characters.
     • Do NOT add diplomatic hedges, balanced framings, or "on the other
       hand" structures that the draft does not already use.
     • Do NOT make the draft sound more articulate, more analytical, or
       more polished than the original voice level.
     • Length must stay within ±15% of the input.

   USER:
   <draft>
   ```

3. Persist both the draft and the polished version in
   `simulation/outputs_closed_draft_polish/transcripts/`. The polished
   version is the canonical one consumed by Phase 2/3/4 history; the draft
   is kept for ablation.

## Direction B — implementation sketch (optional)

The closed model produces a substantive draft from the persona system prompt
alone. Then CD is invoked with:
- `s_pos = persona system prompt + "Rewrite the user's draft in your own voice"`
- `s_neg = generic-reviewer system prompt + same rewrite instruction`
- `user_msg = "Here is a draft of a book-club post for you to rewrite in your own voice:\n\n<draft>"`

Generation proceeds as a normal CD call. This is one extra call per turn to
`ContrastiveDecoder.generate()` with no code changes.

## Risks

### Direction A (open→closed polish)
- **Polish regresses to assistant voice** — the closed model's prior is the
  very thing CD was suppressing. A constrained system prompt helps but does
  not erase the bias. Quantitative guard: measure stylistic distance between
  draft and polished; reject polishes that move > X toward the
  generic-reviewer feature vector.
- **Voice drift on disagreement intensity** — the closed model tends to
  soften strong claims. Add an explicit clause: "If the draft says X is
  *wrong* or *the worst part*, keep that intensity; do not soften to
  *less effective*."
- **Two-model latency** — every turn now pays one CD pass *and* one
  closed-model pass. Sequential. Phase 3 with 2 rounds × 8 personas = 16 of
  these chains; budget accordingly.

### Direction B (closed→CD edit)
- **CD's context budget is short** — passing a long Opus draft as the user
  message means the open model spends most of its window on input, leaving
  little for generation.
- **CD is doing the wrong thing** — CD steers token-by-token toward persona
  voice given the *prefix it has already emitted*. Editing-mode generation
  works against this; the model is conditioned on someone else's full draft.
- **Wastes the strongest signal** — the closed model's substantive content is
  what we wanted; running it through a weaker model as a stylistic filter is
  the opposite of distillation.

I'd recommend Direction A only, and treating Direction B as a control to
demonstrate why the natural direction is open→closed, not closed→open.

## Cost and evaluation

- **Cost (Direction A)**: open-model CD time (same as the existing
  `simulate_compare.py`) plus one Opus call per turn (~80–500 output tokens
  each). Roughly the cost of running `summarize_opus.py` ×N turns.
- **Evaluation**: same metrics as Plan 01. Compare draft (Qwen+CD only) to
  polished (Qwen+CD → Opus polish). Pass criterion: polished should match
  or beat draft on coherence/grammar without losing more than a small
  amount on distinctiveness or anti-conformity.
