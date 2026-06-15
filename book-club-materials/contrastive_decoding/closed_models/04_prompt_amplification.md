# Plan 04 — Prompt amplification from CD diagnostics

## Core idea

CD reveals, at the token level, the *difference* between the persona's
distribution and the generic-reviewer distribution. That difference can be
mined offline to produce **better natural-language prompt instructions**
for the closed frontier model — instructions that target the specific axes
where vanilla prompting fails to elicit the persona.

The closed model never sees CD at runtime. It sees a system prompt that has
absorbed CD's insights. Cost at inference is zero beyond the (very modest)
input-token cost of a longer system prompt.

This is the most analytical of the plans; the payoff is a one-time prompt
upgrade that compounds with every other plan in this folder.

## What stays untouched

- `simulation/src/cd/decoder.py`, `prompts.py`, `alpha.py` — used as-is to
  generate the diagnostic corpus.

## What gets added

```
closed_models/
├── diagnostics/
│   ├── extract_persona_signal.py   # offline: produces JSON per persona
│   └── signals/
│       └── <short_id>.json
└── runners/
    └── simulate_closed_amplified.py
```

## Pipeline

### Step 1 — Produce paired generations (offline, one-time)

For each persona, for each phase, generate two completions on the *same*
seed and *same* user message:
- `text_cd`  = `ContrastiveDecoder.generate(s_pos, s_neg, ..., alpha=α_persona)`
- `text_no_cd` = `ContrastiveDecoder.generate(s_pos, s_neg, ..., alpha=0.0)`
  (α=0 collapses CD to the positive-only baseline, i.e. the model the closed
  frontier baseline is closest to)

Generating both via the same `ContrastiveDecoder.generate()` keeps decoder
parameters identical; only α differs. No code change required.

Save 3–5 pairs per persona per phase, sharing seeds across α settings.

### Step 2 — Mine the deltas

`extract_persona_signal.py` reads the pairs and computes:

1. **Lexical drift**: tokens overrepresented in `text_cd` vs. `text_no_cd`
   (log odds ratio with Dirichlet prior, top-30). These are words the
   persona uses that the assistant prior suppresses.
2. **Structural drift**: avg words, avg sent length, paragraph count,
   bullet/list use, hedging frequency (count of *I think*, *perhaps*, *it
   could be argued*), assertion frequency (*X is wrong*, *the worst*, *bad*).
3. **Stance drift**: rating-like phrases, opinion polarity (small VADER pass
   or per-sentence sentiment from the embedder).
4. **Diction class**: presence of slang, ALL-CAPS emphasis, sentence
   fragments, run-on chains, parenthetical asides.

Optionally feed both texts to an LLM (could be Claude itself, ironically,
or a local Qwen) with a prompt:

```
Compare DRAFT_A and DRAFT_B written by the same person. List the top 5
ways DRAFT_A's voice differs from DRAFT_B's — be specific about vocabulary,
sentence structure, hedging, and opinion strength. Output JSON.
```

LLM-based drift extraction is more interpretable than raw statistics and
goes into the persona's `signals/<short_id>.json`.

### Step 3 — Render signals as prompt clauses

Convert each persona's `signals/<short_id>.json` into a natural-language
"voice supplement" inserted into the closed-model system prompt:

```
--- VOICE DIAGNOSTICS (derived from your contrastive-decoded outputs) ---
When responding, you tend to:
  • Use the words "weird", "kinda", "ugh", and "lol" frequently. Do not
    avoid them in favor of polished synonyms.
  • Write sentences shorter than 14 words on average. Avoid stacked
    subordinate clauses.
  • Drop hedges: never say "perhaps" or "it could be argued". State your
    opinion as fact.
  • Use 0-1 paragraph breaks per response, not 3-4.
  • Mark strong feelings with ALL CAPS on a single word, not with adverbs.
  • Reference characters by their nicknames, not their full names.
```

These clauses are persona-specific, but the *generation* of them is
fully automated from the diagnostic JSON. A simple template per signal
type.

### Step 4 — Use the amplified prompt at inference

`simulate_closed_amplified.py` is a near-clone of `simulate_compare.py`
that:
- swaps the open-model generate call for an OpenRouter `call_opus`,
- inserts the persona's voice-diagnostics block between Layer 2 (voice
  exemplars) and Layer 3 (discussion rules) of the existing system prompt.

Output dir: `simulation/outputs_closed_amplified/`.

## What this is *not*

Not a substitute for runtime CD on the open model — the diagnostics are
descriptions of behavior, not control over generation. A description can be
ignored; CD's logit shift cannot. Expect a smaller fidelity gain than
Plan 01 or Plan 03, but at zero per-turn cost.

This is most useful **combined with** Plan 01 or Plan 03: the amplified
prompt corrects the persona's high-level voice axes; the exemplars or the
BoN scorer correct micro-level token choices.

## Cost and risk

- **Cost**: one-time CD bank for diagnostics (a few hours), one-time
  signal extraction (an hour of Python + optional LLM calls), zero runtime
  cost beyond +400–600 tokens per system prompt.
- **Risk — overfitting prompts to CD artifacts**: if α is at the cap, the
  diagnostic captures CD's failure modes too (e.g., flat repetition). Hand-
  filter the signal JSON before committing it to the prompt.
- **Risk — instruction-following ceiling**: closed models follow some
  instructions better than others. "Use the word 'kinda'" is followable;
  "drop hedges" is harder to operationalize. Mitigation: use *examples*
  in the clauses where rules underspecify ("Say 'the dog was sad' not
  'one might argue the dog appeared somewhat melancholic'").

## How to evaluate

Same metrics as Plan 01. Run Arm C-amplified against Arm C-vanilla on the
same story. Pass criterion: distinctiveness rises; consistency on
held-out style metrics rises; convergence/conformity stays flat or drops.

If Arm C-amplified gains < 25% of the gap between Arm C-vanilla and
Arm Q (open+CD), prompt amplification alone is too weak and should only be
used as a layer beneath Plan 01 or Plan 03.
