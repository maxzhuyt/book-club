# Meta — Improving closed frontier models with open+CD for book-club personas

*Written 2026-05-15. Index and reasoning over the five individual plans
plus one explicit combination plan in this folder.*

---

## What problem is being solved

The CD paper (Dong et al. 2026) shows that token-level contrastive decoding
on the system prompt steers an open model into much sharper persona
behavior than vanilla system-prompt-only prompting. This works because CD
operates on the logits — the post-training "assistant attractor" is
amplified *away from*, instead of just being competed against by the
persona instruction.

Closed frontier models (Claude, GPT, Gemini) have stronger reasoning,
broader world knowledge, and better long-context coherence than the 14B
open model we run CD on (`Qwen/Qwen2.5-14B-Instruct`). But they do not
expose logits, which makes CD's mechanism unavailable. The book-club
simulation suffers a real cost from this — the closed model's default
voice flattens all 8 adult personas into a single articulate-reviewer
register, eroding distinctiveness and (we suspect) accelerating
consensus convergence.

The question is whether the *open* CD signal can be repurposed to lift the
*closed* model's persona behavior, given that the only interface to the
closed model is the input prompt and its raw output.

## The five plans in one paragraph each

- **[01 — Few-shot distillation](01_few_shot_distillation.md)**: run
  open+CD offline to produce a small corpus of persona-faithful book-club
  posts per phase, embed 2 of them as voice exemplars into the closed
  model's system prompt at runtime. Cheapest plan that actually moves the
  needle. Builds a reusable artifact (the exemplar bank) that other plans
  also consume.

- **[02 — Draft / polish pipeline](02_draft_polish_pipeline.md)**: open+CD
  drafts, closed model polishes (Direction A); or closed model drafts,
  open+CD edits (Direction B). Direction A regresses to assistant voice on
  the polish step. Direction B wastes the closed model's substantive
  reasoning. Recorded mostly to document why this natural-looking idea
  doesn't pay off.

- **[03 — Best-of-N with CD-derived scorer](03_best_of_n_cd_scorer.md)**:
  closed model samples N candidates; an offline CD-derived scorer (logprob
  delta under z_pos / z_neg, or embedding distance to a CD reference bank)
  picks the most persona-faithful. CD becomes a discriminator rather than
  a generator. Strong on its own; stronger combined with Plan 01.

- **[04 — Prompt amplification from CD diagnostics](04_prompt_amplification.md)**:
  mine the CD-vs-no-CD generation deltas offline to produce persona-
  specific natural-language voice instructions, inject them into the
  closed-model system prompt. Zero per-turn cost, smaller fidelity gain.
  Best as a base layer beneath Plan 01 or Plan 03.

- **[05 — Iterative critique loop](05_critique_loop.md)**: closed model
  drafts; open+CD generates an in-persona critique pointing out off-voice
  passages; closed model revises. Strongest fidelity ceiling but the most
  expensive plan (3× per-turn cost). Use as a quality bound, not a default.

- **[06 — Combined: few-shot + best-of-N (recommended starting point)](06_combo_few_shot_plus_bon.md)**:
  stack Plan 01 and Plan 03 — biased samples + good selector. The recommended
  first build; clears the largest fraction of the gap per unit of complexity.

## Ranking and reasoning

### Top choice: Plan 06 (Plan 01 + Plan 03 combined)

I'd build this first. The reasoning has three legs.

1. **The two mechanisms attack orthogonal failures.** Plan 01 lifts the
   *distribution* the closed model samples from — every candidate is more
   likely to be persona-faithful. Plan 03 picks the best one. Without
   Plan 01, BoN's ceiling is capped by the assistant-baseline distribution;
   without Plan 03, exemplars are a one-shot intervention with no
   correction signal.

2. **One artifact, two uses.** Both plans need the same offline CD corpus
   — generations of the persona under varying seeds and phases. Building
   that corpus is the single most expensive new step; amortizing it across
   two uses is a strict win. Plan 01 reads the corpus into the prompt;
   Plan 03 uses it as an embedding reference (Score B) and/or computes the
   logprob delta from the same models that produced it (Score A).

3. **It fits the existing architecture with minimal new surface area.**
   The closed-model runner is a near-clone of
   `simulation/src/bookclub/simulate_compare.py` with `decoder.generate()`
   swapped for `summarize_opus.py:call_opus`. The exemplar builder is one
   pass over the existing CD pipeline. The Score A scorer reuses
   `cd.decoder.load_dual_models` and the chat-template encoding — no
   changes to the CD decoder itself.

The plan has a documented failure mode (the six-year-old persona, who
exists at content the closed model simply doesn't produce well under any
prompt). For that persona we should keep the existing Qwen+CD path and
not waste closed-model compute. This is a clean per-persona dispatch, not
a system-wide compromise.

### Second tier: Plan 04 as a base layer under Plan 06

Plan 04 (prompt amplification) and Plan 06 (combined) are not exclusive —
the amplified prompt is just text inserted into the closed-model system
prompt. Stacking it under Plan 06 costs only +400–600 input tokens per
call and gives the closed model an explicit description of the persona's
voice axes *in addition to* the exemplars and the BoN scoring. This is
the path I'd take to ratchet up fidelity once Plan 06 is working.

The reason Plan 04 is a layer rather than a standalone first choice: prompt
instructions can be ignored, and the gain ceiling on a description-only
intervention is lower than on a demonstration-plus-selection intervention.
Plan 04 in isolation is the right call only if per-call latency must be
minimized (one closed-model call, no extra inference compute) — e.g., if
the simulation has to run inside a tight budget. For the current book-club
simulation, that constraint does not apply.

### Third tier: Plan 05 as the upper-bound experiment

The critique loop is the most expensive plan and the most likely to give a
near-ceiling result. It is worth building once, on a single phase
(Phase 3 — the one most prone to consensus collapse), as a quality bound:
*if Plan 06 + Plan 04 are within X% of Plan 05 on the chosen metric, the
cheaper stack is the production path*. If Plan 05 wins by a wide margin,
the iterative correction signal is genuinely doing something the static
plans can't, and it's worth productionizing.

Use Plan 05 as an experiment, not as a default.

### Plans I'd deprioritize: Plan 02

Plan 02 has an attractive shape — "draft and polish" is a common LLM
pattern — but the dynamics don't favor it here.

- **Direction A** (open+CD draft → closed polish): the polish step's job
  is *not* to add reasoning or content, but to fix grammar while preserving
  voice. The closed model's instruction-following on "preserve voice"
  cashes out, in practice, as "lightly polish toward generic articulate
  English" — i.e., it pulls in exactly the direction CD was suppressing.
  The metric we care about, voice distinctiveness, is the metric the
  polish step erodes.

- **Direction B** (closed draft → open+CD edit) wastes the closed model's
  best output (substantive content) by passing it through a 14B model
  whose stylistic editing capability is limited and whose context budget
  on the open model is short.

Plan 02 is documented because the failure mode is informative: it makes
clear *why* the right direction is to use CD as a generator (Plan 01,
prompt material) or a discriminator (Plan 03, scorer) — not as a
rewriter. Building it as a control to confirm the failure mode would be
informative but not on the critical path.

### Plans I considered and rejected from the final set

- **Synthetic-data fine-tuning** (generate a large CD corpus, fine-tune
  the closed model on it). Cleanest "true distillation" path, and would
  almost certainly produce the best result *if* the closed model is fine-
  tunable. Claude (the model already wired up via `summarize_opus.py`) is
  not currently customer-fine-tunable. GPT-4o-mini and Gemini are. Worth
  revisiting if the project switches to a fine-tunable provider; not a
  current option for this codebase.

- **Hybrid generation** (closed model does narrative reasoning, open+CD
  generates the dialogue). The book-club format is monologue-style review
  posts, not dialogue; the natural split point isn't there. If the
  simulation later grows into a conversational format (Phase 3 turning
  into live exchange rather than written replies), this becomes
  interesting.

- **Token-level mixing via API logprobs**. OpenRouter exposes top-k
  logprobs from some providers (not all). Even where exposed, top-k
  rather than full-vocabulary logits makes the CD combination ill-defined.
  Skipped.

## What to do if you can only build one thing

Build **Plan 06**. Per-persona dispatch: route the six-year-old through
the existing Qwen+CD pipeline; route the eight adult personas through
Plan 06.

## What to do if you can build two things

Plan 06, then add Plan 04 as a system-prompt layer underneath it. The
combined system uses:
- the existing 3-layer persona scaffold (identity + voice exemplars +
  rules),
- the Plan 04 voice-diagnostic supplement appended to the scaffold,
- Plan 01's two CD-generated book-club exemplars below that,
- the closed model sampling N candidates at T=0.8,
- Plan 03's two-stage scorer (embedding distance → CD logprob delta)
  picking the winner.

## What to do if you can build three

Add Plan 05 as a Phase-3-only critique pass on personas where the
Plan 06 + Plan 04 output's CD logprob delta falls below a threshold.
This routes per-turn: most turns finish at the BoN winner; only the
turns where the scorer signals low fidelity pay the critique-loop cost.

## Required additions to the existing repo

None of these plans modify `simulation/src/cd/`. They add:
- one offline corpus-builder per plan that calls `ContrastiveDecoder.generate()`
  with existing prompts and saves text outputs;
- one runner per plan that mirrors `simulate_compare.py` with the
  open-model `decoder.generate()` call replaced by an OpenRouter
  `call_opus` (reuse `summarize_opus.py:call_opus` and `load_key`,
  default key name `"Hoyt"`);
- for Plan 03 and Plan 06, a scoring helper that loads the same dual-GPU
  models the CD decoder uses and does forward-only passes (no
  modification to the CD loop).

Output directories follow the existing pattern (`outputs_closed_*/`) and
remain compatible with `summarize_opus.py` for end-of-run summarization.

## Open questions worth resolving before building

1. **Held-out story for exemplars.** Plans 01, 03, 06 all need CD outputs
   on a story that is *not* the eval target. Either pick a third story
   from `personas/` or generate a synthetic micro-story. Note in the
   exemplar manifest which story was used.
2. **N for BoN.** The plans assume N=6 per turn as a reasonable default.
   Phase 1 may need N=2; Phase 3 may need N=8. Worth a small pilot at
   N ∈ {2, 4, 8} before committing.
3. **Per-persona α applied at exemplar-generation time.** The exemplar
   bank should use the same α the production simulator uses (per
   `cd/alpha.py`). This is already the case if `build_exemplars.py`
   copies `simulate_compare.py`'s call site verbatim — just don't forget.
4. **Six-year-old policy.** The dispatch suggestion above ("keep her on
   Qwen+CD") is a recommendation, not a hard call. Worth A/B-ing once on
   a single phase to confirm closed+exemplars genuinely underperforms
   Qwen+CD on her, so the policy is empirical rather than assumed.

---

*Plans referenced:
[01](01_few_shot_distillation.md),
[02](02_draft_polish_pipeline.md),
[03](03_best_of_n_cd_scorer.md),
[04](04_prompt_amplification.md),
[05](05_critique_loop.md),
[06](06_combo_few_shot_plus_bon.md).*
