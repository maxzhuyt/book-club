# Plan 06 — Combined: few-shot exemplars + best-of-N CD reranking

## Core idea

Stack Plan 01 (few-shot exemplars in the system prompt) and Plan 03
(best-of-N with a CD-derived scorer). The exemplars push every closed-model
sample toward the persona's voice; the scorer picks the sample that
actually landed there. Compared to either plan alone, the combination
attacks both halves of the persona-fidelity gap:

- Plan 01 alone: every sample lives in roughly the right voice region, but
  one of them is closest to the persona's center; we have no way to pick.
- Plan 03 alone: samples are distributed widely, often with most candidates
  in the generic-reviewer region, which limits the ceiling of "best of N."
- Combined: candidates are biased toward the right region (so the ceiling
  rises), and the scorer picks the best (so we sit at the ceiling).

This is the plan I'd build first if I were going to build only one of
these. Detailed reasoning in `00_meta.md`.

## What stays untouched

- `simulation/src/cd/decoder.py`, `prompts.py`, `alpha.py`.

## What gets added

```
closed_models/
├── exemplars/                        # from Plan 01
│   ├── build_exemplars.py
│   └── bank/
├── scoring/                          # from Plan 03
│   ├── cd_scorer.py
│   └── reference_bank/
└── runners/
    └── simulate_closed_fewshot_bon.py
```

The exemplar bank and the reference bank are *the same data* — generate
once, used twice. Plan 01 uses them as prompt content; Plan 03 uses them
as embedding-distance reference vectors. Build them as one corpus with the
fields `text` + `tokens` + manifest.

## Per-turn pipeline

```python
# system prompt with few-shot exemplars (Plan 01)
sys_prompt = render_with_exemplars(
    persona=p,
    phase=phase,
    bank_dir=EXEMPLAR_BANK,
    n_exemplars=2,
)

# sample N (Plan 03)
candidates = parallel_call_opus(
    system=sys_prompt,
    user=phase_user_msg,
    n=N,
    temperature=0.8,
    seeds=base_seeds[k],
)

# fast filter with embedding scorer
emb_scores = [score_B(c, p, phase) for c in candidates]
survivors = top_M_by_score(candidates, emb_scores, M=M)

# final selection with CD logprob delta
cd_scores = [score_A(c, p, phase) for c in survivors]
chosen = survivors[argmax(cd_scores)]
```

Recommended defaults: `N=6`, `M=3`, exemplars=2.

## Important subtlety — scorer contamination

If the same CD outputs are used both as in-context exemplars and as Score B
reference vectors, the BoN selector will prefer candidates that *parrot*
the exemplars. Mitigation:

- Split the exemplar bank into two disjoint subsets per persona: SET_A used
  as few-shot demos, SET_B used as scorer references. With ~15 generations
  per persona per phase (3 phases × 5 seeds), an 8/7 split is fine.
- Alternatively, only use Score A (logprob delta), which is reference-free
  and not vulnerable to this.

Use both when budget permits. Score B for early filtering, Score A for the
final pick from SET_A-trained few-shot inputs.

## Cost

- **Setup (one-time)**: same CD corpus as Plan 01. A few hours of dual-GPU
  CD time.
- **Per run**:
  - Opus: N × turns. With N=6 and ~30 turns total across 4 phases × 8
    personas, ~180 Opus calls (vs. ~30 for vanilla closed run, ~30 for
    the few-shot-only run from Plan 01).
  - Score B: negligible.
  - Score A: M=3 forward passes per turn on the dual-GPU rig, ~3s/turn,
    ~90s total. Negligible vs. Opus latency.

OpenRouter can be called in parallel; the per-turn wall-clock is roughly
one Opus call's latency.

## Where this plan stops winning

- If the closed model is *already* persona-faithful enough on a phase
  (Phase 1 is the easiest because the model only has to imitate a review
  style demonstrated in the system prompt), the BoN selector will pick
  near-arbitrary candidates and the cost is wasted. Worth measuring per
  phase before fixing N globally; consider N=2 on Phase 1 and N=6 on
  Phase 3.
- If the persona requires *substantive* drift from the assistant prior
  (the six-year-old being the canonical example), neither exemplars nor a
  reranker can produce content the closed model would not produce under
  any prompt — the model just doesn't know how a six-year-old talks about
  paparazzi-romance plot beats. For those personas, plain Qwen+CD remains
  the right answer, and the closed model is *not* the place to spend
  effort.

This last point is important and worth stating explicitly in the meta:
**the value of CD-on-closed-models concentrates on personas that the
closed model *can* in principle voice but tends to flatten toward
articulate-assistant by default**. The seven adult personas are squarely
in that zone. The six-year-old probably is not.

## How to evaluate

Identical metric set to the individual plans. Comparison arms:

| Arm | System prompt | Sampling | Reranker |
|---|---|---|---|
| Q  | (n/a — Qwen+CD) | CD top-p | — |
| C0 | vanilla persona prompt | T=0.7 | — |
| C1 | + exemplars (Plan 01) | T=0.7 | — |
| C3 | vanilla persona prompt | N=6 candidates | Score A+B |
| C6 | + exemplars | N=6 candidates | Score A+B |

Pass criterion: C6 strictly beats C1, C3, and C0 on distinctiveness; ties
or wins on consistency vs. Q on the personas in scope.
