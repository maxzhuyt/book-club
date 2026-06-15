# Plan 03 — Best-of-N with a CD-derived scorer

## Core idea

Sample N candidate completions from the closed frontier model (varied via
temperature, seed, or minor prompt jitter), then rerank them using a signal
derived from the existing open+CD model. Pick the top-1 (or weighted
ensemble) as the canonical output.

CD's contribution is no longer generation — it is *judgment*. We use the
fact that, for a given persona, the open+CD distribution scores
persona-faithful text higher than persona-drifted text. The closed model
brings reasoning and substance; CD brings the persona discriminator.

## What stays untouched

- `simulation/src/cd/decoder.py`, `prompts.py`, `alpha.py`.

We add a separate scoring helper that loads the same models the CD decoder
uses and does a single forward pass per candidate. No CD-loop change.

## What gets added

```
closed_models/
├── scoring/
│   ├── cd_scorer.py            # build_score(text, persona, phase) -> float
│   └── reference_bank/         # optional: stored CD reference outputs
└── runners/
    └── simulate_closed_bon.py
```

## Score function — two implementations to try

### Score A — Logprob delta under z_pos / z_neg

For a candidate `x = [x_1, ..., x_T]` from the closed model:

```
score_A(x | s_pos, s_neg, user_msg) =
    sum_t [ logp_pos(x_t | x_<t) - logp_neg(x_t | x_<t) ]
```

This is the integral of the per-token CD signal over the candidate text.
A persona-faithful candidate sits in the region where the positive prompt
strongly prefers tokens over the generic-reviewer baseline; a drifted
candidate sits where they roughly agree (delta ≈ 0).

Implementation: load `model_pos` and `model_neg` (same dual-GPU setup as
`load_dual_models`). For each candidate, prepend `s_pos+user_msg` to one
model and `s_neg+user_msg` to the other, then teacher-force the candidate
tokens and sum `log p_pos - log p_neg`. This is exactly the forward pass
inside `decoder.generate()`'s loop, but with the candidate's tokens
substituted in place of sampling. No CD code change — we just call the
underlying models directly with the same chat-template encoding (copy the
`_encode_chat` helper, or import it).

Normalization: divide by candidate length to remove length bias. Optionally
exponentiate to `alpha` to match the regime CD generates from.

### Score B — Embedding distance to a CD reference bank

Pre-generate K reference completions per persona per phase using
`ContrastiveDecoder.generate()` (same bank as Plan 01). At rerank time,
embed each closed-model candidate and each reference with a small embedder
(SBERT MiniLM is plenty for stylistic distance) and score:

```
score_B(x) = mean cosine similarity to top-3 nearest references
```

Cheaper than Score A (one tiny forward pass per candidate, vs. one
large-model pass per candidate), at the cost of not directly using CD's
discriminator — it uses CD's *generations* as a proxy.

Use Score B as a fast first filter; use Score A only on the top-M survivors
to control compute.

## Integration

`simulate_closed_bon.py`:

```python
N = 8   # candidates per turn
M = 3   # top-M from Score B that get Score A

for persona, phase, user_msg in turns:
    candidates = []
    for k in range(N):
        out = call_opus(system_prompt(persona), user_msg, temperature=0.8,
                        seed=base_seed + k)
        candidates.append(out["text"])

    if N > 4:
        embedding_scores = [score_B(c, persona, phase) for c in candidates]
        survivors = top_M_by_score(candidates, embedding_scores, M=M)
    else:
        survivors = candidates

    cd_scores = [score_A(c, persona, phase) for c in survivors]
    chosen = survivors[argmax(cd_scores)]
```

OpenRouter supports parallel requests; the N samples can be fired
concurrently from a thread pool — wall-clock for the sample step is
approximately one call's latency.

Outputs go to `simulation/outputs_closed_bon/`, layout matching the existing
compare-style outputs so `summarize_opus.py` works unchanged.

## What N should be

- N=4 is the cheap default — single batched OpenRouter call with `n=4` if
  the provider exposes it, or four concurrent calls otherwise.
- N=8 with M=3 gives a noticeable lift on the discriminator at 2× cost.
- N=16 is the point where Score A's compute dominates Opus cost.

## Cost and risk

- **Cost — sampling**: linear in N for the closed model. ~4–8× the cost of
  a single-shot run.
- **Cost — scoring**: Score B is negligible. Score A is one forward pass per
  candidate over the candidate's tokens (typically ~200–500 tokens),
  on bf16 14B model on 2 GPUs — a few hundred ms per candidate. M=3 means
  ~1s of GPU time per turn. Cheap.
- **Risk — reward hacking**: a reranker has a finite vocabulary of mistakes.
  If Score A consistently rewards short outputs (because z_pos - z_neg is
  positive on the average token), the BoN winner trends short. Mitigation:
  length-normalize and also length-band the candidates (request the closed
  model for output length matching `persona.avg_review_words // 4`).
- **Risk — CD scorer mirrors CD's blind spots**: any failure mode of the
  open+CD model (e.g., the six-year-old voice occasionally collapsing to
  bullet lists) will be rewarded by the scorer too. Mitigation: keep a
  secondary length / safety filter that rejects degenerate candidates
  before they reach the scorer.

## How to evaluate

Same metrics as Plans 01/02. Additional ablation: compare Score A vs.
Score B as the sole reranker (does the cheaper scorer suffice?), and
compare BoN with N=1 (= no reranking) to N=4 and N=8 to size the lift.

Pass criterion: BoN beats vanilla closed-model on persona distinctiveness
and held-out style consistency, with the score curve still rising at N=8
(if it has saturated by N=2, the scorer isn't doing anything useful).
