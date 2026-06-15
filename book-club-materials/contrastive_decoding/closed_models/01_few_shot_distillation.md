# Plan 01 — Few-shot distillation: CD-generated exemplars as in-context demos

## Core idea

Run the existing open+CD pipeline to produce a small corpus of persona-faithful
book-club outputs per persona, then feed 1–3 of those outputs as **few-shot
demonstrations** in the system prompt sent to the closed frontier model.

The intuition: the existing Layer-2 voice exemplars in
`personas/system_prompt/user_*_system_prompt.txt` are real Goodreads reviews
— they teach the model the persona's *review* voice. They do not teach it the
persona's *book-club discussion* voice (Phase 2/3 reactions, disagreements,
moderator responses). CD on the open model produces exactly that, and few-shot
imitation is the closed model's strongest channel for absorbing demonstrated
style.

## What stays untouched

- `simulation/src/cd/decoder.py`, `prompts.py`, `alpha.py` — used as-is.
- The per-persona system prompts in `personas/system_prompt/`.

## What gets added (all in `closed_models/`)

```
closed_models/
├── exemplars/
│   ├── build_exemplars.py        # one-off: runs CD to fill exemplar bank
│   └── bank/
│       ├── manifest.json
│       ├── <short_id>/
│       │   ├── phase1_review_<seed>.txt
│       │   ├── phase2_reaction_<seed>.txt
│       │   ├── phase3_round1_<seed>.txt
│       │   └── ...
└── runners/
    └── simulate_closed_fewshot.py   # the closed-model simulator
```

## Step 1 — Build the exemplar bank (offline, run once)

`build_exemplars.py`:

1. Load cast via `bookclub.cast.load_cast` and the open model via
   `cd.decoder.load_dual_models` exactly as `simulate_compare.py` does.
2. For each persona, for each phase (1, 2, 3-R1, 3-R2, 4), generate N=3
   completions with different seeds. Reuse the existing user-message builders
   in `cd/prompts.py` and `bookclub/simulate_compare.py`.
3. **Use a held-out story** (not story_1 or story_2) — we want the exemplars
   to teach voice, not contaminate the eval target. Add `story_3.md` to
   `personas/` or pick a synthetic micro-story.
4. Save each completion as a separate `.txt` plus a `manifest.json` recording
   alpha, seed, phase, persona, story_id.

This is the **only** new code that touches the open model. Everything below
runs against text on disk.

## Step 2 — Inject exemplars into the closed-model prompt

`simulate_closed_fewshot.py`: a mirror of `simulate_compare.py` but the
generation call goes to OpenRouter (reuse the OpenRouter helper from
`bookclub/summarize_opus.py`: `load_key`, `call_opus`, key name `"Hoyt"`).

System prompt construction per persona per phase:

```
<existing 3-layer persona system prompt from personas/system_prompt/...>

--- BOOK-CLUB VOICE EXEMPLARS ---
Below are two examples of how YOU specifically write in a book-club
discussion. Match the diction, sentence rhythm, hedging style, opinion
strength, and length. Do not imitate the example's topic; imitate its voice.

EXAMPLE 1 (your Phase-2 reaction style):
<<<
{exemplar_phase2_seed_a}
>>>

EXAMPLE 2 (your Phase-3 disagreement style):
<<<
{exemplar_phase3_seed_b}
>>>
--- END EXEMPLARS ---
```

Exemplar selection rules:
- Pick exemplars from the **same phase** as the current generation when
  available (Phase 1 demo for Phase 1, etc.). Phase 4 reflections fall back
  to Phase 3 if no Phase 4 exemplar is short enough.
- Always 2 exemplars. More crowds the context; one underdetermines voice.
- Skip the demo if it exceeds 800 tokens — pick a shorter sibling instead.

## Step 3 — Run the full 4-phase protocol

Same orchestration as `simulate_compare.py`, but:
- Replace `decoder.generate(...)` with a `call_opus(system, user, ...)` call.
- Drop the `alpha`, `s_neg`, `model_pos/model_neg` plumbing entirely on the
  closed-model path.
- Outputs go to `simulation/outputs_closed_fewshot/` mirroring the existing
  `outputs_qwen2.5_14b_compare/` directory layout, so `summarize_opus.py`
  works on it unchanged.

## Cost and risk

- **Cost**: one-time open-model bank (~9 personas × 5 phases × 3 seeds ≈ 135
  CD generations; under an hour on the existing 2-GPU rig). Per-run closed
  cost is unchanged from the existing `summarize_opus.py` path; few-shot adds
  ~1.5–3K input tokens per call.
- **Risk — voice mimicry without persona substance**: closed model copies
  surface tics but ignores the identity card. Mitigation: keep the original
  3-layer scaffold *above* the exemplars; treat exemplars as voice priming,
  not as the persona definition.
- **Risk — leakage of CD-specific artifacts**: low-quality CD outputs (the
  occasional repetition or syntactic glitch from α near the cap) could teach
  the closed model bad habits. Mitigation: human-review the bank once before
  use; reject obvious failures (the bank is only ~135 short texts).

## How to evaluate

A/B compare on the same story:
- **Arm Q**: Qwen2.5-14B + CD (existing `outputs_qwen2.5_14b_compare/`).
- **Arm C-vanilla**: closed model with just the existing system prompt.
- **Arm C-fewshot**: closed model with system prompt + CD exemplars (this
  plan).

Metrics already supported by the repo: avg words, avg sent length,
emotionality/analyticity (from `bookclub/cast.py` `summary()`-style
recomputation on outputs), pairwise stylistic divergence across personas
(distinctiveness). The held-out style metric from `DESIGN.md §4.1` carries
over directly.

Pass criterion: Arm C-fewshot beats Arm C-vanilla on
*distinctiveness* (pairwise divergence increases) and matches or beats
Arm Q on *consistency* (held-out style match to the real reviewer).
