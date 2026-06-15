# Book-club simulation under system-prompt contrastive decoding

Implementation of the method designed in
[`../CONTRASTIVE_DECODING_DESIGN.md`](../CONTRASTIVE_DECODING_DESIGN.md), applied
to the 8-persona Goodreads-grounded panel from `../personas/`, with a 9th
six-year-old persona added as a stress test of CD's ability to simulate a
strongly-non-default knowledge level.

## Method, briefly

System-prompt contrastive decoding (Dong, Hu, Hui & Collier 2026, Eq. 5):

```
z_pos = model_logits(s_pos, u, x_<t)        # persona system prompt
z_neg = model_logits(s_neg, u, x_<t)        # "generic articulate reviewer" suppressor
z_cd  = z_pos + alpha * (z_pos - z_neg)
x_t   ~ softmax(z_cd / T)   (top-p truncation)
```

`s_pos` is the existing 3-layer scaffold in
`../personas/system_prompt/user_*_system_prompt.txt` (identity card + 2 verbatim
voice exemplars + discussion rules). `s_neg` is a shared "thoughtful articulate
literate book-club reviewer" prompt that verbalizes the post-training assistant
attractor — the thing we want to amplify *away* from. In Phase 3, `s_neg` is
augmented with an anti-conformity clause ("you tend to agree with the group"),
so CD also suppresses agreement-drift.

`alpha` is set per persona from identity-card signals (see `src/cd/alpha.py`):
default 1.0, +0.2-0.4 for low-volume readers, +0.4 for terse-review style,
+0.2 for strong genre-aversion to the target, +0.15 for extreme tone; capped
at 2.0. The six-year-old runs at α=2.0. Phase 4 uses 0.5× the Phase-1 α.

## Base model and hardware

`Qwen/Qwen2.5-14B-Instruct` (bf16) — directly validated for CD in the paper.
Two model copies, one per GPU (RTX A6000 × 2), so the positive and negative
branches run in parallel via async cross-device kernel launches.

## Code layout

```
simulation/
├── README.md
├── src/
│   ├── smoke_test.py                  # paper Fig.1 reproduction
│   ├── cd/
│   │   ├── decoder.py                 # two-branch CD sampling loop
│   │   ├── prompts.py                 # s_pos / s_neg builders, phase user msgs
│   │   └── alpha.py                   # per-persona alpha schedule
│   └── bookclub/
│       ├── cast.py                    # persona dataclass + 6yo
│       ├── simulate.py                # 4-phase runner
│       └── summarize.py               # group-summary generator (alpha=0)
└── outputs/
    ├── manifest.json                  # cast + alphas + seed + model id
    ├── transcripts/
    │   ├── phase1_private.json        # private reviews (independent)
    │   ├── phase2_reactions.json      # broadcast reactions
    │   ├── phase3_discussion.json     # moderated rounds (list of rounds)
    │   └── phase4_reflections.json    # final reflections
    ├── discussion_transcript.md       # human-readable full transcript
    ├── summary_input.md               # input to the summarizer
    ├── group_summary.md               # concise group summary
    └── smoke/melting_point.md         # paper-Fig.1 smoke test result
```

## Reproducing

```
# smoke test (paper's six-year-old / melting-point of iron)
python src/smoke_test.py

# full 4-phase simulation
python src/bookclub/simulate.py \
    --model-id Qwen/Qwen2.5-14B-Instruct \
    --story story_1.md --story-title "Love in the Limelight" \
    --seed 7 --rounds 2

# group summary (alpha=0, neutral note-taker)
python src/bookclub/summarize.py
```

## Phase invariants

1. **Phase 1 is fully independent.** Every persona's private review is
   generated and committed before any persona is shown any peer review.
2. **Phase 2 simultaneously broadcasts all Phase-1 reviews.** Each persona
   sees every peer review (minus their own) at once — no sequential
   reveal that would prime first-mover anchoring.
3. **Phase 3 is round-robin** with a rotating speaker order (so the same
   reader isn't always first), 2 rounds. The moderator's per-round addendum
   pushes for disagreement (R1) and an anti-conformity self-check (R2).
4. **Phase 4 reflection** uses lower α to keep the final stance honest rather
   than amped persona-theater.
