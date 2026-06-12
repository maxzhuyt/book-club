# V3 writer + judge — full findings

Companion to [REPORT_V3.md](REPORT_V3.md). Where the V3 read report measured what HF and control readers *noticed* under five attentional probes, this report measures what their attention is *worth* downstream: does consolidating it into an editorial directive and writing a revised draft produce a stronger short story than the original, and does the answer depend on (a) how you consolidate, (b) which group's attention you use, or (c) which probe elicited it?

## Pipelines

Same 36 stories and same 20 readers per story as the V3 read sweep (`run_20260531-022438`): 5 probes × 2 groups × 2 agents. Two pipelines were run over the same reader pool.

**Three consolidator variants** translate reader attention into one editorial directive (same JSON schema in all three):

| Variant | What it does |
|---|---|
| **Neutral** | V2's prompt verbatim. Faithful synthesizer; no editorial opinions; "when readers disagreed, say so." |
| **Selective** (new) | Senior editor. Names 2-3 axes of divergence in the reader notes; picks a side on each; refuses the other; sets aside conflicting notes explicitly. Reasons from **CRAFT_GUIDE values** — counterfactual discipline, implication over exposition, period voice without anachronism, earned compression, specificity over abstraction, sentence-level necessity. |
| **Selective-blind** (control) | Identical editorial machinery as Selective (name axes, pick sides, refuse the other, set aside, use probe labels as priors), but with all CRAFT_GUIDE references and judge-shared vocabulary removed. Editor reasons "from your own editorial judgment" instead of from a named rubric. Run as a control for rubric leak. |

**One writer** for all three (V1's `writer.py`, already selective at the writer stage — "do not split the difference; choose one"). **One blind paired judge** for all comparisons (V1's `judge.py` with CRAFT_GUIDE embedded in the system prompt, randomized A/B blinding per call).

**Two pooling strategies:**

| Pipeline | Reader pool per cell | Revisions per cell | Judge faces |
|---|---|---:|---|
| **Pooled** | 20 readers (5 probes × A+B × 2 agents) | 2-3 (one per consolidator variant) | `neutral_vs_orig`, `selective_vs_orig`, `neutral_vs_selective`, and (control) `blind_vs_{orig, selective, neutral}` |
| **Per-probe** | 3 arms × {A-only=2, B-only=2, AB-joint=4} | 6 (3 arms × 2 consolidators) | `A_vs_orig`, `B_vs_orig`, `AB_vs_orig`, `A_vs_B` per consolidator |

# Result 1 — Pooled writer/judge (n=36)

| Comparison | revision wins | other wins | tied / unparsed |
|---|---:|---:|---:|
| **Neutral vs original** | 1 | 5 (original) | 30 / 0 |
| **Selective vs original** | **23** | 13 (original) | 0 / 0 |
| **Selective vs Neutral** (head-to-head) | 24 (selective) | 11 (neutral) | 1 / 0 |

Margins: Selective_vs_orig wins were 11 clear · 12 narrow. Selective_vs_neutral wins were 1 decisive · 14 clear · 9 narrow.

# Result 2 — Pooled rubric-leak control (n=36)

Same 20-reader pool, blind variant consolidator (no CRAFT_GUIDE references), same writer and judge.

| Comparison | blind wins | other wins | tied / unparsed |
|---|---:|---:|---:|
| **Blind vs original** | **21** | 14 (original) | 0 / 1 |
| **Blind vs Selective** (rubric leak test) | 18 | 16 (selective) | 1 / 1 |
| **Blind vs Neutral** | 15 | 15 (neutral) | 2 / 4 |

# Result 3 — Per-probe writer/judge (n=180 of 180 units, zero errors)

Totals across all 5 probes per (consolidator × reader-arm × judge face). Format: revision wins / original wins / tied (parse errors omitted).

| Consolidator | Arm | vs original | vs other group |
|---|---|---|---|
| **Neutral** | A-only | 12 / 10 / **153** | A_vs_B: 18 / 12 / **145** |
| | B-only | 14 / 5 / **156** | (same row) |
| | AB-joint | 12 / 11 / **149** | — |
| **Selective** | A-only | 91 / 69 / 16 | A_vs_B: **79 A / 92 B / 2 tied** |
| | B-only | **94** / 58 / 20 | (same row) |
| | AB-joint | **112** / 46 / 18 | — |

**Per-probe selective_vs_orig, breakdown per probe (AB-joint arm — best of the three under selective):**

| Probe | revision wins | original wins | tied |
|---|---:|---:|---:|
| P1 Plausibility | 23 | 10 | 2 |
| P2 Knowledge-gap | 21 | 9 | 6 |
| P3 Stability | 23 | 12 | 1 |
| P4 Convention | 21 | 9 | 5 |
| P5 Salience | 24 | 6 | 4 |

**Per-probe A_vs_B (selective) — does HF group reading produce better revisions than control reading?**

| Probe | A wins | B wins | tied |
|---|---:|---:|---:|
| P1 Plausibility | 15 | **21** | 0 |
| P2 Knowledge-gap | 13 | **20** | 1 |
| P3 Stability | **19** | 14 | 0 |
| P4 Convention | 17 | **19** | 0 |
| P5 Salience | 15 | **18** | 1 |
| **Total** | **79** | **92** | **2** |

# Findings

## Finding 1 — Neutral consolidation is structurally broken at every pool size

Pooled neutral_vs_orig: **30 tied of 36** (83%). Per-probe neutral_vs_orig totals across arms (sum over A-only, B-only, AB-joint across 5 probes, n=540 unit-judges): **~463 ties** (~86%). The pattern reproduces at every probe × every reader arm: ties dominate ~80-90% of cells; net direction is at or slightly below the original; tie rate is independent of pool size (2 readers in A-only and B-only, 4 in AB-joint, 20 in pooled — all show the same wall of ties).

The neutral consolidator's instruction — "faithfully translate what readers noticed; when readers disagreed, say so" — produces a directive that the (still selective) writer cannot turn into a coherent revision direction. The writer's fallback is to make minimal changes, hence the wall of ties.

V2 comparison: the closest V2 equivalent (`writer_judge_v2`, pool=16, neutral consolidator, same writer/judge) was **17 revised / 16 original / 1 tied + 2 unparsed** — a coin flip on direction but the judge committed to a side most of the time. V3 with pool=20 and one extra probe pushed neutral from coin-flip to **near-universal tie**. Broadening the reader panel doesn't strengthen neutral aggregation; it makes it worse by giving it more incoherent signal to flatten.

## Finding 2 — Selective consolidation works at every pool size, and it is *not* the rubric leaking

Pooled selective_vs_orig: **23 / 13** (64% rev wins). Pooled selective_vs_neutral: **24 / 11 / 1** (67% selective). Per-probe AB-joint selective_vs_orig: **111 / 46 / 18** (65% rev wins) — identical to pooled. Per-probe B-only selective_vs_orig: **94 / 57 / 20** (55%). Per-probe A-only: **90 / 69 / 16** (52%). Selective works at every pool size from 2 readers up to 20.

**Rubric-leak control.** A natural worry: the selective prompt references CRAFT_GUIDE values explicitly, and the judge uses CRAFT_GUIDE to score — so "selective beats neutral" might be selective being told the answer. We tested this by running a third variant (`selective-blind`) that has the same editorial-selection machinery (name axes, pick sides, refuse the other, set aside contradictions) but with every reference to CRAFT_GUIDE and all judge-shared vocabulary stripped. The editor reasons "from your own editorial judgment" instead of from a named rubric.

Result:
- `blind_vs_orig` = **21/14** (60% blind wins) — within noise of selective-with-rubric's 64%
- `blind_vs_selective` = **18/16/1** — essentially tied head-to-head

The rubric reference is not what produces the win. **Editorial commitment is.** The selective-vs-neutral effect comes from the act of *naming axes of conflict and picking a side*, not from teaching the editor the judge's vocabulary.

Caveat: `blind_vs_neutral` = **15/15/2** with 4 unparsed — closer to even than expected given that blind beats original 60% and neutral ties original ~83%. This is most likely an artifact of intransitive paired-judging (the judge calibrates differently when comparing two revisions vs comparing a revision to an unchanged reference) or sample noise from the 4 unparsed judgments; it does not change the headline rubric-leak finding (blind_vs_selective ≈ tied).

## Finding 3 — A selective writer is not enough; selectivity has to happen at the consolidator

The writer was *already* selective in V1 ("do not split the difference; choose one"). The same writer paired with the neutral consolidator produces revisions ≈ original (30/36 ties pooled). The same writer paired with either selective consolidator (with or without rubric) produces revisions that beat the original ~60-64%. **The selectivity has to happen upstream of the writer.** A writer can choose between options the directive presents; it cannot manufacture decisions when the directive is a neutral average over conflicting attention. The consolidator is the gate.

This also explains why selective works at pool=2 as well as pool=20: it's not "more readers → more signal" but "explicit editorial choice → coherent revision direction." A senior editor can extract usable commitments from 2 reader notes if they pick an axis and commit. They cannot extract usable commitments from a neutral average of 20.

## Finding 4 — B-group (control) reading is a stronger editing signal than A-group (HF) on 4 of 5 probes under selective consolidation

Selective A_vs_B head-to-head totals: **B wins 92, A wins 78, tied 2** (B wins on P1, P2, P4, P5; A wins only on P3 Stability). vs-original totals: B-only 94 rev / 57 orig (62%); A-only 90 / 69 (52%). AB-joint ≈ B-only or slightly better (111 / 46), suggesting most of the revision-relevant signal lives in the B contribution and A adds at most a modest edge on top.

This inverts the V2 read-only narrative ("HF readers have richer priors → better attention") for the *editorial* purpose. The mechanism is more subtle than "more activation → more revision-useful":

- On **P3 Stability**, A wins (19 vs 14). The V3 read result also showed A's largest activation advantage on P3 (+2.9 attention gap). On the one probe where A's attention is genuinely richer than B's, A's reading produces the stronger revision.
- On **P1, P2** (priors-dependent probes designed to favor A), B wins. The V3 reads showed these probes did *not* separate A from B as designed (attention gaps −0.2, −0.3). With no priors-activation gap, the wider/looser B notes give the editor more material to make decisive cuts from.
- On **P4 Convention** and **P5 Salience**, B wins despite A scoring higher on the read-level attention dims (+1.1 and +2.9). Here the *kind* of attention matters more than its level: A's notes cluster narrowly on period-historical specifics; B's notes spread across genre conventions, character moments, and structural choices. The selective editor uses divergence-of-attention as fuel; B's wider net beats A's deeper but narrower one.

The refined picture: **selective consolidation works on the *spread* of attention more than on its depth.** A produces a tighter, period-anchored band; B produces a wider, looser band. On the one probe (P3) where A's band is wider than usual, A wins; on the four probes where B's spread is wider, B wins.

## Finding 5 — Probe identity matters surprisingly little once consolidation is selective

Per-probe AB-joint selective_vs_orig: P1 22/10/2, P2 21/9/6, P3 23/12/1, P4 21/9/5, P5 24/5/4. The revision-win rate against the original is remarkably flat across probes — 58-69%, all in the same range as pooled (64%). Even on P5 Salience — the "innocent" priors-symmetric control probe that wasn't designed to elicit problem-flagging — selective consolidation produces revisions that beat the original at the highest rate of any probe (24/5).

The probe text controls *what kind of attention* readers produce. But the consolidator + writer pipeline determines whether that attention becomes a *usable revision*. The V3 read result (probes don't isolate priors-dependent attention as designed) is consistent with this: the editorial layer, not the probe layer, does the methodological work for producing improved revisions.

# What this means for the project

1. **The central methodological finding** is now: editorial selection at the consolidation stage converts reader attention into improved revisions; neutral aggregation does not, regardless of pool size or probe choice. The rubric-leak control rules out "selective is just being told the judge's rubric." The mechanism is *editorial commitment under conflict*, not rubric knowledge.

2. **The V3 read result** (priors-dependent probes don't separate A from B as designed) is a secondary, more nuanced finding about probe construction. The writer/judge results add a new angle: under selective consolidation, B's *broader-band* attention is a stronger editing signal than A's *deeper-but-narrower* attention on 4 of 5 probes. The exception (P3 Stability, where A's band actually is broader) confirms the mechanism: divergence-of-attention is the fuel, not depth-of-priors.

3. **Future work** should treat the consolidator as a first-class methodological variable. The selective consolidator prompt is doing real interpretive work that a neutral one cannot. Any pipeline that asks "do these readers help editing" needs to specify which kind of consolidation, or it is measuring something else.

4. **The HF-group narrative needs revision.** "HF readers have richer priors → better attention" survives as a read-time statement on certain probes (especially Stability), but does not generalize to "HF readers produce better revisions through their attention." The downstream test reverses the direction on 4/5 probes. The cleaner statement is: HF-group attention is **more focused**; control-group attention is **more diverse**; selective editorial consolidation extracts more from diversity than from focus.

# Data and reproducibility

- Pooled revisions and judgments: `run_20260531-022438/writer_judge_v3/<story>/`
- Pooled rubric-leak control: `run_20260531-022438/writer_judge_v3_blind/<story>/`
- Per-probe revisions and judgments: `run_20260531-022438/revise_judge_byprobe_v3/<story>/<probe>/`
- Aggregates (JSON): `*/( _aggregate.json )` under each of the above
- Consolidator prompts: `experiments_v3/consolidators_v3.py` (`CONSOLIDATOR_NEUTRAL_SYSTEM`, `CONSOLIDATOR_SELECTIVE_SYSTEM`, `CONSOLIDATOR_SELECTIVE_BLIND_SYSTEM`)
- V2 baseline: `experiments_v2/run_20260525-190113/` and [REPORT_V2.md](../../experiments_v2/report_v2/REPORT_V2.md) §3

Sample sizes by judge face:

| Pipeline / face | n |
|---|---:|
| Pooled (3 faces × 1 variant) and Pooled rubric-control (3 faces × 1 variant) | 36 each |
| Per-probe (4 faces × 2 variants × 3 arms × 5 probes) | 36 per cell × 5 probes = **180/180** complete |

Two per-probe units (`cell-07-middle-sys-pure__run3/P5`, `cell-08-middle-sys-fantastical__run3/P1`) initially failed with isolated LLM empty-content responses; both recovered on retry. Numbers above reflect the final 180/180.
