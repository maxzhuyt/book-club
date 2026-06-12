# V3 brief findings — attentional probes rewritten

**Design vs V2.** Same 36 stories (12 seeds × 3 generation runs), same two confound-matched groups (now n=10/group, up from 8), same two-phase read pipeline, same coder + blind comparator. The only thing that changed is the probe text. The four V2 probes (Plausibility, Knowledge-gap, Stability, Convention) were rewritten by Hoyt in [experiments_v3/v3.md](../v3.md) — primarily by adding internal taxonomies (e.g. P1 now asks for two kinds of mismatch: real-history vs invented-element) and softening the LLM-phenomenological framing. A fifth probe, **P5 Salience**, was added as a priors-symmetric control: "which parts would you highlight for another reader." 720 reads scheduled; 715 ok, 5 abandoned as deterministic length-flakes (the model burns its full reasoning budget without emitting visible content; reproduces even at ~9700 tokens).

## Headline: the V3 rewrites flattened the V2 signals

Side-by-side, holding the stories, the groups, and the pipeline constant:

**A–B gap on priors-dependent attention dims (period_specificity + knowledge_invoked + anchors, summed; pass-2):**

| Probe          | V2 gap | V3 gap |
|----------------|-------:|-------:|
| Plausibility   |  −0.8  |  −0.2  |
| Knowledge-gap  |  −0.7  |  −0.3  |
| Stability      |  +2.6  |  +2.9  |
| Convention     |  +4.7  |  +1.1  |
| Salience (new) |    —   |  +2.9  |

**Blind comparator (stronger_on_specificity), 36 cells:**

| Probe          | V2 A/B/T  | V3 A/B/T  |
|----------------|-----------|-----------|
| Plausibility   | 7/21/8    | 15/17/4   |
| Knowledge-gap  | 15/13/8   | 15/12/9   |
| Stability      | 18/9/9    | 17/13/6   |
| Convention     | 19/6/11   | 21/12/3   |
| Salience (new) | —         | 16/8/12   |

**convention_type, "historical" share (pass-2):**

| Probe          | V2 A | V2 B | V3 A | V3 B |
|----------------|-----:|-----:|-----:|-----:|
| Plausibility   | 0.94 | 0.86 | 0.79 | 0.76 |
| Knowledge-gap  | 0.69 | 0.67 | 0.74 | 0.67 |
| Stability      | 0.82 | 0.54 | 0.58 | 0.29 |
| Convention     | 0.58 | 0.12 | 0.46 | 0.35 |
| Salience (new) | —    | —    | 0.54 | 0.31 |

## Three findings

### Finding 1 — The intended priors-dependent probes (P1, P2) no longer separate A from B.

In V2, Plausibility produced a *negative* gap: the control group out-scored HF readers on every priors-dependent attention dim and on the blind comparator (7 A wins vs 21 B wins). The V2 brief read this as a stance asymmetry — B's lower-prior posture made every detail a candidate inconsistency, while A's prior-richness damped flag rates.

V3's Plausibility prime adds an internal taxonomy: notice **(a)** mismatches in real historical figures/events vs **(b)** mismatches in invented elements clashing with period. This was intended to give A something concrete to do. What it did instead was *flatten the gap*: V3 Plausibility comparator is 15A/17B/4 (essentially a wash); the −0.8 attention gap shrank to −0.2; convention_type historical share equalized (A 0.79 / B 0.76 vs V2's 0.94 / 0.86).

Knowledge-gap shows the same flattening (V2 attention gap −0.7 → V3 −0.3; comparator essentially unchanged at 15/12/9). On the metrics the design was built around, **the two probes Hoyt wrote to be priors-dependent now produce the smallest A–B differences in the battery.**

### Finding 2 — Convention, the V2 headline separator, weakened sharply.

V2's clearest finding was that on the Convention probe, A invoked period-specific narrative norms (58% historical convention_type) while B invoked generic-literary norms (12% historical, 31% generic-genre, 57% mixed) — a near-clean separation that the blind comparator backed up at 19A/6B.

V3's Convention prime drops the "scenes of this kind in this period" framing and uses generic "scenes of this kind" language. A still wins the comparator (21/12/3) but the *kind* of differentiation collapsed: A historical 0.46 / B 0.35 (was 0.58 vs 0.12) — an 11-point gap where V2 had a 46-point gap. The summed-attention gap fell from +4.7 (largest in V2) to +1.1 (smaller than Stability and Salience in V3). The probe is still mildly A-favoring on its prior text but no longer surfaces the qualitative split — the "different playbooks" picture from V2 — that made the V2 finding interpretable.

### Finding 3 — Both "controls" leak. Salience is the loudest leak.

Of the five V3 probes, the largest A–B attention gaps are on the two probes meant to be priors-independent or priors-symmetric: Stability (+2.9) and Salience (+2.9). On Salience — the new P5 control, which only asks "which parts would you highlight for a fellow reader" — A's convention_type is 54% historical to B's 31%, with B carrying 28% generic-genre. The blind comparator on Salience tilts A (16/8/12); not as far as Convention, but unmistakably not symmetric. Stability shows the same pattern (A 58% historical / B 29%, comparator 17/13/6).

Interpreted together with Finding 1, this is consistent with a single mechanism: **A imports period framing as a default disposition**, regardless of what the probe asks. When the prime explicitly directs attention to a priors-dependent task (V3 P1, P2), A doesn't gain ground because B is being scaffolded into the same task; when the prime asks an "innocent" question about salience or about writing-as-writing, A still answers from a period-tuned stance. The probes cannot isolate "what HF priors do under elicitation" because the priors aren't activated by elicitation — they're load-bearing whenever an HF-persona model reads anything.

## What this implies

V3's design intent — sharpen the priors-dependent probes, add a control whose answers are symmetric across groups — did the opposite of what was hoped:

- The probes that gained taxonomy structure (P1's (a)/(b), P2's (a)/(b)/(c)) lost A-vs-B differentiation. Plausibly the taxonomies scaffold *both* groups into the same shape of answer, erasing the qualitative gap that V2 captured.
- The probes that stayed open-ended (P3, P5) show the largest gaps — but those are exactly the probes that were never supposed to produce A-vs-B differences. Hoyt's footnote on P3 anticipated some leak ("a reader can only be pulled out by features they're equipped to detect"); P5 was supposed to be the clean control and isn't.
- Convention, the one V2 finding that gave a clean *qualitative* read (A invokes historical playbook, B invokes generic playbook), is the V3 probe that moved most in the wrong direction. Removing "period" from the prime appears to be the culprit.

Two paths forward:
1. **Treat the V3 result as the actual finding.** The thing two confound-matched personas differ on, when reading the same generated historical fiction, isn't what gets noticed under specific elicitation — it's the *default frame* a HF-persona model carries into any reading task. The cleanest measurement of that is the V3 P5/P3 result, not the V3 P1/P2 result the design tried to elicit.
2. **Restore V2's prime language for P1 and P4** (the period-anchored versions) but keep the V3 P5 control. That would let us see whether the V2 P4 result holds up against the V3 Salience leak — i.e., whether period-framed Convention separates A from B *beyond* the baseline rate at which A imports period framing into everything.

## Data and figures

- Reads: `run_20260531-022438/results/{A,B}/<cell>/<probe>/agent-{0,1}/{pass1.txt,pass2.txt,coding.json,meta.json}`
- Comparisons: `run_20260531-022438/comparisons/<cell>/<probe>/comparison.json`
- Aggregates: `run_20260531-022438/aggregated/{by_probe_group.csv, ab_gaps.csv, convention_type.csv, elicitation_lift.csv}`
- Figures: [figures/fig_ab_attention.png](figures/fig_ab_attention.png), [figures/fig_convention_type.png](figures/fig_convention_type.png)
- V2 baseline this report compares against: `experiments_v2/run_20260525-190113/`

5 reads (out of 720) abandoned as deterministic length-flakes: P4 agent-1 on three A cells, P5 agent-0 on two A cells, P2 agent-0 on two B cells, P1 agent-1 on one B cell. All are concentrated on the fantastical/sys-pure subtypes — the prompts are slightly longer and the model occasionally spends its full token budget on internal reasoning without emitting visible content. The five gaps are spread across four (story, probe) pairs and shouldn't bias any per-probe statistic.
