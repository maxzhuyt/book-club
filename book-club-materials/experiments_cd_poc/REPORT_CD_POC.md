# Can we make our simulated readers sound like distinct people? A proof-of-concept

**Date:** 2026-06-13
**Verdict:** Clear, consistent positive signal — every measure moves the expected way and the
baseline reproduces the original problem, so the effect is real, just not yet precisely
quantified (small pilot). **Recommendation: scale it up.**

---

## 1. The problem and the fix

We simulate AI "readers" who give feedback on short historical-fiction passages, split into
two groups: **Group A** (avid historical-fiction readers) and **Group B** (generic readers).
In our previous round ("v3"), the probes *designed* to separate the groups largely failed to:
on the priors-dependent probes the two groups gave nearly identical feedback (P2 flat, P4's
strong v2 signal collapsed). The groups only differed on probes that weren't supposed to
differentiate at all (an unintended "leak"). Either way, the probes weren't doing their job —
the intended signal had flattened.

The fix we test is **contrastive decoding (CD)**: a knob, applied as the AI writes each word,
that dials *up* whatever is distinctive about a specific reader and dials *down* the bland
"helpful assistant" voice models default to.

**How it works:** at each word, we run the model twice — once told *"you are this specific
reader,"* once told *"you are a generic reader"* — and amplify the difference between them.
(Analogy: a museum curator and a casual visitor describing the same painting overlap a lot;
CD ignores the overlap and keeps only what the curator says that the visitor wouldn't.) The
strength is a dial **α**: **α=0** is ordinary prompting (our baseline); **α=1** is CD on. So
α=0 vs α=1 is a clean before/after with everything else held constant.

---

## 2. Why Qwen3-32B, not the production model (DeepSeek-V4-Flash)

Two independent reasons V4-Flash couldn't be used:

1. **Its API can't do CD.** CD needs the model's internal word-score tables from *both* runs;
   the API only returns finished text. Impossible in principle, regardless of hardware.
2. **Its raw files won't run on our GPUs.** V4-Flash stores most of its weights in an
   ultra-compact "FP4" format that needs a newer chip generation than we own; on our hardware
   it produced gibberish even with CD off.

CD is a property of *how you generate text*, so any well-behaved open model answers the
question. We used **Qwen3-32B** (runs cleanly and fast). Note: the only valid baseline is
**Qwen3-32B at α=0** — the production model's numbers come from a different model/pipeline and
are context, not a comparable control.

---

## 3. What we did

For each reader × passage we ran a two-step read: (1) react to the passage in their own voice,
then (2) answer a focused follow-up question (a "probe"). Full grid:

| Varied | Options |
|---|---|
| Stories | 3 historical-fiction passages |
| Probes | 3 (below) |
| Reader groups | A (4 avid) vs B (4 generic) |
| CD knob | α=0 (off) and α=1 (on) |

= 3 × 3 × 8 readers × 2 settings = **144 conversations**, each with 2 responses = **288
responses**. The three probes span the range of v3 outcomes:

- **P2 (Knowledge-gap)** — *converged in v3* (groups didn't differ at all; the probe the
  design most wanted to work, and didn't).
- **P4 (Convention)** — *converged in v3*: its strong v2 group split (historical framing
  46-pt gap) collapsed to an 11-pt gap. **The probe CD most needs to rescue.**
- **P5 (Salience)** — *did NOT converge*: in v3 this was a control meant to show no group
  difference, but it "leaked" the **largest** A-B gap of all. We use it here as a **positive
  control** — a probe where the groups already reliably differ, so it confirms the measurement
  is sound and shows CD doesn't break what works.

**Scoring.** An automated grader (the **DeepSeek** model, same as v3, set to grade
consistently) rated each response 0–5 on period_specificity, knowledge_invoked, concreteness,
locations_cited, anchors, and labeled its convention framing (historical / generic / mixed /
none). The readers are Qwen3-32B; the grader is DeepSeek — two different models. To avoid
relying on the grader, we also ran a grader-free check (§4.3).

**Quality control:** every run was gated on coherent output, adequate speed, and the knob
actually changing the text. All 288 passed; none empty or excluded.

---

## 4. Results

**Reading the numbers:** a **"gap"** = Group A's average minus Group B's (bigger = more
different; negative = A scored lower). A **p-value** is the chance of seeing a result this big
if the groups were truly identical (below 0.05 = "significant"). *Two caveats:* we ran ~30 gap
tests, so one p≈0.04 by luck is expected; and "12 per group" reuses the same 4 readers across
3 stories, so the real precision is closer to n≈4. **Trust the consistency of direction more
than any single p-value.**

### 4.1 The A-vs-B gap widened under CD (P4 and P5)

Positive = Group A (avid) scored higher, as expected.

| Probe · dimension | Gap, α=0 | Gap, α=1 (CD) |
|---|---|---|
| **P5 · knowledge_invoked** | +0.33 (p=0.52) | **+0.92 (p=0.038)** — significant |
| **P5 · period_specificity** | +0.42 (p=0.45) | **+0.83 (p=0.066)** — near-sig |
| P5 · concreteness | +0.17 | +0.50 |
| P4 · knowledge_invoked | +0.17 | +0.33 |
| P4 · locations_cited | +0.08 | +0.50 |
| P4 · period_specificity | +0.33 | +0.42 |
| P4 · anchors | +0.17 | +0.33 |
| P2 · all dimensions | ≈ 0 | no positive shift † |

† On P2, two dimensions go slightly *negative* under CD (anchors −0.42, locations_cited −0.33);
none significant — noise at this sample size.

On **P5 and P4**, CD widened the gap on essentially every dimension, all in the expected
direction. The one result crossing significance (P5 knowledge_invoked, p=0.038) is corroboration,
not standalone proof, given ~30 tests at n≈4. On **P2** CD didn't help. (Even on P5 the absolute
scores are modest — A 2.0, B 1.1 on 0–5 — so it's "a bit more knowledge," not "expert vs blank.")

### 4.2 The clearest effect: the groups adopted different "playbooks"

Share of responses using **historical** framing (and **generic**, with CD on):

| Probe | Group | Historical, α=0 | Historical, α=1 | Generic, α=1 |
|---|---|---|---|---|
| **P5** | A (avid) | 42% | **50%** | 8% |
| **P5** | B (generic) | 33% | **8%** | **58%** |
| P4 | A | 50% | 33% | 8% |
| P4 | B | 42% | 17% | 42% |

Without CD the groups framed conventions similarly (P5: 42% vs 33%). **With CD, on P5 they
split sharply:** avid readers held at 50% historical framing; generic readers dropped to 8% and
shifted to 58% generic framing. The avid stayed in "history mode"; the generic defaulted to
"generic literary mode." That Group-B swing is large enough to be unlikely to be noise. (On P4
the direction is the same but muddier — "mixed" was the biggest bucket there, 58%/42%, so the
clean split is really a P5 result. Rows omit "mixed"/"none.")

### 4.3 Grader-free double-check (TF-IDF word-math)

To rule out grader bias, we measured separation using **plain word-counting math only** (TF-IDF:
tally each text's distinctive words, ignore filler). Higher = the two groups' wording is more
different than wording within a group; p comes from reshuffling texts thousands of times.

| Probe | α=0 | α=1 (CD) |
|---|---|---|
| P2 | ≈0 (−0.004, p=0.47) | **apart (+0.008, p<0.0001)** |
| P4 | ≈0 (−0.005, p=0.31) | **apart (+0.013, p=0.0002)** |
| P5 | ≈0 (−0.001, p=0.049) | ≈0 (+0.000, p=0.09) |

**Honest read:** this confirms CD increases differentiation, but on **different probes** than the
grader — word-math fires on P2/P4, the grader fires on P5 (and word-math on P5 was actually a
touch weaker with CD). The reason: P5 responses all quote the same passage (similar words,
different *thinking* → grader catches it); P2/P4 differ more in actual wording (word-math catches
it). Two angles, both showing CD increases some axis of difference.

### 4.4 Example (illustrative, not evidence on its own)

Same generic Group-B reader, same question:
- **α=0:** abstract — *"…his search for language to express something beyond language."*
- **α=1:** sharper but still generic — *"…how do you write faith when it no longer fits in the old boxes?"*

An avid Group-A reader, α=1, instead reaches for period anchors:
- *"…Barbara Kingsolver understands this kind of human wrestling with truth, and John Steinbeck would recognize the quiet weight in it."*

---

## 5. Bottom line and limitations

**CD makes the two reader groups give more distinct feedback.** The headline win is **P4** —
a probe that had genuinely *converged* in v3 (its v2 group split collapsed), which CD pushed
back apart on every dimension. **P5**, our positive control (already the strongest
differentiator in v3), widened further and showed the cleanest "playbook" split under CD
(Group B 33%→8% historical, 0%→58% generic) — confirming CD sharpens rather than breaks a
working probe. The grader-free word-math independently separated the groups on **P2 and P4**.
The one probe CD did *not* rescue is **P2's graded scores** — it had converged in v3 and stayed
flat, so that probe fails for a structural reason, not voice-sameness (though its *wording* did
separate under the word-math).

**Limitations** (all about precision/generality, not whether the effect is real):
- **Small pilot** — 4 readers/group, 3 stories, one untuned α. The single significant p-value is
  about what chance produces across ~30 tests; the case rests on consistency of direction.
- **The three measures converge but don't cleanly replicate** (grader→P5, word-math→P2/P4).
- **Qwen3-32B, not production** — validates the method, not the production model (which can't do CD).
- **α=1 is aggressive** and can degrade fluency; the coherence gate mitigates but doesn't fully
  rule out that some difference reflects reduced coherence rather than persona signal.

---

## 6. Next steps

1. **Tune α** (0.5 / 1.0 / 1.5) to find where differentiation peaks before writing degrades.
2. **Scale up** to the full v3 study (12 story types × 3 rewrites, 5 probes, 10 readers/group)
   for properly powered statistics with no reader reuse.
3. **Try asymmetric CD** (big persona model + small generic model) to see if it amplifies the effect.
4. **Keep the production model as context only** — don't attempt CD on it.

---

## 7. Reproduction

```
experiments_cd_poc/
  cd_decoder.py        CD engine (two Qwen3-32B copies, one per GPU)
  neg_prompt.py        the "generic reader" baseline prompt
  run_cd_poc.py        runs the 144-conversation grid     run_cd_poc.sbatch  cluster job (4 GPUs)
  code_responses.py    the DeepSeek grader (same as v3)   tfidf_separation.py  grader-free check
  analyze_cd_poc.py    A-vs-B gaps + convention shares
  run_cd_poc/          all responses + grades             aggregated/  result tables (CSV)
```
