# Book Club V3 — Reader Pipeline (brief)

**Scope.** The reader half of V3 only: do the attentional **probes** and the
**A-vs-B reader contrast** produce differentiated attention? Figures 1–4 and
their tables.

## Setup

36 LLM-generated counterfactual historical-fiction stories. Per story, 20 LLM
"reader" personas (deepseek-v4-flash) read and respond:

- **Group A** — 10 personas matched to avid historical-fiction readers
  (Goodreads-derived; mean HF reading fraction ≈ 0.42).
- **Group B** — 10 controls matched to A on review length, analyticity, and book
  count, but with low HF reading (≈ 0.04). The contrast.

Each reads under one of five **probes** — P1 Plausibility, P2 Knowledge-gap,
P3 Stability, P4 Convention, P5 Salience — and answers twice: pass-1 generic,
**pass-2 probed** (all analyses use pass-2). A separate LLM coder then scores
each pass-2 response on five attention dimensions — *period specificity,
concreteness, outside knowledge* (0–5 ratings) and *locations cited, historical
anchors* (counts) — plus a categorical **convention frame** (historical /
generic-genre / mixed / none). **n = 720** coded pass-2 responses
(36 stories × 5 probes × 2 groups × 2 slots).

### Glossary

| Term | Meaning |
|---|---|
| **Group A / Group B** | A = personas matched to avid historical-fiction readers; B = matched controls who rarely read the genre. The contrast of interest. |
| **Probe (P1–P5)** | The lens a reader answers under. P1 Plausibility, P2 Knowledge-gap, P3 Stability, P4 Convention, P5 Salience. P1/P2/P4 invite period knowledge; P3/P5 ("control" probes) do not. |
| **pass-2** | Each reader answers twice — pass-1 generic, pass-2 under the probe. All analyses use pass-2. |
| **η² (eta-squared)** | Share of a dimension's variance explained by which probe was used. 0 = probe doesn't matter, 1 = probe explains everything. |
| **Attention dimensions** | What an LLM coder scores per response: *period specificity, concreteness, outside knowledge* (0–5 ratings) and *locations cited, historical anchors* (counts of items named). |
| **Convention frame** | The kind of craft a reader invokes: *historical* (period craft), *generic-genre* (story craft), *mixed*, or *none*. One label per response. |

### How the coding works

Every figure rests on numbers, but the readers produce free text. A separate
LLM pass converts each pass-2 response into the scores above. For each of the 720
responses we make **one independent call** to a coder model (deepseek-v4-flash):

- **Blind input.** The coder sees *only the reader's response text* — not the
  story, not the group label (A or B), not which probe was used. It therefore
  cannot tilt a score toward an expected result, which is what lets the A-vs-B and
  per-probe comparisons mean something.
- **Fixed rubric.** A system prompt tells it to score *attention*, not story
  quality, and to use the full 0–5 range (5 = expert-level attention, 0 = none).
  It returns one JSON object: the three 0–5 ratings, the two counts, the
  convention-frame label, and a one-clause evidence note.
- **Settings.** Temperature 0.2 (near-deterministic) so re-coding is stable;
  outputs are parsed strictly, with any missing field defaulting to 0 / "none".

So "Group A scores 4.2 on period specificity under P3" is the mean of 72
independent, blind, rubric-based codings — not a human read or a keyword count.

---

## 1. Probes drive what readers attend to

A one-way permutation test (10,000 shuffles of the probe label) asks how much of
each attention dimension's variance is explained by *which probe* the reader
answered:

![Figure 1](figures_intermediate/fig_c1_eta2.png)

***Figure 1. Probes shape what readers notice.** η² is the share of an attention
dimension's variance explained by which probe the reader answered (one-way
permutation test, 10,000 shuffles of the probe label; every bar p ≤ 0.001).
**(a)** the full sample (A + B pooled, n = 720); **(b)** Group A only (n = 360);
**(c)** Group B only (n = 360) — all on a shared scale, dimensions ordered by the
full-sample η². The probe effect is large for* outside knowledge *and* period
specificity *(~0.3 overall) and survives inside each group, so it is not an
artifact of group composition. It is in fact **stronger in Group B** (0.41 /
0.38) than Group A (0.25 / 0.26): A's historical-fiction readers bring period
attention fairly regardless of probe (a high, probe-insensitive baseline —
cf. Fig. 3), whereas the controls' attention is moved more by the probe itself.*

| Dimension | η² (probe, full sample) | F-like | p |
|---|---:|---:|---:|
| period_specificity | 0.309 | 0.45 | 0.0001 |
| knowledge_invoked | 0.319 | 0.47 | 0.0001 |
| anchors | 0.158 | 0.19 | 0.0001 |
| concreteness | 0.108 | 0.12 | 0.0001 |
| locations_cited | 0.040 | 0.04 | 0.0001 |

Probe identity explains **~31% of the variance** in *outside knowledge* and
*period specificity* — the most semantically loaded dimensions. The per-probe
attention profiles:

![Figure 2](figures_intermediate/fig_c1_probe_dim_heatmap.png)

***Figure 2. Each probe elicits a distinct attention profile.** Each cell is the
mean over the 144 reader responses for that probe (720 total). The five
dimensions are **two different units**, shown in two panels: **left** = the mean
of a **0–5 rubric rating** (period specificity, concreteness, outside
knowledge); **right** = the mean **count of items named per response** (locations
cited = distinct passage locations the reader pointed to; historical anchors =
concrete period objects/terms/events named). So "P3 = 0.99 on locations cited"
means readers under the Stability probe pointed to **~1 specific location per
response on average** (often none) — not a 0.99-out-of-5 score; likewise
"anchors = 5.62" for P1 is a count, not a rating. Each panel has its own white
(low) → dark blue (high) scale. Reading across a row gives the probe's
signature: **P3 Stability is the coolest on every dimension** — it asks about
the prose, not the period — yet that is exactly the probe where the A-vs-B gap
appears (Fig. 3).*

The reader pipeline is not producing homogeneous boilerplate; each probe elicits
a distinct attention signature.

---

## 2. The A-vs-B contrast surfaces on the "control" probes (P3, P5)

Does the HF-reader (A) vs control (B) contrast produce different attention, and
is that contrast probe-dependent? A–B gap per (probe, dimension), 5,000-perm:

![Figure 3](figures_intermediate/fig_c1_probe_group_dim.png)

***Figure 3. HF-reader "disposition" surfaces on the control probes (P3, P5).**
Each panel is one attention dimension; within it, paired bars compare Group A
(historical-fiction readers, blue) and Group B (matched controls, orange) for
each of the five probes, with 95% confidence intervals. A vermillion ★ and
bracket flag the probe×dimension cells where the A–B gap is significant
(p < 0.05, 5,000-perm). The significant gaps cluster on **P3 Stability** (4 of 5
dimensions) and **P5 Salience** (3 of 5) — the two probes that never ask for
period knowledge — so A's historical-fiction priors bleed in even when the
prompt doesn't invite them. This is a disposition effect, not prompt-following.*

| Probe | Significant A-B gaps (p < 0.05, pass2) |
|---|---|
| P1 Plausibility | 1/5 (locations_cited, +B) |
| P2 Knowledge-gap | 0/5 |
| **P3 Stability** | **4/5** (period_specif +0.76, concrete +0.71, knowl_invoked +0.82, anchors +1.06; all p<0.01) |
| P4 Convention | 0/5 |
| **P5 Salience** | **3/5** (period_specif +0.85, concrete +0.46, knowl_invoked +1.11; p<0.01) |

The same pattern shows up in the **convention frame** readers invoked (the
categorical dimension; one label per pass-2 response, assigned by an LLM coder
from the response text alone):

![Figure 4](figures_intermediate/fig_c1_convention_type.png)

***Figure 4. Group A imports a "historical" reading frame even on the control
probes.** For each probe, two stacked bars (A, then B) give the share of pass-2
responses by the kind of convention the reader invoked — one label per response,
assigned by an LLM coder (deepseek-v4-flash) from the response text alone (not
the story): historical = period craft (blue), generic-genre = story craft
(vermillion), mixed (green), or none (grey); n = 72 per bar (36 stories × 2
slots). On the priors-forcing probes (P1, P2) both groups are ~80% historical,
leaving no room to differ. On the control probes **P3 and P5, B's generic-genre
share jumps to ~28% (vs A's ~3–6%)** while A stays historical — the
convention-frame signature of the same disposition effect in Fig. 3 (A-vs-B χ²:
P3 = 19.0, P5 = 20.9, both p = 0.0002; P1/P2/P4 n.s.). "generic-genre" is
essentially a B-group tell that only surfaces where the probe doesn't force a
frame.*

Underlying counts (pass-2; n = 72 per probe×group cell), as **% (raw count)**:

| Probe | Group | `historical` | `generic-genre` | `mixed` | `none` |
|---|---|---:|---:|---:|---:|
| P1 Plausibility | A | 86% (62) | 0% (0) | 13% (9) | 1% (1) |
| P1 Plausibility | B | 81% (58) | 0% (0) | 18% (13) | 1% (1) |
| P2 Knowledge-gap | A | 81% (58) | 0% (0) | 11% (8) | 8% (6) |
| P2 Knowledge-gap | B | 72% (52) | 1% (1) | 17% (12) | 10% (7) |
| **P3 Stability** | **A** | **60% (43)** | **6% (4)** | **13% (9)** | **22% (16)** |
| **P3 Stability** | **B** | **31% (22)** | **28% (20)** | **10% (7)** | **32% (23)** |
| P4 Convention | A | 50% (36) | 14% (10) | 36% (26) | 0% (0) |
| P4 Convention | B | 35% (25) | 22% (16) | 43% (31) | 0% (0) |
| **P5 Salience** | **A** | **57% (41)** | **3% (2)** | **1% (1)** | **39% (28)** |
| **P5 Salience** | **B** | **35% (25)** | **28% (20)** | **6% (4)** | **32% (23)** |

Per-probe χ² of A-vs-B independence on the convention frame (df = 3; permutation
floor p = 0.0002 = 1/5000):

| Probe | χ² | p | A-vs-B differs? |
|---|---:|---:|---|
| P1 Plausibility | 0.86 | 0.73 | no |
| P2 Knowledge-gap | 2.20 | 0.55 | no |
| **P3 Stability** | **18.96** | **0.0002** | **yes** |
| P4 Convention | 3.81 | 0.16 | no |
| **P5 Salience** | **20.90** | **0.0002** | **yes** |
| across all 5 probes | 247.15 (df=12) | 0.0002 | yes |

The `generic-genre` column is the tell — a B-group signature that only appears on
the unconstrained probes (P3: A=4 vs B=20; P5: A=2 vs B=20). On P1/P2 both groups
are near-saturated on `historical` (the probe forces the frame); on P4 the
explicit Convention probe pushes everyone toward conventions. Only on the two
"control" probes, where nothing in the prompt forces a frame, does the reader's
disposition fill the vacuum.

### What the divergence actually looks like (one story, cell-09)

To check that these codings track a real difference and not a coder artifact, here
are verbatim excerpts from **one story** — a Southern-Song minister reading siege
dispatches (cell-09). The story is held fixed; only the **group** (A = HF reader,
B = control) and the **probe** vary. Each excerpt is tagged with the coder's
pass-2 scores.

**Priors-forcing probe (P2 Knowledge-gap) — the groups converge.** Both readers
supply real period knowledge; the control reader does so *despite* announcing she
is out of her genre:

> **Group A** *(historical; period 5, knowledge 5):* "…any reader who knows the
> history of the Song Dynasty's fall knows that Xiangyang *did* fall eventually,
> that the counterweight trebuchets (the 'hui hui pao') were decisive, and that
> the Southern Song collapsed within a few years."

> **Group B** *(historical; period 4, knowledge 4):* "I mostly read contemporary
> romance and thrillers… historical fiction like this feels really different to
> me. […] The whole setup with Xiangyang under Mongol siege is based on a real
> historical event (the Song dynasty siege that eventually fell in 1273)… it
> name-drops 'Prime Minister Jia's faction' — Jia Sidao, the corrupt Song
> chancellor."

When the probe asks what the text assumes you know about the history, the
control reader **demonstrably has** the period knowledge (Jia Sidao, 1273, the
trebuchet). So the gap on the control probes below is **not** a capability gap.

**Control probe (P3 Stability) — the same story, the groups diverge.** Asked
only whether the prose called attention to itself, both readers flag the *same two
lines* (a narrator's aside about future historians; an opening maple-leaf image) —
but read them through opposite frames:

> **Group A** *(historical; period 4, knowledge 5):* "…a real minister in **1273**
> wouldn't be worrying about how historians would judge his *feelings*… But I'd
> argue that in historical fiction, especially of this quality, those moments of
> visibility can be part of the pleasure."

> **Group B** *(none; period 0, knowledge 0):* "Alright, I see you, author. […]
> the very first line about the maple surrendering… felt a bit too *on the nose*…
> we get it, autumn = decay and loss. It's pretty, but it's a little
> heavy-handed."

Same story, same noticed lines — but A reaches for the period (a 1273 minister,
historical-fiction craft) while B stays in generic story-craft (foreshadowing,
heavy-handedness) and invokes no history at all. That single contrast is the
whole disposition effect in miniature: it is what drives Group B's `generic-genre`
and `none` shares on P3/P5 in Figure 4, and the A–B rating gaps in Figure 3.

---

## Bottom line

1. **Probes differentiate attention** strongly (η² up to ~0.32) and do so
   *within* each group, not just overall (§1).
2. **The A-vs-B difference is concentrated on the two "control" probes** (P3, P5)
   that don't ask for period knowledge — so it is a **disposition effect** (A's
   historical-fiction priors leak in unprompted), not prompt-following (§2).
