# Book Club V3

**An LLM "book club" for probing how reader disposition and attentional framing shape the reading of counterfactual historical fiction — and whether an editor LLM that consolidates those readings improves a story without simply leaking the evaluator's rubric.**

Book Club V3 is a three-stage LLM pipeline run over 36 counterfactual historical-fiction stories. A panel of LLM "reader" personas reads each story under one of five attentional **probes**; an editor LLM **consolidates** the panel's reactions into a revision directive; a writer LLM **revises** the story and a judge LLM scores it. The design lets us separate three things that are usually confounded: what a *probe* makes readers attend to, what a reader's *disposition* (genre experience) adds on top, and whether an editor's gains come from genuine editorial judgment or from **leaking the judge's craft rubric**.

> **Status.** Research code + intermediate results. The headline numbers and every figure are reproducible from the small aggregate tables committed here; the bulky per-response LLM outputs (~15.6k files) are regenerable from the code and are not committed (see [Reproducing the data](#reproducing-the-data)).

---

## The question

Historical fiction asks a reader to hold two things at once: a story, and a *period* the story claims to depict. We ask:

1. **Do attentional probes differentiate reading?** If you prime a reader to attend to plausibility vs. prose stability vs. genre convention, do their reactions actually diverge — or do LLM readers produce the same boilerplate regardless?
2. **Does reader disposition leak in unprompted?** Group A personas are avid historical-fiction readers; Group B are matched controls who rarely read the genre. When a probe does *not* ask about the period, does Group A still import a "historical" reading frame?
3. **Is editorial gain real or rubric leak?** An editor that consolidates reader reactions into a revision can win on a judge's scorecard either by improving the story or by quietly echoing the judge's own vocabulary. We test this with a **blind** editor variant that shares no rubric language with the judge.

---

## Design

**Corpus — 36 stories.** A 12-cell grid generated in 3 independent runs each:

```
temporal distance   {recent, middle, distant}     (how far in the past)
narrative scope     {sp, sys}                      (cell-name axis)
counterfactual type {pure, fantastical}            (plausible alt-history vs. fantastical premise)
   3 × 2 × 2 = 12 cells × 3 runs = 36 stories       e.g. cell-09-distant-sp-pure__run2
```

**Readers — 20 personas per story, two groups (n = 10 each).** Personas are built from a Goodreads review corpus and balanced across groups on review length, analyticity, and book count, differing only in historical-fiction reading fraction:

| Group | Who | HF reading fraction |
|---|---|---|
| **A** | avid historical-fiction readers | ≈ 0.42 |
| **B** | matched controls | ≈ 0.04 |

**Probes — five attentional primes.** Each reader answers under one probe, **twice**: pass-1 generic, then pass-2 under the probe (all analyses use pass-2).

| Probe | Name | Type | Asks about… |
|---|---|---|---|
| P1 | Plausibility | priors-dependent | whether the world holds together |
| P2 | Knowledge-gap | priors-dependent | what the text assumes you know / leaves out |
| P3 | Stability | priors-independent\* | whether the *prose* calls attention to itself |
| P4 | Convention | priors-dependent | genre/period conventions invoked |
| P5 | Salience | priors-symmetric (control) | what stood out, no period framing |

\* P3 and P5 are the "control" probes — nothing in their prompt asks for period knowledge, so any Group-A vs. Group-B difference there is a **disposition** effect, not prompt-following.

---

## Pipeline

```
                 ┌─────────────────────────────────────────────────────────┐
   36 stories ──▶│ 1. READ        run_v3.py / read_pipeline.py              │
                 │   20 personas × 5 probes × 2 passes  → reader responses  │
                 │   coding.py    → 5 attention dims + convention frame     │
                 └─────────────────────────────────────────────────────────┘
                                            │
                 ┌─────────────────────────────────────────────────────────┐
                 │ 2. CONSOLIDATE  consolidators_v3.py                      │
                 │   editor turns the panel's reactions into a revision     │
                 │   directive, in three variants × three arms:             │
                 │     NEUTRAL · SELECTIVE · SELECTIVE-BLIND   (A / B / AB)  │
                 │   code_directives.py + code_directive_extras.py          │
                 │     → 10-dim directive coding + invention/camp           │
                 └─────────────────────────────────────────────────────────┘
                                            │
                 ┌─────────────────────────────────────────────────────────┐
                 │ 3. WRITE + JUDGE  writer_judge_v3.py (pooled)            │
                 │   revise_judge_byprobe_v3.py (per-probe)                 │
                 │   writer_judge_v3_blind.py (rubric-leak control)         │
                 │   writer revises → judge scores vs. original / head-to-  │
                 │   head, blind to which variant produced each revision    │
                 └─────────────────────────────────────────────────────────┘
```

**The three editor variants** (the core of the rubric-leak test):

- **NEUTRAL** — summarizes the panel without taking an editorial side.
- **SELECTIVE** — names the axes of disagreement, picks a side, refuses the other — reasoning *from the judge's craft rubric* (period voice, anachronism, counterfactual discipline, earned compression, …).
- **SELECTIVE-BLIND** — the same editorial machinery, but with **all rubric vocabulary stripped**; the editor reasons "from its own judgment." If BLIND tracks SELECTIVE, the gain is editorial commitment per se; if it collapses to NEUTRAL, SELECTIVE was winning by rubric leak. **SELECTIVE-BLIND is the reported default; SELECTIVE is kept only for the leak comparison.**

**Coding** is itself done by an LLM (`deepseek/deepseek-v4-flash`), blind to story/group/probe — see the [reader-pipeline brief](book-club-materials/experiments_v3/report_v3/REPORT_V3_READERS_BRIEF.md) for the exact rubric. Reader responses get 5 attention dimensions (`period_specificity`, `concreteness`, `knowledge_invoked` as 0–5 ratings; `locations_cited`, `anchors` as counts) plus a categorical convention frame. Directives get a 10-dimension schema plus relational invention-rate and craft-camp codings.

---

## Selected findings

From the [intermediate report](book-club-materials/experiments_v3/report_v3/REPORT_V3_INTERMEDIATE.md) and [reader brief](book-club-materials/experiments_v3/report_v3/REPORT_V3_READERS_BRIEF.md):

- **Probes differentiate attention.** Probe identity explains ≈ 31% of the variance (η²) in `period_specificity` and `knowledge_invoked` — and it does so *within* each group, so it is not an artifact of group composition.
- **Disposition leaks in on the control probes.** On P3/P5 (which never ask for period knowledge), Group A keeps a "historical" reading frame while Group B drops to generic story-craft: A-vs-B convention-frame χ² = 19.0 (P3) and 20.9 (P5), both *p* = 0.0002; P1/P2/P4 n.s. It is *not* a capability gap — controls supply real period knowledge when P2 asks for it.
- **Editorial gain survives the blind control** (writer/judge side) — see [REPORT_V3_WRITER_JUDGE.md](book-club-materials/experiments_v3/report_v3/REPORT_V3_WRITER_JUDGE.md) and §3.5 / §4.4 of the intermediate report.

---

## Repository layout

```
book-club/
├── README.md  ·  requirements.txt  ·  LICENSE  ·  HANDOFF.md   (dev resume notes)
└── book-club-materials/
    ├── canons/CRAFT_GUIDE.md                 judge's craft rubric (consumed by judge.py)
    ├── experiments/runners/                  V1 writer + judge, reused by the V3 sweeps
    └── experiments_v3/                        ← main code; run scripts from here
        ├── client_v2.py                       OpenRouter async client (reads $NARRATIVE)
        ├── groups.py · groups.json            A/B reader membership + balance stats
        ├── persona_build.py · personas_v3/    20 persona prompts + index
        ├── probes.py                          the five probe primes/elicitations
        ├── read_pipeline.py · coding.py · comparator.py
        ├── run_v3.py                          stage 1: the reader sweep
        ├── consolidators_v3.py                stage 2: NEUTRAL/SELECTIVE/BLIND editors
        ├── writer_judge_v3.py · _blind.py     stage 3: pooled write+judge (+ leak control)
        ├── revise_byprobe_blind_v3.py         stage 3: per-probe SELECTIVE-BLIND sweep
        ├── revise_judge_byprobe_v3.py         stage 3: per-probe write+judge
        ├── code_directives.py · code_directive_extras.py · extend_coding_coverage.py
        ├── analyze_v3.py · analyze_v3_differentiation.py · analyze_directives.py
        ├── compile_revisions_dataset.py
        ├── make_intermediate_figures.py       regenerates every figure from aggregates
        ├── report_v3/                          reports, figures, local viewer (serve_report.py)
        └── run_20260531-022438/                the run (aggregates + 36 story stimuli kept;
                                                raw per-response outputs gitignored)
```

---

## Setup

```bash
git clone https://github.com/maxzhuyt/book-club.git
cd book-club
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**API key.** The pipeline calls models through [OpenRouter](https://openrouter.ai). The client (`client_v2.py`) reads the key from the `NARRATIVE` environment variable, looking in `$DOTENV_PATH`, then `~/.env`, then a `.env` walking up from the script. Set it any one way:

```bash
export NARRATIVE=sk-or-...
# or:  echo 'NARRATIVE=sk-or-...' > ~/.env
```

`.env` files are git-ignored — **never commit your key.** A key is only needed for stages that make LLM calls (sweeps and coders); analysis and figure regeneration need none.

---

## Reproducing the figures (no API key, no raw data)

Every figure regenerates from the aggregate tables committed in `run_20260531-022438/`:

```bash
cd book-club-materials/experiments_v3
python make_intermediate_figures.py     # reads aggregated_differentiation/ + directive_aggregates/
                                        # writes report_v3/figures_intermediate/*.png
```

View the reports with the bundled local server (renders markdown + figures, with a
text-selection annotation layer):

```bash
python report_v3/serve_report.py --report report_v3/REPORT_V3_READERS_BRIEF.md --port 8911
# open http://127.0.0.1:8911/
```

---

## Reproducing the data

The raw per-response outputs (every reader pass, coding, revision, and judge call —
~15.6k files) are **regenerable from the code** and are not committed. To rebuild
them you need an OpenRouter key; all sweeps are idempotent (they skip units whose
`meta.json` says `status=="ok"`) and seeded. Order:

```bash
cd book-club-materials/experiments_v3
python run_v3.py                  # stage 1: reader sweep  → results/  (+ codings)
python writer_judge_v3.py         # stage 3: pooled write+judge
python writer_judge_v3_blind.py   # stage 3: rubric-leak control
python revise_judge_byprobe_v3.py # stage 3: per-probe
python code_directives.py         # code the directives
python compile_revisions_dataset.py
# then re-aggregate (no LLM calls) and re-plot:
python analyze_v3_differentiation.py   # rebuilds aggregated_differentiation/ from results/
python analyze_directives.py           # rebuilds directive_aggregates/
python make_intermediate_figures.py
```

`HANDOFF.md` documents seeds, resume behavior, known artifacts, and the exact
smoke-test numbers in detail.

### A note on the data

LLM-reader responses are generated under a fixed token budget; because the reader
model is a reasoning model, the hidden reasoning draws on that budget and ≈ 24% of
responses are cut off mid-sentence (the client keeps partial completions). This is
balanced enough across groups/probes that it does not drive the headline pattern,
but it is a real data-quality caveat — see the reader brief.

---

## Reports

| Report | Scope |
|---|---|
| [`REPORT_V3_READERS_BRIEF.md`](book-club-materials/experiments_v3/report_v3/REPORT_V3_READERS_BRIEF.md) | concise, external-facing: the reader pipeline (probes × groups), Figures 1–4 |
| [`REPORT_V3_INTERMEDIATE.md`](book-club-materials/experiments_v3/report_v3/REPORT_V3_INTERMEDIATE.md) | full intermediate report, all stages, 12 figures |
| [`REPORT_V3.md`](book-club-materials/experiments_v3/report_v3/REPORT_V3.md) · [`REPORT_V3_WRITER_JUDGE.md`](book-club-materials/experiments_v3/report_v3/REPORT_V3_WRITER_JUDGE.md) | prior reader-side and writer/judge findings |

---

## Model & license

- **Model:** `deepseek/deepseek-v4-flash` via OpenRouter (readers, editors, writer, judge, and coders).
- **License:** MIT — see [LICENSE](LICENSE).
