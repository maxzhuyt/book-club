# Book Club V3 — Midway handoff

Snapshot of the V3 intermediate state, including all scripts (with relative
paths so the project relocates cleanly), the full run state, and a coding-based
analysis with figures.

The primary report is at
`book-club-materials/experiments_v3/report_v3/REPORT_V3_INTERMEDIATE.md` — read
that first. This handoff doc covers only how to **resume / extend** the work.

## TL;DR — get it running on Midway

```bash
# 1. Extract anywhere you like — paths are all script-relative
cd ~                                          # or /project/<group>, /scratch/...
tar xzf book-club-v3-intermediate.tar.gz
cd book-club-v3-intermediate

# 2. Set the OpenRouter API key. Three options, pick one:
#    a) export it for the session
export NARRATIVE=sk-or-...
#    b) put it in ~/.env  (key=value, one per line)
echo 'NARRATIVE=sk-or-...' > ~/.env
#    c) any .env file pointed at by $DOTENV_PATH

# 3. Create a venv with the deps
python -m venv .venv
source .venv/bin/activate
pip install openai matplotlib numpy pandas pyarrow

# 4. Smoke-test: should print the report's headline numbers without making any LLM call
cd book-club-materials/experiments_v3
python analyze_v3_differentiation.py    # ~30s, reads run_20260531-022438/results/
python analyze_directives.py            # ~5s,  reads run_20260531-022438/directive_codings/

# 5. Re-generate figures (needs matplotlib)
python make_intermediate_figures.py     # writes report_v3/figures_intermediate/*.png
```

If `analyze_v3_differentiation.py` prints `eta2=0.309 ... p=0.0001` for
period_specificity, you're good — every analysis in the report is reproducible
from the bundled data.

## Bundle layout

```
book-club-v3-intermediate/
  HANDOFF.md                                       (this file)
  book-club-materials/
    canons/CRAFT_GUIDE.md                          (judge's rubric, ~40 KB)
    experiments/runners/                           (V1 writer + judge that V3 reuses)
      writer.py, judge.py, ...
    experiments_v3/
      *.py                                         (all V3 scripts, relative paths)
      groups.json                                  (final A/B reader membership + balance stats)
      personas_v3/                                 (20 .txt persona prompts + index.json)
      report_v3/
        REPORT_V3.md                               (reader-side findings, prior)
        REPORT_V3_WRITER_JUDGE.md                  (writer/judge findings, prior)
        REPORT_V3_INTERMEDIATE.md                  (intermediate progress, this session)
        figures_intermediate/                      (9 PNGs cited in the intermediate report)
      run_20260531-022438/                         (full run state — 85 MB)
        meta.json                                  (run config: 36 stories, seed=17)
        stories/<story_id>/story.txt               (36 originals)
        results/<group>/<story_id>/<probe>/agent-<slot>/
          pass1.txt, pass2.txt                     (LLM reader responses)
          coding.json                              (720/720 attention codings, clean)
          meta.json
        comparisons/                               (180 blind A/B comparator narratives)
        aggregated/                                (4 V3-original aggregation CSVs)
        aggregated_differentiation/                (THIS SESSION: probe x group x dim, A-B perm, conv chi²)
        writer_judge_v3/                           (pooled writer + judge: 36 stories x 2 variants)
        writer_judge_v3_blind/                     (rubric-leak control: 36 stories x 1 variant)
        revise_judge_byprobe_v3/                   (per-probe: 36 x 5 x 3 x 2 = 1080 revisions)
        revisions_dataset/                         (THIS SESSION: 1224 .md + manifest.csv + README)
        directive_codings/                         (THIS SESSION: 1188 directive codings)
        directive_aggregates/                      (THIS SESSION: pooling-strategy differentiation)
```

Per-story file count: 34 revisions in `revisions_dataset/`. Total bundle ≈ 85 MB.

## What's already done vs what to do next

Done in the most recent session (2026-06-07):
- All reader codings now 720/720 clean (re-coded 78 parse-error stubs + filled
  5 missing slots).
- 1,188 directives coded on a 10-dimension schema (commitment, specificity,
  takeaway_count, attribution rates, craft_vocab, edit_emphasis,
  conflict_acknowledged, conflict_resolution, note).
- Re-aggregated reader codings with 10k-perm permutation tests by
  (probe, group, dim) and cell-axis breakdowns.
- Aggregated directive codings with chi² + pairwise perm tests by
  (scope, variant, arm).
- Generated 9 figures.
- Wrote `REPORT_V3_INTERMEDIATE.md` with a glossary and explicit resource asks.

Not done (the proposed next-phase work from §5 of the report):
- Human-rater calibration pilot (~$5–6k, the central ask).
- Multi-judge replication (Opus 4.7 / GPT-5.2 / a smaller open model).
- Larger persona pool (n=20/group rather than n=10).
- V3 vs V2 probe-prime A/B on P1/P4.

## Resuming individual stages

All sweep scripts are idempotent — they check meta.json `status=="ok"` per unit
and skip-cache. To extend the work, just rerun with the new option.

```bash
cd book-club-materials/experiments_v3

# Re-aggregate / re-run analyses (cheap, no LLM calls)
python analyze_v3_differentiation.py    # writes <run>/aggregated_differentiation/
python analyze_directives.py            # writes <run>/directive_aggregates/
python make_intermediate_figures.py     # writes report_v3/figures_intermediate/

# Re-code something (LLM calls)
python extend_coding_coverage.py        # fills missing reader-response codings
python code_directives.py               # codes any newly-added directives
python compile_revisions_dataset.py     # rebuilds revisions_dataset/

# Re-run a sweep (LLM calls) — all resumable
python writer_judge_v3.py               # pooled, 36 stories x 2 consolidators
python writer_judge_v3_blind.py         # rubric-leak control, 36 stories x 1
python revise_judge_byprobe_v3.py       # per-probe, 36 x 5 x 3 x 2 = 1080
# All sweep scripts accept --max-concurrent <int>, --pilot <substring>
# and --aggregate-only.

# Re-run the upstream reader pipeline (LLM calls, very large)
python run_v3.py                        # 36 stories x 5 probes x 2 groups x 2 slots x 2 passes
                                        #   = 1,440 reader passes + 720 codings
                                        # Will write a new run_<timestamp>/ unless --resume
```

## Where each absolute path used to live (and how it resolves now)

| File | Was | Now |
|---|---|---|
| `experiments/runners/judge.py` | `CRAFT_GUIDE_PATH = Path("/home/maxzhuyt/.../canons/CRAFT_GUIDE.md")` | Resolves relative to script: `<repo>/canons/CRAFT_GUIDE.md`. Override with `CRAFT_GUIDE_PATH` env var. |
| `experiments_v3/run_v3.py` | `STORIES_DIR = Path("/home/maxzhuyt/.../generations-historical-fiction")` | Relative: `<repo>/generations-historical-fiction`. Override with `STORIES_DIR` env var. **Not bundled — only needed to start a *new* run.** Existing run uses `run_20260531-022438/stories/` already on disk. |
| `experiments_v3/writer_judge_v3.py`, `writer_judge_v3_blind.py`, `revise_judge_byprobe_v3.py` | `sys.path.insert(0, "/home/maxzhuyt/.../experiments/runners")` | `HERE.parent / "experiments" / "runners"` (script-relative). |
| `experiments_v3/client_v2.py` | `Path("/home/maxzhuyt/.env")` | Searches `$DOTENV_PATH`, `~/.env`, then `<repo>/.env` walking up. |
| `experiments_v3/groups.py`, `persona_build.py` | `CORPUS = Path("/home/maxzhuyt/book_club/scraping_goodreads")` | `Path.home() / "book_club" / "scraping_goodreads"`. Override with `GOODREADS_CORPUS` env var. **Not bundled — only needed to regenerate groups/personas from scratch.** Existing `groups.json` and `personas_v3/` are bundled. |

## What's not bundled (and how to get it if needed)

| Asset | Where it was | Why excluded | When you'd need it |
|---|---|---|---|
| `generations-historical-fiction/` (12 cell dirs × 3 runs of new_scene.txt + beats) | `~/book-club-materials/generations-historical-fiction/` | Already copied as `story.txt` into `run_20260531-022438/stories/`; the run dir is sufficient for everything downstream of generation | Only if starting a NEW run from scratch (i.e., re-running the reader pipeline) |
| Goodreads scrape (`user_books.parquet`, `books.parquet`, `user_reviews_full.parquet`) | `~/book_club/scraping_goodreads/` | Large; outside the project | Only if re-deriving A/B groups or regenerating personas |
| `canons/<short stories>` (Hilary Mantel, Chekhov, etc.) | `~/book-club-materials/canons/*.txt` | Only `CRAFT_GUIDE.md` is consumed by the pipeline | Never — not used by any V3 script |
| API key | `~/.env` | Secrets must not travel in tarballs | Always; set as described in TL;DR |
| Earlier V1/V2 reports/figures | `~/book-club-materials/experiments*` | V3 supersedes them | Reference only; not on critical path |

## Known artifacts at the time of handoff

These are flagged in `revisions_dataset/manifest.csv` (column `flag`):

| story | config | issue |
|---|---|---|
| cell-05-middle-sp-pure__run3 | P3_B_selective | writer returned a Chinese-language refusal (1 word) |
| cell-05-middle-sp-pure__run3 | P3_AB_neutral | same |
| cell-10-distant-sp-fantastical | P3_AB_neutral | hit writer max_tokens=8000 (~6000 words; original is itself 4,295 words) |

The 2 refusals can be retried cheaply with a different seed/model. The
truncation is structural (writer budget too small for that story); raising
WRITER_TOKENS to 12000 in `revise_judge_byprobe_v3.py` would resolve it.

## Reproducibility notes

- Model: `deepseek/deepseek-v4-flash` via OpenRouter (`client_v2.py:MODEL`).
- All sweep scripts use fixed seeds (writer_judge_v3 SEED=23,
  writer_judge_v3_blind SEED=41, revise_judge_byprobe_v3 SEED=47, run_v3 SEED=17).
- Analysis seeds: `analyze_v3_differentiation.py` RNG_SEED=17,
  `analyze_directives.py` RNG_SEED=31. Permutation counts are explicit in code
  (10k for probe-level, 5k for A-B and chi²).
- The intermediate report's numbers are reproducible bit-for-bit from the
  bundled data — no LLM calls needed to verify.

## Python deps

Minimal:
- `openai` (for the OpenRouter client — only needed if running sweeps / coders)
- `matplotlib` + `numpy` (for figures only)
- `pandas` + `pyarrow` (only needed for `groups.py` / `persona_build.py`)

Python 3.10+ recommended (uses `str | None` type hints in a few places).

```bash
pip install openai matplotlib numpy pandas pyarrow
```

## Quick reproducibility check

Run from `book-club-materials/experiments_v3/`:

```bash
python analyze_v3_differentiation.py 2>&1 | grep -E "eta2|gap=\+0\.764|gap=\+0\.847"
```

You should see:
```
  period_specificity      eta2=0.309  F=0.45  p=0.0001  n=720
  knowledge_invoked       eta2=0.319  F=0.47  p=0.0001  n=720
  P3 Stability      period_specificity      gap=+0.764  p=0.0038
  P5 Salience       period_specificity      gap=+0.847  p=0.0022
```

If those match, you've successfully relocated the project.

---

*Bundled 2026-06-07 from `/home/maxzhuyt/book-club-materials/`. Source code at
that path is updated to use relative paths, so the bundle and source match.*
