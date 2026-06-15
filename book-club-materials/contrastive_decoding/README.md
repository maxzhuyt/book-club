# contrastive_decoding — Book-club persona simulation with system-prompt contrastive decoding

The original contrastive-decoding experiments: a simulated book club of
Goodreads-grounded reader personas that discuss a story over a four-phase
protocol, with **system-prompt contrastive decoding** (Dong et al., 2026) used to
keep the personas' voices distinct rather than collapsing to a single "assistant"
voice. Supports a single-story mode and an A-vs-B comparative mode (which of two
drafts should the author ship). Run across several open models (Qwen2.5-14B,
Qwen3-14B/32B, Qwen3.5-9B, Mistral-Small-24B, Gemma3-12B).

> The design write-ups and the reference paper (`DESIGN.md`,
> `CONTRASTIVE_DECODING_DESIGN.md`, `CONTRASTIVE_DECODING_COMPARATIVE_DESIGN.md`,
> `contrastive_decoding_paper.pdf`) and the `closed_models/` plan docs were left
> in the original `/project/jevans/maxzhuyt/contrastive_decoding/` directory —
> this folder holds the **code and data** only.

## Layout

| Path | Role |
|---|---|
| `simulation/src/cd/` | CD core: `decoder.py` (two-branch sampling), `prompts.py`, `alpha.py` (per-persona α schedule) |
| `simulation/src/bookclub/` | `cast.py`, `simulate.py` (single story), `simulate_compare.py` (A vs B), `summarize*.py` |
| `simulation/src/smoke_test.py` | CD validation (the paper's 6-year-old / melting-point check) |
| `simulation/scripts/` | `weekend_run.sh`, `large_models_run.sbatch`, `smoke_models.py` |
| `simulation/outputs_*/` | Per-model run results (transcripts, summaries, votes) — **git-ignored** (regenerable) |
| `personas/` | 8 Goodreads-grounded personas (3-layer system prompts) + `story_1.md`, `story_2.md` |
| `closed_models/src/` | Stubs for lifting persona fidelity on closed API models (few-shot / best-of-N + CD scorer) |

## Run

```bash
cd simulation/src
# single-story 4-phase book club
python -m bookclub.simulate --model-id Qwen/Qwen2.5-14B-Instruct \
    --story ../../personas/story_1.md --seed 7 --rounds 2
# comparative (which draft to ship)
python -m bookclub.simulate_compare --model-id Qwen/Qwen2.5-14B-Instruct \
    --story-a ../../personas/story_1.md --story-b ../../personas/story_2.md --seed 13
```

`simulation/scripts/large_models_run.sbatch` runs the 30B-range sweep on 2× H200
and was updated to this folder's new path after the move.
