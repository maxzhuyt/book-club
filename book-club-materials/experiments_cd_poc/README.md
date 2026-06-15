# experiments_cd_poc — Contrastive decoding for reader-group differentiation

Proof-of-concept testing whether **contrastive decoding (CD)** increases the
differentiation between two reader groups (Group A = avid historical-fiction
readers, Group B = generic readers) when they respond to the same
historical-fiction passages under the `experiments_v3` probe protocol.

**Findings:** see [`REPORT_CD_POC.md`](REPORT_CD_POC.md) (and
`full_report_draft.md` for the narrative writeup). Headline: CD widens the A–B
gap — most clearly, it opens a group difference on the *convention* probe, which
is statistically flat under probing alone.

## Layout

| File | Role |
|---|---|
| `cd_decoder.py` | Dual-copy contrastive decoder (two Qwen3-32B bf16 copies, one per GPU) |
| `neg_prompt.py` | The "generic reader" negative system prompt used for the contrast |
| `run_cd_poc.py` | Grid runner: 3 stories × 3 probes × 2 groups × 4 personas × {α=0, α=1} |
| `run_cd_poc.sbatch` | SLURM job (4× H200, two cell-shards in parallel), smoke-gated |
| `code_responses.py` | Re-codes responses with the v3 attention coder via the DeepSeek API |
| `analyze_cd_poc.py` | A–B gaps (permutation tests) + convention-frame shares |
| `tfidf_separation.py` | Grader-free lexical (TF-IDF) A–B separation check |
| `aggregated/` | Result tables (CSV) — committed |
| `run_cd_poc/` | Per-response generations + codings — **git-ignored** (regenerable) |

Reuses `../experiments_v3` for the probes, personas, and coder prompts.

## Run

```bash
# generation (on a GPU node)
sbatch run_cd_poc.sbatch          # full grid; or: python run_cd_poc.py --smoke-only

# coding (needs DEEPSEEK_API_KEY in ~/.env)
python code_responses.py

# analysis
python analyze_cd_poc.py
python tfidf_separation.py
```

> Why Qwen3-32B and not the production model (DeepSeek-V4-Flash): CD needs paired
> full-vocabulary logits, which the API never exposes, and V4-Flash's FP4 experts
> have no kernel on our H200s. See `REPORT_CD_POC.md` §2.
