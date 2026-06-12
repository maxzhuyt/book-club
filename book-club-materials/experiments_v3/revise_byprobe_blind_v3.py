"""V3 per-probe SELECTIVE-BLIND consolidation + revision (rubric-leak control).

Companion to `revise_judge_byprobe_v3.py`, which runs the NEUTRAL and SELECTIVE
consolidators per (story, probe, arm). That sweep did NOT include the
SELECTIVE-BLIND variant — blind was only ever run at *pooled* scope
(`writer_judge_v3_blind.py`). This script fills that gap: it runs the
SELECTIVE-BLIND consolidator (same editorial-selection machinery as SELECTIVE
but with all CRAFT_GUIDE vocabulary stripped) across all five probes and all
three arms, producing one revision per (story, probe, arm).

Scope deliberately excludes judges. The intermediate report's findings are at
the coding layer (directive codings), not the LLM judge, so this only produces
the directives + revisions needed for the revisions dataset and for a future
per-probe rubric-leak coding pass.

Outputs land alongside the existing per-probe files so `compile_revisions_dataset.py`
picks them up with the canonical `P{i}_{arm}_selective_blind` config name:

  run_<ts>/revise_judge_byprobe_v3/<story>/<probe>/
    directive_<arm>_selective_blind.json    arm in {A, B, AB}
    revision_<arm>_selective_blind.txt
    meta_blind.json                         status for THIS sweep only

The existing meta.json (neutral/selective status) is never touched. Resumable:
re-running skips any (story, probe) whose meta_blind.json status == "ok", and
within a unit skips any arm whose directive+revision files already exist.

Token budgets and the consolidator user-prompt builder are identical to the
SELECTIVE per-probe run, so blind is apples-to-apples with selective.
"""
from __future__ import annotations
import argparse, asyncio, json, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments" / "runners"))

import client_v2  # noqa: E402
import consolidators_v3 as cons  # noqa: E402
import writer as v1_writer  # noqa: E402

SEED = 53  # documentation only; no stochastic step here (no judge flipping)
MAX_CONCURRENT = 4
DEFAULT_RUN_DIR = HERE / "run_20260531-022438"

# Match the SELECTIVE per-probe run exactly so blind is comparable.
CONS_TOKENS = 5000
WRITER_TOKENS = 8000

ARMS = ("A", "B", "AB")
VARIANT = "selective_blind"


def load_probe_readers(run_dir: Path, story_id: str, probe_key: str) -> dict[str, list[dict]]:
    """Returns {'A': [...], 'B': [...], 'AB': [...]} of pass2 reader items."""
    by_g = {"A": [], "B": []}
    for g in ("A", "B"):
        for slot in (0, 1):
            f = run_dir / "results" / g / story_id / probe_key / f"agent-{slot}" / "pass2.txt"
            if f.exists():
                by_g[g].append({"probe": probe_key, "group": g, "slot": slot,
                                "text": f.read_text().strip()})
    return {"A": by_g["A"], "B": by_g["B"], "AB": by_g["A"] + by_g["B"]}


async def consolidate_blind(story_id: str, probe_key: str, readers: list[dict]) -> dict:
    """Per-probe SELECTIVE-BLIND consolidation: blind system prompt + the same
    per-probe user builder the selective per-probe run uses."""
    user = cons.consolidator_user_byprobe(story_id, probe_key, readers)
    last_err = None
    for attempt in range(2):
        try:
            raw = await client_v2.chat(
                [{"role": "system", "content": cons.CONSOLIDATOR_SELECTIVE_BLIND_SYSTEM},
                 {"role": "user", "content": user}],
                max_tokens=CONS_TOKENS,
                temperature=0.2 + 0.2 * attempt)
            return cons.extract_json(raw)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"selective-blind consolidator failed twice: {last_err}")


async def write_revision(story_id: str, story_text: str, directive: dict) -> str:
    user_msg = v1_writer.writer_user(story_id, story_text, directive)
    return await client_v2.chat(
        [{"role": "system", "content": v1_writer.WRITER_SYSTEM},
         {"role": "user", "content": user_msg}],
        max_tokens=WRITER_TOKENS, temperature=0.55)


def load_story(run_dir: Path, story_id: str) -> str:
    """Load the original from the bundled run dir (the source generations dir is
    not bundled; run_<ts>/stories/<sid>/story.txt is the canonical copy used by
    every downstream step, e.g. compile_revisions_dataset.py)."""
    return (run_dir / "stories" / story_id / "story.txt").read_text()


def file_ok(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


async def run_unit(story_id: str, probe_key: str, run_dir: Path, sem: asyncio.Semaphore):
    out = run_dir / "revise_judge_byprobe_v3" / story_id / probe_key
    out.mkdir(parents=True, exist_ok=True)
    meta_p = out / "meta_blind.json"
    if meta_p.exists():
        try:
            if json.loads(meta_p.read_text()).get("status") == "ok":
                return {"id": f"{story_id}/{probe_key}", "status": "skip"}
        except Exception:
            pass

    meta = {"id": f"{story_id}/{probe_key}", "variant": VARIANT,
            "started": time.time(), "steps": []}
    async with sem:
        print(f"  ▸ {story_id}/{probe_key}", flush=True)
        try:
            arm_readers = load_probe_readers(run_dir, story_id, probe_key)
            if not arm_readers["A"] or not arm_readers["B"]:
                meta["status"] = "skip_missing_readers"
                meta["counts"] = {k: len(v) for k, v in arm_readers.items()}
                meta_p.write_text(json.dumps(meta, indent=2))
                return meta
            story_text = load_story(run_dir, story_id)

            for arm in ARMS:
                dir_p = out / f"directive_{arm}_{VARIANT}.json"
                rev_p = out / f"revision_{arm}_{VARIANT}.txt"
                if file_ok(dir_p) and file_ok(rev_p):
                    meta["steps"].append(f"{arm}:cached")
                    continue
                directive = await consolidate_blind(story_id, probe_key, arm_readers[arm])
                dir_p.write_text(json.dumps(directive, indent=2, ensure_ascii=False))
                revised = await write_revision(story_id, story_text, directive)
                rev_p.write_text(revised)
                meta["steps"].append(f"{arm}:fresh")

            meta["status"] = "ok"
        except Exception as e:
            meta["status"] = "error"; meta["error"] = str(e)
            meta["tb"] = traceback.format_exc()
        finally:
            meta["finished"] = time.time()
            meta_p.write_text(json.dumps(meta, indent=2))
        print(f"  ▸ {story_id}/{probe_key}  {meta['status']}", flush=True)
        return meta


# ===================== ENTRYPOINT =====================

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--pilot", default=None,
                    help="substring match on story_id; runs only matching stories")
    ap.add_argument("--probes", default=None,
                    help="comma-separated probe keys, e.g. P1,P3")
    ap.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    meta = json.loads((run_dir / "meta.json").read_text())
    stories = meta["stories"]
    if args.pilot:
        stories = [s for s in stories if args.pilot in s]
    pks = (args.probes.split(",") if args.probes else cons.PROBE_ORDER)
    units = [(s, pk) for s in stories for pk in pks]
    print(f"revise_byprobe_blind_v3 over {len(units)} (story,probe) units "
          f"× {len(ARMS)} arms, concurrency={args.max_concurrent}")
    sem = asyncio.Semaphore(args.max_concurrent)
    results = await asyncio.gather(*[run_unit(s, pk, run_dir, sem) for s, pk in units])

    from collections import Counter
    statuses = Counter(x.get("status", "?") for x in results)
    print(f"\n=== done: {dict(statuses)} ===")


if __name__ == "__main__":
    asyncio.run(main())
