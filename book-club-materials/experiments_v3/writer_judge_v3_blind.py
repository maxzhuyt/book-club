"""V3 pooled writer + judge — rubric-leak control.

Adds a third consolidator variant, SELECTIVE-BLIND, which has the same
editorial-selection machinery as SELECTIVE but no references to CRAFT_GUIDE
values. Runs one revision per story and judges it three ways:

  - revised_selective_blind  vs original
  - revised_selective_blind  vs revised_selective   (rubric leak test)
  - revised_selective_blind  vs revised_neutral     (commitment-vs-aggregation
                                                     under no rubric)

Reuses readers and revisions already on disk from writer_judge_v3. Resumable.

Layout under run_<ts>/writer_judge_v3_blind/<story>/:
  directive_selective_blind.json, revised_selective_blind.txt
  judge_blind_vs_orig.json,       blinding_blind_vs_orig.json
  judge_blind_vs_selective.json,  blinding_blind_vs_selective.json
  judge_blind_vs_neutral.json,    blinding_blind_vs_neutral.json
  meta.json
"""
from __future__ import annotations
import argparse, asyncio, json, random, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments" / "runners"))

import client_v2  # noqa: E402
import consolidators_v3 as cons  # noqa: E402
import writer as v1_writer  # noqa: E402
import judge as v1_judge   # noqa: E402
import run_v3 as r          # noqa: E402

SEED = 41
MAX_CONCURRENT = 8
DEFAULT_RUN_DIR = HERE / "run_20260531-022438"

CONS_TOKENS = 6000
WRITER_TOKENS = 8000
JUDGE_TOKENS = 8000


def load_all_readers(run_dir: Path, story_id: str) -> list[dict]:
    out = []
    for pk in cons.PROBE_ORDER:
        for g in ("A", "B"):
            for slot in (0, 1):
                f = run_dir / "results" / g / story_id / pk / f"agent-{slot}" / "pass2.txt"
                if f.exists():
                    out.append({"probe": pk, "group": g, "slot": slot,
                                "text": f.read_text().strip()})
    return out


async def consolidate_blind(story_id: str, readers: list[dict]) -> dict:
    user = cons.consolidator_user_pooled(story_id, readers)
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


async def judge_paired(story_id: str, left: str, right: str,
                       left_label: str, right_label: str, rng: random.Random):
    flip = rng.random() < 0.5
    if flip:
        A, B = right, left
        blinding = {"A": right_label, "B": left_label}
    else:
        A, B = left, right
        blinding = {"A": left_label, "B": right_label}
    user_msg = v1_judge.judge_user(story_id, A, B)
    raw = await client_v2.chat(
        [{"role": "system", "content": v1_judge.judge_system()},
         {"role": "user", "content": user_msg}],
        max_tokens=JUDGE_TOKENS, temperature=0.20)
    try:
        parsed = cons.extract_json(raw)
    except Exception as e:
        parsed = {"_parse_error": str(e), "_raw": raw}
    return blinding, parsed


def file_ok(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def load_existing_revision(run_dir: Path, story_id: str, variant: str) -> str | None:
    p = run_dir / "writer_judge_v3" / story_id / f"revised_{variant}.txt"
    return p.read_text() if p.exists() else None


async def run_one(story_id: str, run_dir: Path, sem: asyncio.Semaphore):
    out = run_dir / "writer_judge_v3_blind" / story_id
    out.mkdir(parents=True, exist_ok=True)
    meta_p = out / "meta.json"
    if meta_p.exists():
        try:
            if json.loads(meta_p.read_text()).get("status") == "ok":
                return {"story": story_id, "status": "skip"}
        except Exception:
            pass

    rng = random.Random(f"{story_id}/{SEED}")
    meta = {"story": story_id, "started": time.time(), "steps": []}
    async with sem:
        print(f"  ▸ {story_id}", flush=True)
        try:
            # --- need both existing revisions for the head-to-head judges
            rev_selective = load_existing_revision(run_dir, story_id, "selective")
            rev_neutral   = load_existing_revision(run_dir, story_id, "neutral")
            if rev_selective is None or rev_neutral is None:
                meta["status"] = "skip_missing_baseline"
                meta["missing"] = {"selective": rev_selective is None,
                                   "neutral":   rev_neutral is None}
                meta_p.write_text(json.dumps(meta, indent=2))
                return meta

            # --- selective-blind consolidate + write (cached on disk)
            dir_p = out / "directive_selective_blind.json"
            rev_p = out / "revised_selective_blind.txt"
            if file_ok(dir_p) and file_ok(rev_p):
                rev_blind = rev_p.read_text()
                meta["steps"].append("blind:cached")
            else:
                readers = load_all_readers(run_dir, story_id)
                if len(readers) < 12:
                    meta["status"] = "skip_too_few_readers"
                    meta["n_readers"] = len(readers)
                    meta_p.write_text(json.dumps(meta, indent=2))
                    return meta
                story_text = r.load_story(story_id)
                directive = await consolidate_blind(story_id, readers)
                dir_p.write_text(json.dumps(directive, indent=2, ensure_ascii=False))
                rev_blind = await write_revision(story_id, story_text, directive)
                rev_p.write_text(rev_blind)
                meta["steps"].append("blind:fresh")

            story_text = r.load_story(story_id)

            # --- 3 judge faces
            faces = [
                ("blind_vs_orig",      rev_blind, story_text,
                 "revised_selective_blind", "original"),
                ("blind_vs_selective", rev_blind, rev_selective,
                 "revised_selective_blind", "revised_selective"),
                ("blind_vs_neutral",   rev_blind, rev_neutral,
                 "revised_selective_blind", "revised_neutral"),
            ]
            for face, left, right, lL, rL in faces:
                jp = out / f"judge_{face}.json"
                bp = out / f"blinding_{face}.json"
                if file_ok(jp) and file_ok(bp):
                    meta["steps"].append(f"{face}:cached")
                    continue
                blinding, jud = await judge_paired(story_id, left, right, lL, rL, rng)
                jp.write_text(json.dumps(jud, indent=2, ensure_ascii=False))
                bp.write_text(json.dumps(blinding, indent=2))
                meta["steps"].append(f"{face}:fresh")

            meta["status"] = "ok"
        except Exception as e:
            meta["status"] = "error"; meta["error"] = str(e)
            meta["tb"] = traceback.format_exc()
        finally:
            meta["finished"] = time.time()
            meta_p.write_text(json.dumps(meta, indent=2))
        print(f"  ▸ {story_id}  {meta['status']}", flush=True)
        return meta


def _winner_real(judgment: dict, blinding: dict) -> str:
    w = judgment.get("overall_winner")
    if w in ("A", "B"):
        return blinding.get(w, "?")
    return "tied" if w == "tied" else "?"


def aggregate(run_dir: Path):
    from collections import Counter
    base = run_dir / "writer_judge_v3_blind"
    rows = []
    for d in sorted(base.iterdir()) if base.exists() else []:
        if not d.is_dir():
            continue
        m = d / "meta.json"
        if not m.exists():
            continue
        try:
            meta = json.loads(m.read_text())
        except Exception:
            continue
        if meta.get("status") != "ok":
            continue
        row = {"story": d.name}
        for face in ("blind_vs_orig", "blind_vs_selective", "blind_vs_neutral"):
            jp = d / f"judge_{face}.json"
            bp = d / f"blinding_{face}.json"
            if not (jp.exists() and bp.exists()):
                continue
            try:
                jud = json.loads(jp.read_text())
                blind = json.loads(bp.read_text())
            except Exception:
                continue
            row[f"{face}_winner"] = _winner_real(jud, blind)
            row[f"{face}_margin"] = jud.get("overall_margin", "")
        rows.append(row)

    def tally(field):
        return Counter(r.get(field, "?") for r in rows if field in r)

    print(f"\n=== Selective-blind probe, n={len(rows)} stories ===")
    for face in ("blind_vs_orig", "blind_vs_selective", "blind_vs_neutral"):
        t = tally(f"{face}_winner")
        print(f"  [{face}]  {dict(t)}")

    summary = {
        "n": len(rows),
        "rows": rows,
        "tallies": {face: dict(tally(f"{face}_winner"))
                    for face in ("blind_vs_orig", "blind_vs_selective", "blind_vs_neutral")},
    }
    (base / "_aggregate.json").write_text(json.dumps(summary, indent=2))
    return summary


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--pilot", default=None,
                    help="substring match on story_id; runs only matching stories")
    ap.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    if args.aggregate_only:
        aggregate(run_dir); return

    meta = json.loads((run_dir / "meta.json").read_text())
    stories = meta["stories"]
    if args.pilot:
        stories = [s for s in stories if args.pilot in s]
    print(f"writer_judge_v3_blind over {len(stories)} stories, "
          f"concurrency={args.max_concurrent}")
    sem = asyncio.Semaphore(args.max_concurrent)
    await asyncio.gather(*[run_one(s, run_dir, sem) for s in stories])
    aggregate(run_dir)


if __name__ == "__main__":
    asyncio.run(main())
