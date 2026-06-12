"""V3 per-probe writer + judge with full factorial.

For each (story, probe), three reader configurations:
  - A-only   : 2 readers (A:0, A:1)
  - B-only   : 2 readers (B:0, B:1)
  - AB-joint : 4 readers (A:0, A:1, B:0, B:1)
Each config × both consolidators (neutral, selective). 6 revisions per (story, probe).

Judges per (story, probe, consolidator):
  - A_vs_orig
  - B_vs_orig
  - AB_vs_orig
  - A_vs_B        (the V2-comparable A-informed vs B-informed contrast)

We use V1's writer + V1's blind paired judge with CRAFT_GUIDE — same writer/judge
as the pooled pipeline so the only thing that varies across runs is the consolidation
step.

Layout under run_<ts>/revise_judge_byprobe_v3/<story>/<probe>/:
  directive_<arm>_<variant>.json     arm ∈ {A,B,AB}, variant ∈ {neutral,selective}
  revision_<arm>_<variant>.txt
  judge_<variant>_<face>.json        face ∈ {A_vs_orig,B_vs_orig,AB_vs_orig,A_vs_B}
  blinding_<variant>_<face>.json
  meta.json
"""
from __future__ import annotations
import argparse, asyncio, json, random, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments" / "runners"))

import client_v2, probes  # noqa: E402
import consolidators_v3 as cons  # noqa: E402
import writer as v1_writer  # noqa: E402
import judge as v1_judge   # noqa: E402
import run_v3 as r          # noqa: E402

SEED = 47
MAX_CONCURRENT = 4
DEFAULT_RUN_DIR = HERE / "run_20260531-022438"

# Token budgets — generous so per-probe consolidations (small contexts) don't waste
# headroom and pooled-style AB consolidations (4 readers) don't length-flake.
CONS_TOKENS = 5000
WRITER_TOKENS = 8000
JUDGE_TOKENS = 8000

ARMS = ("A", "B", "AB")
VARIANTS = ("neutral", "selective")
FACES = ("A_vs_orig", "B_vs_orig", "AB_vs_orig", "A_vs_B")


def load_probe_readers(run_dir: Path, story_id: str, probe_key: str) -> dict[str, list[dict]]:
    """Returns dict with keys 'A', 'B', 'AB' -> list[reader_item]."""
    by_g = {"A": [], "B": []}
    for g in ("A", "B"):
        for slot in (0, 1):
            f = run_dir / "results" / g / story_id / probe_key / f"agent-{slot}" / "pass2.txt"
            if f.exists():
                by_g[g].append({"probe": probe_key, "group": g, "slot": slot,
                                "text": f.read_text().strip()})
    return {"A": by_g["A"], "B": by_g["B"], "AB": by_g["A"] + by_g["B"]}


async def consolidate(story_id: str, probe_key: str, readers: list[dict], variant: str) -> dict:
    sysm = (cons.CONSOLIDATOR_NEUTRAL_SYSTEM if variant == "neutral"
            else cons.CONSOLIDATOR_SELECTIVE_SYSTEM)
    user = cons.consolidator_user_byprobe(story_id, probe_key, readers)
    last_err = None
    for attempt in range(2):
        try:
            raw = await client_v2.chat(
                [{"role": "system", "content": sysm}, {"role": "user", "content": user}],
                max_tokens=CONS_TOKENS,
                temperature=0.2 + 0.2 * attempt)
            return cons.extract_json(raw)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"consolidator failed twice: {last_err}")


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


async def run_unit(story_id: str, probe_key: str, run_dir: Path, sem: asyncio.Semaphore):
    out = run_dir / "revise_judge_byprobe_v3" / story_id / probe_key
    out.mkdir(parents=True, exist_ok=True)
    meta_p = out / "meta.json"
    if meta_p.exists():
        try:
            if json.loads(meta_p.read_text()).get("status") == "ok":
                return {"id": f"{story_id}/{probe_key}", "status": "skip"}
        except Exception:
            pass

    rng = random.Random(f"{story_id}/{probe_key}/{SEED}")
    meta = {"id": f"{story_id}/{probe_key}", "started": time.time(), "steps": []}
    async with sem:
        print(f"  ▸ {story_id}/{probe_key}", flush=True)
        try:
            arm_readers = load_probe_readers(run_dir, story_id, probe_key)
            if not arm_readers["A"] or not arm_readers["B"]:
                meta["status"] = "skip_missing_readers"
                meta["counts"] = {k: len(v) for k, v in arm_readers.items()}
                meta_p.write_text(json.dumps(meta, indent=2))
                return meta
            story_text = r.load_story(story_id)

            # --- consolidations + revisions (cached on disk)
            # rev[(arm, variant)] -> revised text
            rev: dict[tuple[str, str], str] = {}
            for arm in ARMS:
                for variant in VARIANTS:
                    dir_p = out / f"directive_{arm}_{variant}.json"
                    rev_p = out / f"revision_{arm}_{variant}.txt"
                    if file_ok(dir_p) and file_ok(rev_p):
                        rev[(arm, variant)] = rev_p.read_text()
                        meta["steps"].append(f"{arm}/{variant}:cached")
                        continue
                    directive = await consolidate(story_id, probe_key,
                                                  arm_readers[arm], variant)
                    dir_p.write_text(json.dumps(directive, indent=2, ensure_ascii=False))
                    revised = await write_revision(story_id, story_text, directive)
                    rev_p.write_text(revised)
                    rev[(arm, variant)] = revised
                    meta["steps"].append(f"{arm}/{variant}:fresh")

            # --- 4 judge faces × 2 variants = 8 judges (cached on disk)
            for variant in VARIANTS:
                face_pairs = {
                    "A_vs_orig":  (rev[("A",  variant)], story_text,
                                   f"A_{variant}",  "original"),
                    "B_vs_orig":  (rev[("B",  variant)], story_text,
                                   f"B_{variant}",  "original"),
                    "AB_vs_orig": (rev[("AB", variant)], story_text,
                                   f"AB_{variant}", "original"),
                    "A_vs_B":     (rev[("A",  variant)], rev[("B", variant)],
                                   f"A_{variant}",  f"B_{variant}"),
                }
                for face in FACES:
                    jp = out / f"judge_{variant}_{face}.json"
                    bp = out / f"blinding_{variant}_{face}.json"
                    if file_ok(jp) and file_ok(bp):
                        meta["steps"].append(f"{variant}/{face}:cached")
                        continue
                    left, right, lL, rL = face_pairs[face]
                    blinding, jud = await judge_paired(story_id, left, right, lL, rL, rng)
                    jp.write_text(json.dumps(jud, indent=2, ensure_ascii=False))
                    bp.write_text(json.dumps(blinding, indent=2))
                    meta["steps"].append(f"{variant}/{face}:fresh")

            meta["status"] = "ok"
        except Exception as e:
            meta["status"] = "error"; meta["error"] = str(e)
            meta["tb"] = traceback.format_exc()
        finally:
            meta["finished"] = time.time()
            meta_p.write_text(json.dumps(meta, indent=2))
        print(f"  ▸ {story_id}/{probe_key}  {meta['status']}", flush=True)
        return meta


# ===================== AGGREGATION =====================

def _winner_real(judgment: dict, blinding: dict) -> str:
    w = judgment.get("overall_winner")
    if w in ("A", "B"):
        return blinding.get(w, "?")
    return "tied" if w == "tied" else "?"


def aggregate(run_dir: Path):
    from collections import Counter, defaultdict
    base = run_dir / "revise_judge_byprobe_v3"
    if not base.exists():
        print("no run dir"); return
    # per-probe × variant × face tally
    tally = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    rows = []
    for story_d in sorted(base.iterdir()):
        if not story_d.is_dir():
            continue
        for probe_d in sorted(story_d.iterdir()):
            if not probe_d.is_dir():
                continue
            m = probe_d / "meta.json"
            if not m.exists():
                continue
            try:
                meta = json.loads(m.read_text())
            except Exception:
                continue
            if meta.get("status") != "ok":
                continue
            pk = probe_d.name
            row = {"story": story_d.name, "probe": pk}
            for variant in VARIANTS:
                for face in FACES:
                    jp = probe_d / f"judge_{variant}_{face}.json"
                    bp = probe_d / f"blinding_{variant}_{face}.json"
                    if not (jp.exists() and bp.exists()):
                        continue
                    try:
                        jud = json.loads(jp.read_text())
                        blind = json.loads(bp.read_text())
                    except Exception:
                        continue
                    real = _winner_real(jud, blind)
                    margin = jud.get("overall_margin", "")
                    row[f"{variant}_{face}"] = real
                    row[f"{variant}_{face}_margin"] = margin
                    tally[pk][variant][face][real] += 1
            rows.append(row)

    print(f"\n=== Per-probe writer_judge_v3, units={len(rows)} ===")
    for pk in cons.PROBE_ORDER:
        name = probes.PROBES[pk]["name"]
        print(f"\n--- Probe {pk} ({name}) ---")
        for variant in VARIANTS:
            print(f"  consolidator={variant}")
            for face in FACES:
                t = tally[pk][variant][face]
                print(f"    {face:>12}  {dict(t)}")

    summary = {
        "rows": rows,
        "tallies": {pk: {v: {f: dict(tally[pk][v][f]) for f in FACES}
                         for v in VARIANTS}
                    for pk in cons.PROBE_ORDER},
    }
    (base / "_aggregate.json").write_text(json.dumps(summary, indent=2))
    return summary


# ===================== ENTRYPOINT =====================

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--pilot", default=None,
                    help="substring match on story_id; runs only matching stories")
    ap.add_argument("--probes", default=None,
                    help="comma-separated probe keys, e.g. P1,P3")
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
    pks = (args.probes.split(",") if args.probes else cons.PROBE_ORDER)
    units = [(s, pk) for s in stories for pk in pks]
    print(f"revise_judge_byprobe_v3 over {len(units)} (story,probe) units, "
          f"concurrency={args.max_concurrent}")
    sem = asyncio.Semaphore(args.max_concurrent)
    await asyncio.gather(*[run_unit(s, pk, run_dir, sem) for s, pk in units])
    aggregate(run_dir)


if __name__ == "__main__":
    asyncio.run(main())
