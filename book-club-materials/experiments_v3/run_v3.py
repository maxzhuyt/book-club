"""V2 orchestrator: 12 stories × 2 groups × (4 probes × 2 agents) two-phase reads,
then per-response coding and per-(story,probe) blind A/B comparison. Resume/retry by
skipping any (group,cell,probe,agent) whose meta.json says ok."""
from __future__ import annotations
import argparse, asyncio, json, random, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probes, read_pipeline, coding, comparator  # noqa: E402

# Defaults to <repo-root>/generations-historical-fiction relative to this file
# (which lives at <repo>/experiments_v3/run_v3.py). Override with STORIES_DIR
# env var if running from a different layout.
import os as _os
STORIES_DIR = Path(_os.environ.get("STORIES_DIR",
                                    HERE.parent / "generations-historical-fiction"))
SEED = 17
AGENTS_PER_PROBE = 2
MAX_CONCURRENT = 4


def all_cells():
    return sorted(d.name for d in STORIES_DIR.iterdir()
                  if d.is_dir() and d.name.startswith("cell-"))


RUNS = [1, 2, 3]


def all_stories():
    """36 story units. run-1 keeps the bare cell name (back-compat with the original
    12-cell run); run-2/run-3 get a __run{K} suffix."""
    out = []
    for cell in all_cells():
        for k in RUNS:
            out.append(cell if k == 1 else f"{cell}__run{k}")
    return out


def _resolve(story_id):
    if "__run" in story_id:
        cell, k = story_id.split("__run"); return cell, int(k)
    return story_id, 1


def load_story(story_id):
    cell, k = _resolve(story_id)
    return (STORIES_DIR / cell / f"run-{k}" / "new_scene.txt").read_text()


def load_groups():
    idx = json.loads((HERE / "personas_v3" / "index.json").read_text())
    A = [u for u, v in idx.items() if v["group"] == "A"]
    B = [u for u, v in idx.items() if v["group"] == "B"]
    return {"A": A, "B": B}


def persona_text(uid):
    return (HERE / "personas_v3" / f"{uid}.txt").read_text()


def agent_assignments(group_users):
    """Map 8 users to (probe, slot): 2 users per probe, in listed order."""
    out = []
    pk = list(probes.PROBES.keys())  # P1..P4
    for i, uid in enumerate(group_users[:len(pk) * AGENTS_PER_PROBE]):
        out.append((pk[i // AGENTS_PER_PROBE], i % AGENTS_PER_PROBE, uid))
    return out


async def run_agent(out_dir, group, cell, probe_key, slot, uid, story):
    d = out_dir / "results" / group / cell / probe_key / f"agent-{slot}"
    d.mkdir(parents=True, exist_ok=True)
    meta = {"group": group, "cell": cell, "probe": probe_key, "slot": slot,
            "user_id": uid, "started": time.time()}
    try:
        r = await read_pipeline.run_two_phase(persona_text(uid), probes.PROBES[probe_key], story)
        (d / "pass1.txt").write_text(r["pass1"]); (d / "pass2.txt").write_text(r["pass2"])
        c1 = await coding.code_response(r["pass1"])
        c2 = await coding.code_response(r["pass2"])
        (d / "coding.json").write_text(json.dumps({"pass1": c1, "pass2": c2}, indent=2))
        meta["status"] = "ok"
    except Exception as e:
        meta["status"] = "error"; meta["error"] = str(e); meta["tb"] = traceback.format_exc()
    finally:
        meta["finished"] = time.time()
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def agent_ok(out_dir, group, cell, probe_key, slot):
    mp = out_dir / "results" / group / cell / probe_key / f"agent-{slot}" / "meta.json"
    if not mp.exists():
        return False
    try:
        return json.loads(mp.read_text()).get("status") == "ok"
    except Exception:
        return False


async def run_reads(out_dir, stories, groups, resume):
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def guarded(group, cell, pk, slot, uid, story):
        async with sem:
            if resume and agent_ok(out_dir, group, cell, pk, slot):
                return {"status": "skip"}
            print(f"  ▸ {group} {cell} {pk} a{slot} ({uid})", flush=True)
            m = await run_agent(out_dir, group, cell, pk, slot, uid, story)
            print(f"  ▸ {group} {cell} {pk} a{slot} {m['status']}", flush=True)
            return m

    tasks = []
    for cell in stories:
        story = load_story(cell)
        for group in ("A", "B"):
            for pk, slot, uid in agent_assignments(groups[group]):
                tasks.append(guarded(group, cell, pk, slot, uid, story))
    for coro in asyncio.as_completed(tasks):
        await coro


async def run_comparisons(out_dir, stories):
    for cell in stories:
        for pk in probes.PROBES:
            cdir = out_dir / "comparisons" / cell / pk
            cdir.mkdir(parents=True, exist_ok=True)

            def answers(group):
                res = []
                for slot in range(AGENTS_PER_PROBE):
                    f = out_dir / "results" / group / cell / pk / f"agent-{slot}" / "pass2.txt"
                    if f.exists():
                        res.append(f.read_text())
                return res

            a, b = answers("A"), answers("B")
            if not a or not b:
                continue
            sub = random.Random(f"{cell}/{pk}/{SEED}")
            out = await comparator.compare(probes.PROBES[pk]["elicitation"], a, b, sub)
            (cdir / "comparison.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
            print(f"  ⇄ {cell} {pk} compared", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default=None, help="one cell name (smoke test)")
    ap.add_argument("--resume", default=None, metavar="RUN_DIR")
    ap.add_argument("--reads-only", action="store_true")
    ap.add_argument("--compare-only", default=None, metavar="RUN_DIR")
    args = ap.parse_args()

    groups = load_groups()
    stories = all_stories()
    if args.pilot:
        stories = [s for s in stories if s.startswith(args.pilot)] or [args.pilot]

    if args.compare_only:
        out_dir = Path(args.compare_only)
        asyncio.run(run_comparisons(out_dir, stories)); return

    out_dir = Path(args.resume) if args.resume else (HERE / f"run_{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stories").mkdir(exist_ok=True)
    for cell in stories:
        cd = out_dir / "stories" / cell; cd.mkdir(exist_ok=True)
        (cd / "story.txt").write_text(load_story(cell))
    (out_dir / "meta.json").write_text(json.dumps(
        {"stories": stories, "groups": groups, "seed": SEED, "started": time.time()}, indent=2))

    print(f"Run dir: {out_dir}\nStories: {len(stories)}  Agents/group: 8  Total reads: {len(stories)*16}")
    asyncio.run(run_reads(out_dir, stories, groups, resume=bool(args.resume)))
    if not args.reads_only:
        asyncio.run(run_comparisons(out_dir, stories))
    print(f"Done. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
