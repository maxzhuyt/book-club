"""Close gaps in V3 reader-response coding.

(1) Re-code the 78 pass1/pass2 parse-error stubs (where coding.json[phase] has
    '_parse_error') using the existing pass{1,2}.txt content.
(2) Fill the 5 missing (group, story, probe, agent-N) slots by coding both
    passes if pass1.txt and pass2.txt exist.

Backs up the original coding.json files to coding.json.bak before overwriting
so the original run state is recoverable.
"""
from __future__ import annotations
import argparse, asyncio, json, shutil, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import coding  # noqa: E402

RUN_DIR = HERE / "run_20260531-022438"
RESULTS = RUN_DIR / "results"
GROUPS = ("A", "B")
PROBES = ("P1", "P2", "P3", "P4", "P5")
AGENTS = ("agent-0", "agent-1")
PHASES = ("pass1", "pass2")


def find_stubs() -> list[tuple[Path, str]]:
    """Return list of (coding_json_path, phase) for stubs that need re-coding."""
    out = []
    for cj in RESULTS.rglob("coding.json"):
        try:
            obj = json.loads(cj.read_text())
        except Exception:
            continue
        for ph in PHASES:
            if "_parse_error" in obj.get(ph, {}):
                out.append((cj, ph))
    return out


def find_missing_slots() -> list[Path]:
    """Return agent dirs (group/story/probe/agent-N) where coding.json is absent
    but pass1.txt OR pass2.txt is present."""
    out = []
    meta = json.loads((RUN_DIR / "meta.json").read_text())
    for g in GROUPS:
        for sid in meta["stories"]:
            for pk in PROBES:
                for ag in AGENTS:
                    adir = RESULTS / g / sid / pk / ag
                    cj = adir / "coding.json"
                    if cj.exists():
                        continue
                    if (adir / "pass1.txt").exists() or (adir / "pass2.txt").exists():
                        out.append(adir)
    return out


async def code_phase(text: str) -> dict:
    return await coding.code_response(text)


async def recode_stub(cj_path: Path, phase: str, sem: asyncio.Semaphore):
    async with sem:
        text_p = cj_path.parent / f"{phase}.txt"
        if not text_p.exists():
            return {"path": str(cj_path), "phase": phase, "status": "no_text"}
        text = text_p.read_text().strip()
        if not text:
            return {"path": str(cj_path), "phase": phase, "status": "empty_text"}
        try:
            new_code = await code_phase(text)
        except Exception as e:
            return {"path": str(cj_path), "phase": phase,
                    "status": "error", "err": str(e)}
        # back up + overwrite
        try:
            obj = json.loads(cj_path.read_text())
        except Exception:
            obj = {}
        bak = cj_path.with_suffix(".json.bak")
        if not bak.exists():
            shutil.copyfile(cj_path, bak)
        obj[phase] = new_code
        cj_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
        return {"path": str(cj_path), "phase": phase,
                "status": "ok",
                "has_parse_error": "_parse_error" in new_code}


async def fill_missing(agent_dir: Path, sem: asyncio.Semaphore):
    """Code both passes for a missing slot and write a fresh coding.json."""
    async with sem:
        out = {}
        for ph in PHASES:
            text_p = agent_dir / f"{ph}.txt"
            if not text_p.exists():
                continue
            text = text_p.read_text().strip()
            if not text:
                continue
            try:
                out[ph] = await code_phase(text)
            except Exception as e:
                out[ph] = {"_parse_error": str(e), "_raw": text[:200]}
        if not out:
            return {"path": str(agent_dir), "status": "no_text"}
        cj = agent_dir / "coding.json"
        cj.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        return {"path": str(agent_dir), "status": "ok",
                "phases_filled": list(out.keys()),
                "has_parse_error_any": any("_parse_error" in v for v in out.values())}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-concurrent", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=("stubs", "missing", "both"), default="both")
    args = ap.parse_args()

    stubs = find_stubs() if args.only in ("stubs", "both") else []
    missing = find_missing_slots() if args.only in ("missing", "both") else []
    print(f"stubs to re-code: {len(stubs)}")
    print(f"missing slots to fill: {len(missing)}")
    if missing:
        for m in missing:
            print(f"  missing: {m.relative_to(RESULTS)}")
    if args.dry_run:
        return

    sem = asyncio.Semaphore(args.max_concurrent)
    t0 = time.time()
    stub_results, miss_results = await asyncio.gather(
        asyncio.gather(*[recode_stub(p, ph, sem) for p, ph in stubs]),
        asyncio.gather(*[fill_missing(d, sem) for d in missing]),
    )
    dt = time.time() - t0
    ok_stubs = [r for r in stub_results if r.get("status") == "ok"]
    parse_err_stubs = [r for r in ok_stubs if r.get("has_parse_error")]
    ok_miss = [r for r in miss_results if r.get("status") == "ok"]
    parse_err_miss = [r for r in ok_miss if r.get("has_parse_error_any")]

    print(f"\ndone in {dt:.1f}s")
    print(f"  stubs ok: {len(ok_stubs)}/{len(stub_results)}  "
          f"(still parse-error after retry: {len(parse_err_stubs)})")
    print(f"  missing filled: {len(ok_miss)}/{len(miss_results)}  "
          f"(parse-error any phase: {len(parse_err_miss)})")
    if parse_err_stubs:
        print("\nstill parse-error stubs:")
        for r in parse_err_stubs[:10]:
            print(f"  {r['path']}  phase={r['phase']}")
    if parse_err_miss:
        print("\nstill parse-error missing-slots:")
        for r in parse_err_miss[:10]:
            print(f"  {r['path']}")


if __name__ == "__main__":
    asyncio.run(main())
