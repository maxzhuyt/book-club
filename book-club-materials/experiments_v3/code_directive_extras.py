"""C2-extras: two RELATIONAL codings of each directive against the reader notes
it was built from. Unlike code_directives.py (which scores the directive in
isolation), this coder is given BOTH the readers the consolidator saw AND the
directive it produced, and assesses:

  (A) Grounding / invention  — a validity check on editorial commitment.
        n_moves      distinct actionable moves in the directive
        n_grounded   of those, how many trace to >=1 reader note
        n_invented   = n_moves - n_grounded (consolidator-introduced, no reader basis)
        invention_rate = n_invented / n_moves
      Pairs with the rubric-leak check: rubric-leak asks whether SELECTIVE borrows
      the JUDGE's vocabulary; invention asks whether SELECTIVE fabricates moves
      readers never raised. If SELECTIVE's invention_rate ~ NEUTRAL's, its
      commitment is grounded in real reader signal rather than editorial fiat.

  (B) Craft camp  — which craft priority the directive sides with when it commits.
        camp in {period, generic, balanced, none}
          period   historical/period accuracy, period voice, material grounding (A-style)
          generic  pacing, tension, structure, genre beats, readability      (B-style)
          balanced deliberately balances both
          none     takes no craft-priority side (pure relay/aggregation)
        camp_strength 0-5
      Addresses the open §4.4 question (B-group readers produce stronger SELECTIVE
      revisions, mechanism unknown): does feeding B-only readers push the directive
      to side 'generic', and SELECTIVE more than NEUTRAL?

Both are produced in ONE call per directive (the directive + readers are in context
either way). Reuses code_directives.enumerate_directives() so it covers the same
1,728 directives, including the per-probe SELECTIVE-BLIND sweep.

Outputs cached under <run>/directive_extras_codings/<scope>/<sid>/...
plus a flat manifest at <run>/directive_extras_codings/manifest.csv.
"""
from __future__ import annotations
import argparse, asyncio, csv, json, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import client_v2          # noqa: E402
import code_directives as cd  # noqa: E402  (reuse enumerate_directives + RUN/PROBES)

RUN = cd.RUN
OUT_BASE = RUN / "directive_extras_codings"

VALID_CAMP = {"period", "generic", "balanced", "none"}


# ============================ READER LOADING ============================

def _read_pass2(sid: str, probe: str, group: str, slot: int) -> str | None:
    f = RUN / "results" / group / sid / probe / f"agent-{slot}" / "pass2.txt"
    return f.read_text().strip() if f.exists() else None


def load_pooled_readers(sid: str) -> list[dict]:
    out = []
    for pk in cd.PROBES:
        for g in ("A", "B"):
            for slot in (0, 1):
                t = _read_pass2(sid, pk, g, slot)
                if t is not None:
                    out.append({"probe": pk, "group": g, "slot": slot, "text": t})
    return out


def load_arm_readers(sid: str, probe: str, arm: str) -> list[dict]:
    by = {"A": [], "B": []}
    for g in ("A", "B"):
        for slot in (0, 1):
            t = _read_pass2(sid, probe, g, slot)
            if t is not None:
                by[g].append({"probe": probe, "group": g, "slot": slot, "text": t})
    return {"A": by["A"], "B": by["B"], "AB": by["A"] + by["B"]}[arm]


def readers_for(item: dict) -> list[dict]:
    if item["scope"] == "pooled":
        return load_pooled_readers(item["story_id"])
    return load_arm_readers(item["story_id"], item["probe"], item["arm"])


def reader_block(readers: list[dict]) -> str:
    if not readers:
        return "(no reader notes available)"
    return "\n\n".join(
        f"[Group {it['group']} agent {it['slot']} — Probe {it['probe']}]\n{it['text']}"
        for it in readers)


# ============================ CODER ============================

EXTRAS_SYSTEM = (
    "You are a careful auditor of editorial directives. A consolidator read several "
    "reader responses to a short historical-fiction draft and produced a structured "
    "directive telling a downstream writer what to change. You are given (1) the reader "
    "notes the consolidator saw and (2) the directive it produced. Perform two "
    "assessments. Output ONLY one JSON object, no prose, no fences."
)

DIMENSIONS_TEXT = """\
=== TASK A: grounding / invention ===
A directive 'move' is a distinct, actionable change the writer would execute (count
across takeaways, overarching, inlineEdits, or whatever the directive uses;
de-duplicate near-identical moves).
- n_moves (integer): total distinct actionable moves in the directive.
- n_grounded (integer): of those moves, how many are SUPPORTED by at least one reader
  note — i.e. some reader actually raised the underlying observation or concern, even
  loosely or in different words. A move with no reader basis is INVENTED by the
  consolidator (do not count it as grounded). 0 <= n_grounded <= n_moves.
- examples_invented (list of up to 2 short strings): brief paraphrases of invented
  moves (moves with no reader support). Empty list if none.

=== TASK B: craft camp ===
When the directive commits to a direction, which CRAFT PRIORITY does it side with?
- camp (label), one of:
    "period"   — prioritizes historical/period accuracy, period voice, material or
                 factual grounding, how the past is rendered.
    "generic"  — prioritizes generic narrative craft: pacing, tension, structure,
                 genre beats, clarity/readability.
    "balanced" — deliberately balances period and generic priorities.
    "none"     — takes no craft-priority side (pure relay or aggregation, no commitment).
- camp_strength (0-5): how strongly it leans toward the chosen camp. 0 if camp="none",
  5 = emphatic single-minded lean.
- note (str): one short clause of evidence (max 20 words).

Output exactly: {"n_moves": int, "n_grounded": int, "examples_invented": [str],
  "camp": "period"|"generic"|"balanced"|"none", "camp_strength": int, "note": str}"""


def parse_extras(text: str) -> dict:
    t = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return {"_parse_error": "no json", "_raw": text[:300]}
    raw = m.group()
    obj = None
    for fixup in (lambda s: s,
                  lambda s: re.sub(r",(\s*[\]}])", r"\1", s)):
        try:
            obj = json.loads(fixup(raw))
            break
        except json.JSONDecodeError:
            obj = None
    if obj is None:
        return {"_parse_error": "json parse failed", "_raw": text[:300]}

    def _int(v, d=0):
        try:
            return int(v)
        except (TypeError, ValueError):
            return d

    n_moves = max(0, _int(obj.get("n_moves"), 0))
    n_grounded = max(0, min(_int(obj.get("n_grounded"), 0), n_moves))
    n_invented = n_moves - n_grounded
    rate = round(n_invented / n_moves, 3) if n_moves else 0.0
    camp = obj.get("camp", "none")
    if camp not in VALID_CAMP:
        camp = "none"
    strength = max(0, min(_int(obj.get("camp_strength"), 0), 5))
    if camp == "none":
        strength = 0
    ex = obj.get("examples_invented", [])
    if not isinstance(ex, list):
        ex = []
    ex = [str(x)[:160] for x in ex][:2]
    return {"n_moves": n_moves, "n_grounded": n_grounded, "n_invented": n_invented,
            "invention_rate": rate, "camp": camp, "camp_strength": strength,
            "examples_invented": ex, "note": str(obj.get("note", ""))[:200]}


async def code_extras(directive_json: str, readers: list[dict]) -> dict:
    user = (DIMENSIONS_TEXT
            + "\n\n=== READER NOTES (what the consolidator saw) ===\n"
            + reader_block(readers)
            + "\n\n=== DIRECTIVE JSON (what it produced) ===\n"
            + directive_json
            + "\n=== END ===")
    raw = await client_v2.chat(
        [{"role": "system", "content": EXTRAS_SYSTEM},
         {"role": "user", "content": user}],
        max_tokens=500, temperature=0.2)
    return parse_extras(raw)


# ============================ CODING LOOP ============================

def enumerate_items() -> list[dict]:
    """Reuse code_directives' enumeration, re-rooting dst under directive_extras_codings."""
    items = cd.enumerate_directives()
    for it in items:
        rel = it["dst"].relative_to(cd.OUT_BASE)
        it["dst"] = OUT_BASE / rel
    return items


async def code_one(item: dict, sem: asyncio.Semaphore, retry: int = 2) -> dict:
    if item["dst"].exists():
        try:
            obj = json.loads(item["dst"].read_text())
            if "_parse_error" not in obj:
                return {**item, "status": "cached"}
        except Exception:
            pass
    async with sem:
        try:
            directive_json = item["src"].read_text()
        except Exception as e:
            return {**item, "status": "read_error", "err": str(e)}
        readers = readers_for(item)
        last = None
        for attempt in range(retry + 1):
            try:
                code = await code_extras(directive_json, readers)
                if "_parse_error" not in code:
                    item["dst"].parent.mkdir(parents=True, exist_ok=True)
                    item["dst"].write_text(json.dumps(code, indent=2, ensure_ascii=False))
                    return {**item, "status": "ok", "attempt": attempt + 1}
                last = code
            except Exception as e:
                last = {"_parse_error": str(e)}
        item["dst"].parent.mkdir(parents=True, exist_ok=True)
        item["dst"].write_text(json.dumps(last, indent=2))
        return {**item, "status": "parse_error", "err": last.get("_parse_error", "")}


FIELDS = ("n_moves", "n_grounded", "n_invented", "invention_rate",
          "camp", "camp_strength", "note")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-concurrent", type=int, default=8)
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--scope", choices=("pooled", "per_probe", "both"), default="both")
    args = ap.parse_args()

    items = enumerate_items()
    if args.scope != "both":
        items = [it for it in items if it["scope"] == args.scope]
    print(f"directive-extras coder over {len(items)} directives "
          f"(concurrency={args.max_concurrent})")
    if args.pilot:
        items = items[:args.pilot]
        print(f"pilot: running first {args.pilot}")

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.max_concurrent)
    t0 = time.time()
    results = []
    BATCH = 60
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        results.extend(await asyncio.gather(*[code_one(it, sem) for it in batch]))
        print(f"  ... {min(i + BATCH, len(items))}/{len(items)}   "
              f"ok={sum(1 for r in results if r['status'] in ('ok', 'cached'))} "
              f"err={sum(1 for r in results if r['status'] == 'parse_error')} "
              f"elapsed={time.time() - t0:.0f}s", flush=True)

    rows = []
    for r in results:
        rec = {"scope": r["scope"], "story_id": r["story_id"], "config": r["config"],
               "arm": r["arm"], "probe": r["probe"], "variant": r["variant"],
               "status": r["status"], "dst": str(r["dst"].relative_to(RUN))}
        if r["status"] in ("ok", "cached"):
            try:
                c = json.loads(r["dst"].read_text())
                for k in FIELDS:
                    rec[k] = c.get(k, "")
            except Exception:
                pass
        rows.append(rec)

    header = ["scope", "story_id", "config", "arm", "probe", "variant", "status",
              *FIELDS, "dst"]
    with (OUT_BASE / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in header})

    from collections import Counter
    print(f"\nfinished in {time.time() - t0:.0f}s")
    print("status:", dict(Counter(r["status"] for r in results)))
    print(f"manifest: {OUT_BASE / 'manifest.csv'}")


if __name__ == "__main__":
    asyncio.run(main())
