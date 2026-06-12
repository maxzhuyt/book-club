"""C2: code the consolidator directives produced by V3.

Sweeps to cover:
  pooled (writer_judge_v3)             neutral, selective       -> 72 directives
  pooled-blind (writer_judge_v3_blind) selective_blind          -> 36 directives
  per-probe (revise_judge_byprobe_v3)  {A,B,AB} x {neutral,selective} per probe
                                                                -> 1080 directives
  per-probe-blind (revise_byprobe_blind_v3) {A,B,AB} x selective_blind per probe
                                                                ->  540 directives
Total: 1728  (was 1188 before the per-probe blind sweep was added)

The directives have heterogeneous schemas (NEUTRAL outputs are often wrapped under
'directive' or 'editorial_directive' keys; SELECTIVE/SELECTIVE_BLIND outputs follow
the canonical 8-key schema). The LLM-coder reads the raw JSON (any shape) and scores
on a fixed schema designed to surface DIFFERENTIATION across pooling strategy:

  editorial_commitment   (0-5)   commit to one direction vs. aggregate competing views
  specificity            (0-5)   concrete named edits vs. abstract advice
  takeaway_count         (int)   distinct actionable editorial moves
  reader_attribution     (0/1)   attributes moves to specific reader slots (A:0, B:1)
  probe_attribution      (0/1)   mentions probe names (Plausibility, Convention, ...)
  craft_vocab_count      (int)   CRAFT_GUIDE-aligned terms (period voice, anachronism,
                                 implication, compression, counterfactual, specificity,
                                 sentence-level, exposition)
  edit_emphasis          (label) cut | expand | voice_shift | structural | mixed | unclear
  conflict_acknowledged  (0/1)   explicitly names reader disagreements
  conflict_resolution    (label) commit | aggregate | punt | none
  note                   (str)   one short evidentiary clause

Outputs cached per-directive under <run>/directive_codings/<scope>/<story_id>/<config>.json
plus a flat manifest at <run>/directive_codings/manifest.csv.
"""
from __future__ import annotations
import argparse, asyncio, json, re, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import client_v2  # noqa: E402

RUN = HERE / "run_20260531-022438"
OUT_BASE = RUN / "directive_codings"
PROBES = ("P1", "P2", "P3", "P4", "P5")
ARMS = ("A", "B", "AB")
VARIANTS = ("neutral", "selective")                              # pooled scope
PERPROBE_VARIANTS = ("neutral", "selective", "selective_blind")  # per-probe scope


CODER_SYSTEM = (
    "You are a careful coder of editorial directives. A directive is a structured "
    "instruction produced by an LLM-based consolidator after reading multiple human or "
    "agent reader responses; it tells a downstream writer what to change about a "
    "historical-fiction draft. You score the DIRECTIVE itself, not the underlying "
    "story or the readers. You will be given the directive as raw JSON (its key "
    "structure varies — some directives nest content under a 'directive' or "
    "'editorial_directive' key; others use the full schema with axes_of_divergence, "
    "takeaways, overarching, inlineEdits, allSuggestions, reader_response_paragraph). "
    "Treat all of them uniformly: read whatever editorial content exists. Output ONLY "
    "one JSON object, no prose, no fences."
)

DIMENSIONS_TEXT = """\
Score this directive on the following schema. Use the FULL 0-5 range where applicable;
0 reserved for absent, 5 reserved for unambiguous strong cases.

- editorial_commitment (0-5): does the directive COMMIT to a single coherent direction,
  or aggregate competing views without choosing? 0 = pure aggregation/synthesis,
  3 = mild lean, 5 = explicit "take side X, refuse side Y".
- specificity (0-5): are the proposed edits concrete (named phrases, quoted lines,
  specific sentence-level operations) or abstract (general advice)? 0 = vague,
  5 = repeatedly names exact lines and exact substitutions.
- takeaway_count (integer): count distinct, actionable editorial moves the writer
  would execute. Count items under takeaways, overarching, inlineEdits, recommended_changes,
  suggested_revisions — whatever the directive uses. De-duplicate near-identical moves.
- reader_attribution (0 or 1): does the directive tag specific reader slots like
  "A:0", "B:1", or "agent-0"? 1 if yes, even once.
- probe_attribution (0 or 1): does it mention any of the probe names — Plausibility,
  Knowledge-gap, Stability, Convention, Salience, or refer to "P1"/"P2"/etc.? 1 if yes.
- craft_vocab_count (integer): count occurrences (across the whole directive content,
  case-insensitive substring match) of these CRAFT_GUIDE-aligned terms:
  "period voice", "anachronism", "implication", "exposition", "compression",
  "counterfactual", "specificity", "sentence-level", "earned" (in the sense of
  "earned compression" or "earned effect"). One count per term per occurrence.
- edit_emphasis (label): the primary type of revision the directive pushes. One of:
    "cut"         — mostly delete/compress/trim
    "expand"      — mostly add detail/context/material
    "voice_shift" — mostly change register, tone, perspective, prose style
    "structural"  — mostly reorder, restructure, change beat sequence
    "mixed"       — no single emphasis dominates
    "unclear"     — directive doesn't propose anything actionable.
- conflict_acknowledged (0 or 1): does it explicitly name a disagreement between
  readers (e.g. "readers disagree on X", "camp one wants Y while camp two wants Z")?
- conflict_resolution (label): if conflict_acknowledged is 1, which strategy:
    "commit"     — picks one side, refuses other
    "aggregate"  — keeps both sides somehow
    "punt"       — defers decision to writer/reader
    "none"       — no conflict acknowledged (or conflict_acknowledged=0).
- note (str): one short clause of evidence (max 20 words).

Output exactly: {"editorial_commitment": int, "specificity": int, "takeaway_count": int,
  "reader_attribution": 0|1, "probe_attribution": 0|1, "craft_vocab_count": int,
  "edit_emphasis": "cut"|"expand"|"voice_shift"|"structural"|"mixed"|"unclear",
  "conflict_acknowledged": 0|1, "conflict_resolution": "commit"|"aggregate"|"punt"|"none",
  "note": str}"""


VALID_EMPHASIS = {"cut", "expand", "voice_shift", "structural", "mixed", "unclear"}
VALID_RESOLUTION = {"commit", "aggregate", "punt", "none"}


def parse_coding(text: str) -> dict:
    t = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return {"_parse_error": "no json", "_raw": text[:300]}
    raw = m.group()
    for fixup in (lambda s: s,
                  lambda s: re.sub(r",(\s*[\]}])", r"\1", s),
                  lambda s: re.sub(r'([}\]"0-9truefalsn])(\s+)(["\{\[])', r"\1,\2\3",
                                  re.sub(r",(\s*[\]}])", r"\1", s))):
        try:
            obj = json.loads(fixup(raw))
            break
        except json.JSONDecodeError:
            obj = None
    if obj is None:
        return {"_parse_error": "json parse failed", "_raw": text[:300]}
    out = {}
    for k in ("editorial_commitment", "specificity", "takeaway_count",
             "reader_attribution", "probe_attribution", "craft_vocab_count"):
        try:
            out[k] = int(obj.get(k, 0))
        except (TypeError, ValueError):
            out[k] = 0
    out["edit_emphasis"] = obj.get("edit_emphasis", "unclear")
    if out["edit_emphasis"] not in VALID_EMPHASIS:
        out["edit_emphasis"] = "unclear"
    out["conflict_acknowledged"] = 1 if obj.get("conflict_acknowledged") in (1, True, "1", "yes") else 0
    out["conflict_resolution"] = obj.get("conflict_resolution", "none")
    if out["conflict_resolution"] not in VALID_RESOLUTION:
        out["conflict_resolution"] = "none"
    out["note"] = str(obj.get("note", ""))[:200]
    return out


async def code_directive(raw_json_text: str) -> dict:
    user = (DIMENSIONS_TEXT
            + "\n\n=== DIRECTIVE JSON ===\n"
            + raw_json_text
            + "\n=== END ===")
    raw = await client_v2.chat(
        [{"role": "system", "content": CODER_SYSTEM},
         {"role": "user", "content": user}],
        max_tokens=400, temperature=0.2)
    return parse_coding(raw)


# ============================ ENUMERATE DIRECTIVES ============================

def enumerate_directives() -> list[dict]:
    """Returns list of {scope, story_id, config, arm, probe, variant, src, dst}.

    src = path to the directive_*.json file produced by a sweep
    dst = path under directive_codings/ where the coded output should go
    config = human-readable config name (matches manifest.csv)
    """
    out = []
    meta = json.loads((RUN / "meta.json").read_text())
    stories = meta["stories"]

    for sid in stories:
        # pooled writer_judge_v3 -- neutral, selective
        for variant in VARIANTS:
            src = RUN / "writer_judge_v3" / sid / f"directive_{variant}.json"
            if src.exists():
                out.append({
                    "scope": "pooled", "story_id": sid,
                    "config": f"pooled_{variant}",
                    "arm": "pool20", "probe": "all", "variant": variant,
                    "src": src,
                    "dst": OUT_BASE / "pooled" / sid / f"{variant}.json",
                })

        # pooled blind writer_judge_v3_blind
        src = RUN / "writer_judge_v3_blind" / sid / "directive_selective_blind.json"
        if src.exists():
            out.append({
                "scope": "pooled", "story_id": sid,
                "config": "pooled_blind",
                "arm": "pool20", "probe": "all", "variant": "selective_blind",
                "src": src,
                "dst": OUT_BASE / "pooled" / sid / "selective_blind.json",
            })

        # per-probe revise_judge_byprobe_v3 (neutral/selective) +
        # revise_byprobe_blind_v3 (selective_blind), all written into the same dir
        for probe in PROBES:
            for arm in ARMS:
                for variant in PERPROBE_VARIANTS:
                    src = (RUN / "revise_judge_byprobe_v3" / sid / probe
                           / f"directive_{arm}_{variant}.json")
                    if src.exists():
                        out.append({
                            "scope": "per_probe", "story_id": sid,
                            "config": f"{probe}_{arm}_{variant}",
                            "arm": arm, "probe": probe, "variant": variant,
                            "src": src,
                            "dst": OUT_BASE / "per_probe" / sid / probe
                                   / f"{arm}_{variant}.json",
                        })
    return out


# ============================ CODING LOOP ============================

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
            raw_text = item["src"].read_text()
        except Exception as e:
            return {**item, "status": "read_error", "err": str(e)}
        last = None
        for attempt in range(retry + 1):
            try:
                code = await code_directive(raw_text)
                if "_parse_error" not in code:
                    item["dst"].parent.mkdir(parents=True, exist_ok=True)
                    item["dst"].write_text(json.dumps(code, indent=2))
                    return {**item, "status": "ok", "attempt": attempt + 1}
                last = code
            except Exception as e:
                last = {"_parse_error": str(e)}
        # save the failed code so we don't retry indefinitely on the next run
        item["dst"].parent.mkdir(parents=True, exist_ok=True)
        item["dst"].write_text(json.dumps(last, indent=2))
        return {**item, "status": "parse_error", "err": last.get("_parse_error", "")}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-concurrent", type=int, default=8)
    ap.add_argument("--pilot", type=int, default=0,
                    help="if >0, run only first N items (for dry-run/cost check)")
    ap.add_argument("--scope", choices=("pooled", "per_probe", "both"), default="both")
    args = ap.parse_args()

    items = enumerate_directives()
    if args.scope != "both":
        items = [it for it in items if it["scope"] == args.scope]
    print(f"directive coder over {len(items)} directives "
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
        batch_results = await asyncio.gather(*[code_one(it, sem) for it in batch])
        results.extend(batch_results)
        print(f"  ... {min(i + BATCH, len(items))}/{len(items)}   "
              f"ok={sum(1 for r in results if r['status'] in ('ok', 'cached'))} "
              f"err={sum(1 for r in results if r['status'] == 'parse_error')} "
              f"elapsed={time.time() - t0:.0f}s", flush=True)

    # Manifest
    rows = []
    for r in results:
        rec = {"scope": r["scope"], "story_id": r["story_id"],
               "config": r["config"], "arm": r["arm"], "probe": r["probe"],
               "variant": r["variant"], "status": r["status"],
               "dst": str(r["dst"].relative_to(RUN))}
        if r["status"] in ("ok", "cached"):
            try:
                c = json.loads(r["dst"].read_text())
                for k in ("editorial_commitment", "specificity", "takeaway_count",
                          "reader_attribution", "probe_attribution",
                          "craft_vocab_count", "edit_emphasis",
                          "conflict_acknowledged", "conflict_resolution", "note"):
                    rec[k] = c.get(k, "")
            except Exception:
                pass
        rows.append(rec)

    import csv
    header = ["scope", "story_id", "config", "arm", "probe", "variant", "status",
              "editorial_commitment", "specificity", "takeaway_count",
              "reader_attribution", "probe_attribution", "craft_vocab_count",
              "edit_emphasis", "conflict_acknowledged", "conflict_resolution",
              "note", "dst"]
    with (OUT_BASE / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in header})

    print(f"\nfinished in {time.time() - t0:.0f}s")
    from collections import Counter
    print("status:", dict(Counter(r["status"] for r in results)))
    print(f"manifest: {OUT_BASE / 'manifest.csv'}")


if __name__ == "__main__":
    asyncio.run(main())
