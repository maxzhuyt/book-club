"""CD proof-of-concept runner.

Design: 3 stories x 3 probes x 2 groups x 4 personas x 2 alphas, two-pass
read pipeline identical to experiments_v3 (pass-1 generic notice question,
pass-2 probe elicitation), generated locally with Qwen3-32B (bf16) under
dual-copy contrastive decoding (one model copy per GPU). The grader is the
DeepSeek API (see code_responses.py).

  alpha = 0.0  -> baseline (plain persona prompting, same code path)
  alpha = 1.0  -> CD against the generic-reader negative prompt

Probes: P2 Knowledge-gap (dead in v3: A-B gap +0.014, p=1.0),
        P4 Convention   (collapsed in v3: +0.361, p=0.166),
        P5 Salience     (works in v3: +0.769, p<0.001 -> positive control).

Personas: 4 per group, verbosity-matched pairs across groups.

Seeds are stable per (uid, cell, probe) and SHARED across alphas, so the two
conditions are paired: same seed, only the contrast term differs.

Layout mirrors v3:
  <out>/alpha{0|1}/results/{A|B}/{cell}/{probe}/agent-{uid}/pass1.txt,
  pass2.txt, meta.json
Resumable: a cell/probe/agent/alpha is skipped when its pass2.txt exists.

Run smoke gate only:  python run_cd_poc.py --smoke-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

V3_DIR = Path("/project/jevans/maxzhuyt/book-club-v3/book-club-materials/experiments_v3")
sys.path.insert(0, str(V3_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probes import PROBES, GENERIC_PASS1            # noqa: E402  (v3 import)
from neg_prompt import build_negative_system        # noqa: E402
from cd_decoder import ContrastiveDecoder, GenerationConfig, load_dual_qwen  # noqa: E402

STORIES_DIR = V3_DIR / "run_20260531-022438" / "stories"
PERSONAS_DIR = V3_DIR / "personas_v3"

CELLS = [
    "cell-01-recent-sp-pure",
    "cell-06-middle-sp-fantastical",
    "cell-09-distant-sp-pure",
]
PROBE_IDS = ["P2", "P4", "P5"]
PERSONAS = {
    # verbosity-matched pairs (avg review words): 71/82, 134/135, 260/269, 488/489
    "A": ["441197", "2260345", "22227336", "632247"],
    "B": ["181538130", "430758", "152291541", "35794399"],
}
ALPHAS = [0.0, 1.0]

CFG = GenerationConfig(max_new_tokens=1200, temperature=0.75, top_p=0.95)


def stable_seed(*parts: str) -> int:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def build_pass1_messages(persona_prompt: str, probe: dict, story: str,
                         negative: bool = False) -> list[dict]:
    if negative:
        system = build_negative_system(probe["prime"])
    else:
        system = persona_prompt.rstrip() + "\n\n--- ATTENTIONAL FOCUS ---\n" + probe["prime"]
    user1 = f"=== PASSAGE ===\n{story}\n=== END PASSAGE ===\n\n{GENERIC_PASS1}"
    return [{"role": "system", "content": system},
            {"role": "user", "content": user1}]


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# A small set of very common English function/content words. Coherent reading
# responses are dense in these; FP8 token-salad (code/JSON fragments, hex, random
# identifiers) is not. We require a minimum coverage fraction to pass the gate.
_COMMON_WORDS = set("""the a an and or but if then of to in on at for with as by from into about
over under again further is are was were be been being have has had do does did i you he she it we they
this that these those there here what which who when where why how not no yes can could would should will
my your his her its our their me him them us reader story passage scene period historical history feel felt
think thought notice noticed read reading writing prose moment detail sense character world seems seemed
because while where which would about more most some very much really quite first second part attention""".split())


def english_word_fraction(text: str) -> float:
    import re
    toks = re.findall(r"[A-Za-z]+", text.lower())
    if len(toks) < 10:
        return 0.0
    hits = sum(1 for t in toks if t in _COMMON_WORDS)
    return hits / len(toks)


def looks_coherent(text: str) -> tuple[bool, dict]:
    """Distinguish real English prose from FP8 token-salad.

    Token-salad ('0x1.0.msg_merged', '#endregion', '$this->get') has very low
    common-word density, high symbol density, and odd alpha ratios. Real reading
    responses are dense in common words and mostly alphabetic.
    """
    import re
    n = len(text)
    if n < 80:
        return False, {"reason": "too short", "len": n}
    alpha = sum(c.isalpha() or c.isspace() for c in text) / n
    common = english_word_fraction(text)
    # symbols common in code salad
    symbol_density = sum(text.count(c) for c in "{}<>=*/\\|@#$_`") / n
    ok = (common >= 0.30 and alpha >= 0.75 and symbol_density <= 0.03)
    return ok, {"common_word_frac": round(common, 3),
                "alpha_space_frac": round(alpha, 3),
                "symbol_density": round(symbol_density, 4), "len": n}


def run_smoke(dec: ContrastiveDecoder) -> bool:
    """Quick validation: generation is COHERENT (not FP8 salad), CD steering
    is visible, and throughput is sane. alpha=0 coherence is the hard gate —
    if plain generation is garbage, the FP8 serving is broken."""
    story = (STORIES_DIR / CELLS[0] / "story.txt").read_text()[:4000]
    probe = PROBES["P5"]
    persona = (PERSONAS_DIR / "441197.txt").read_text()  # romance-leaning HF reader
    cfg = GenerationConfig(max_new_tokens=200, temperature=0.75, top_p=0.95, seed=1234)

    m_pos = build_pass1_messages(persona, probe, story)
    m_neg = build_pass1_messages(persona, probe, story, negative=True)

    log("--- SMOKE alpha=0 (plain generation; tests FP8 correctness) ---")
    r0 = dec.generate(m_pos, None, cfg, alpha=0.0)
    coh0, m0 = looks_coherent(r0["text"])
    log(f"alpha=0: {r0['n_tokens']} tok @ {r0['tok_per_s']} tok/s | coherent={coh0} {m0}")
    log(r0["text"][:500])

    log("--- SMOKE alpha=1 (CD) ---")
    r1 = dec.generate(m_pos, m_neg, cfg, alpha=1.0)
    coh1, m1 = looks_coherent(r1["text"])
    log(f"alpha=1: {r1['n_tokens']} tok @ {r1['tok_per_s']} tok/s | coherent={coh1} {m1}")
    log(r1["text"][:500])

    # Gate 1: alpha=0 must be COHERENT (else serving is broken).
    # Gate 2: throughput must be sane (catches device-placement regressions).
    # Gate 3: CD must be ACTIVE — alpha=1 output must differ from alpha=0 at the
    #         same seed, proving the negative branch is actually subtracted.
    cd_active = r1["text"] != r0["text"]
    fast_enough = r0["tok_per_s"] >= 8.0
    ok = coh0 and r0["n_tokens"] > 20 and cd_active and fast_enough
    if not coh0:
        log("SERVING BROKEN: alpha=0 plain generation is incoherent token-salad.")
    if not cd_active:
        log("CD INACTIVE: alpha=1 == alpha=0; negative branch not being subtracted.")
    if not fast_enough:
        log(f"TOO SLOW: {r0['tok_per_s']} tok/s (<8); check device placement.")
    if coh0 and not coh1:
        log("NOTE: alpha=1 incoherent — CD alpha=1.0 may be too strong; consider tuning.")
    log(f"smoke ok={ok} (coherent={coh0} cd_active={cd_active} fast={fast_enough})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str,
                    default="/project/jevans/maxzhuyt/models/Qwen3-32B")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "run_cd_poc")
    ap.add_argument("--device-pos", type=str, default="cuda:0")
    ap.add_argument("--device-neg", type=str, default="cuda:1")
    # Cell-sharding for running two instances on two GPU pairs in parallel.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()

    log(f"loading {args.model_path} on {args.device_pos}/{args.device_neg} ...")
    t0 = time.time()
    dec = load_dual_qwen(args.model_path, args.device_pos, args.device_neg)
    log(f"models loaded in {time.time() - t0:.0f}s")

    if not run_smoke(dec):
        log("SMOKE FAILED — aborting before the full run")
        sys.exit(1)
    if args.smoke_only:
        log("smoke-only mode, exiting 0")
        return

    personas = {g: {u: (PERSONAS_DIR / f"{u}.txt").read_text() for u in us}
                for g, us in PERSONAS.items()}
    cells = [c for i, c in enumerate(CELLS) if i % args.nshards == args.shard]
    log(f"shard {args.shard}/{args.nshards}: cells {cells}")
    stories = {c: (STORIES_DIR / c / "story.txt").read_text() for c in cells}

    total = len(cells) * len(PROBE_IDS) * sum(len(v) for v in PERSONAS.values()) * len(ALPHAS)
    done = 0
    for cell in cells:
        for pid in PROBE_IDS:
            probe = PROBES[pid]
            for alpha in ALPHAS:
                atag = f"alpha{int(alpha)}" if alpha == int(alpha) else f"alpha{alpha}"
                for group, uids in PERSONAS.items():
                    for uid in uids:
                        done += 1
                        outd = (args.out_dir / atag / "results" / group
                                / cell / pid / f"agent-{uid}")
                        if (outd / "pass2.txt").exists():
                            continue
                        outd.mkdir(parents=True, exist_ok=True)
                        seed = stable_seed(uid, cell, pid)
                        cfg = GenerationConfig(
                            max_new_tokens=CFG.max_new_tokens,
                            temperature=CFG.temperature,
                            top_p=CFG.top_p, seed=seed)

                        m1_pos = build_pass1_messages(personas[group][uid], probe,
                                                      stories[cell])
                        m1_neg = (build_pass1_messages(personas[group][uid], probe,
                                                       stories[cell], negative=True)
                                  if alpha != 0 else None)
                        r1 = dec.generate(m1_pos, m1_neg, cfg, alpha=alpha)
                        (outd / "pass1.txt").write_text(r1["text"])

                        m2_pos = m1_pos + [
                            {"role": "assistant", "content": r1["text"]},
                            {"role": "user", "content": probe["elicitation"]},
                        ]
                        m2_neg = None
                        if alpha != 0:
                            m2_neg = m1_neg + [
                                {"role": "assistant", "content": r1["text"]},
                                {"role": "user", "content": probe["elicitation"]},
                            ]
                        cfg.seed = seed + 1
                        r2 = dec.generate(m2_pos, m2_neg, cfg, alpha=alpha)
                        (outd / "pass2.txt").write_text(r2["text"])

                        meta = {
                            "group": group, "cell": cell, "probe": pid,
                            "user_id": uid, "alpha": alpha, "seed": seed,
                            "temperature": cfg.temperature, "top_p": cfg.top_p,
                            "max_new_tokens": cfg.max_new_tokens,
                            "model": Path(args.model_path).name,
                            "pass1": {k: r1[k] for k in
                                      ("n_tokens", "finish", "had_think",
                                       "seconds", "tok_per_s")},
                            "pass2": {k: r2[k] for k in
                                      ("n_tokens", "finish", "had_think",
                                       "seconds", "tok_per_s")},
                        }
                        (outd / "meta.json").write_text(json.dumps(meta, indent=1))
                        log(f"[{done}/{total}] {atag} {group}/{cell}/{pid}/{uid} "
                            f"p1={r1['n_tokens']}t p2={r2['n_tokens']}t "
                            f"@{r2['tok_per_s']}tok/s")

    log("ALL DONE")


if __name__ == "__main__":
    main()
