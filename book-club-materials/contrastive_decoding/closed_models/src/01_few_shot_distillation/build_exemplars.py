"""
Plan 01 — build the CD exemplar bank.

Runs the existing 4-phase CD simulation (simulation/src/bookclub/simulate.py)
on a HELD-OUT story for K different seeds, then reorganizes the resulting
transcripts into a per-persona, per-phase bank that the closed-model runner
will read from.

The CD code in simulation/src/cd is used as-is — we call run_simulation()
directly and post-process its returned dict.

Output bank layout:

    bank/
    ├── manifest.json
    └── <short_id>/
        ├── phase1.json     # [{"seed": ..., "text": "...", "n_chars": ...}, ...]
        ├── phase2.json
        ├── phase3_r1.json
        ├── phase3_r2.json
        └── phase4.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "simulation" / "src"))

from bookclub.simulate import run_simulation  # noqa: E402
from bookclub.cast import load_cast, speaker_tag  # noqa: E402
from cd import prompts as prompt_lib  # noqa: E402
from cd.alpha import compute_alpha  # noqa: E402


def harvest(sim_out: dict, seed: int) -> dict[str, dict[str, dict]]:
    """
    Reorganize a run_simulation() return value into per-persona/per-phase
    entries for a single seed.

    Returns: { short_id: { "phase1": {seed, text}, "phase2": ..., ... } }
    """
    cast_entries = sim_out["cast"]
    out: dict[str, dict[str, dict]] = {e["short_id"]: {} for e in cast_entries}

    for sid, text in sim_out["private_reviews"].items():
        out[sid]["phase1"] = {"seed": seed, "text": text}
    for sid, text in sim_out["phase2_reactions"].items():
        out[sid]["phase2"] = {"seed": seed, "text": text}

    # phase3_rounds is a list (one entry per round) of lists of
    # {speaker, text} dicts. Speaker tags include the persona's short_id
    # at the front, e.g. "Reader-A (The Emotional Reader) [Phase 3 R1]".
    for r_idx, round_entries in enumerate(sim_out["phase3_rounds"], start=1):
        key = f"phase3_r{r_idx}"
        for entry in round_entries:
            speaker = entry["speaker"]
            sid = speaker.split(" ", 1)[0]
            if sid in out:
                out[sid][key] = {"seed": seed, "text": entry["text"]}

    for sid, text in sim_out["phase4_reflections"].items():
        out[sid]["phase4"] = {"seed": seed, "text": text}
    return out


def merge(bank: dict[str, dict[str, list[dict]]],
          one_seed: dict[str, dict[str, dict]]) -> None:
    """Accumulate per-seed harvests into the persistent bank dict."""
    for sid, phase_map in one_seed.items():
        bank.setdefault(sid, {})
        for phase, entry in phase_map.items():
            bank[sid].setdefault(phase, []).append(entry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--personas-dir", type=Path,
        default=REPO_ROOT / "personas",
        help="Persona system-prompt directory (has cast_summary.json).",
    )
    ap.add_argument(
        "--story", type=str, required=True,
        help="Filename inside personas-dir of the HELD-OUT story to use for "
             "exemplar generation. MUST be different from the story used in "
             "the eval simulation (otherwise exemplars contaminate eval).",
    )
    ap.add_argument(
        "--story-title", type=str, required=True,
        help="Title to pass into the user message (display label only).",
    )
    ap.add_argument(
        "--bank-dir", type=Path,
        default=Path(__file__).resolve().parent / "bank",
        help="Where to write the exemplar bank.",
    )
    ap.add_argument(
        "--workdir", type=Path,
        default=Path(__file__).resolve().parent / "_seed_runs",
        help="Scratch directory for per-seed run_simulation outputs "
             "(written by run_simulation; we keep them for reproducibility).",
    )
    ap.add_argument(
        "--cd-model-id", type=str, default="Qwen/Qwen2.5-14B-Instruct",
        help="Open model used for CD (positive and negative branches).",
    )
    ap.add_argument("--num-seeds", type=int, default=3,
                    help="Number of CD-generated samples per persona/phase.")
    ap.add_argument("--base-seed", type=int, default=20260515)
    ap.add_argument("--rounds", type=int, default=2,
                    help="Phase 3 rounds (controls phase3_r1, phase3_r2).")
    ap.add_argument("--no-child", action="store_true",
                    help="Omit the 6-year-old. Recommended for adult-only "
                         "closed-model runs (see closed_models/00_meta.md).")
    args = ap.parse_args()

    args.bank_dir.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)

    story_path = args.personas_dir / args.story
    if not story_path.exists():
        raise SystemExit(f"Story file not found: {story_path}")

    print(f"[plan01] story = {story_path.name}  seeds = {args.num_seeds}",
          flush=True)
    print(f"[plan01] bank  = {args.bank_dir}", flush=True)

    bank: dict[str, dict[str, list[dict]]] = {}
    cast_snapshot: list[dict] = []

    for k in range(args.num_seeds):
        seed = args.base_seed + k
        out_dir = args.workdir / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[plan01] === seed {seed} (run {k+1}/{args.num_seeds}) ===",
              flush=True)
        t0 = time.time()
        sim_out = run_simulation(
            personas_dir=args.personas_dir,
            output_dir=out_dir,
            model_id=args.cd_model_id,
            story_path=story_path,
            story_title=args.story_title,
            seed=seed,
            n_discussion_rounds=args.rounds,
            include_six_year_old=not args.no_child,
        )
        print(f"[plan01] seed {seed} done in {time.time()-t0:.1f}s",
              flush=True)
        if not cast_snapshot:
            cast_snapshot = sim_out["cast"]
        merge(bank, harvest(sim_out, seed))

    # Write per-persona / per-phase json blobs.
    for sid, phase_map in bank.items():
        persona_dir = args.bank_dir / sid
        persona_dir.mkdir(exist_ok=True)
        for phase, entries in phase_map.items():
            entries_sorted = sorted(entries, key=lambda e: e["seed"])
            (persona_dir / f"{phase}.json").write_text(
                json.dumps(entries_sorted, indent=2, ensure_ascii=False)
            )

    # Manifest
    cast = load_cast(args.personas_dir)
    alpha_table = {
        p.short_id: {
            "phase1": compute_alpha(p.summary(), 1),
            "phase3": compute_alpha(p.summary(), 3),
            "phase4": compute_alpha(p.summary(), 4),
        }
        for p in cast
    }
    manifest = {
        "plan": "01_few_shot_distillation",
        "cd_model_id": args.cd_model_id,
        "story": args.story,
        "story_title": args.story_title,
        "num_seeds": args.num_seeds,
        "base_seed": args.base_seed,
        "rounds": args.rounds,
        "include_six_year_old": not args.no_child,
        "cast": cast_snapshot,
        "alphas": alpha_table,
        "phases_present": ["phase1", "phase2"]
                          + [f"phase3_r{r+1}" for r in range(args.rounds)]
                          + ["phase4"],
    }
    (args.bank_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(f"\n[plan01] bank manifest written to "
          f"{args.bank_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
