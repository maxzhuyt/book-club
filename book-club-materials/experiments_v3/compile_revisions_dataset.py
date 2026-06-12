"""Compile all V3 revised stories into a single dataset mirroring the 36-story
canonical layout.

Output: <run>/revisions_dataset/<story_id>/<config>.md  (plus original.md)
        <run>/revisions_dataset/manifest.csv

Config naming:
  original                                          the source story.txt
  pooled_neutral, pooled_selective                  writer_judge_v3 (20 readers, 2 variants)
  pooled_blind                                      writer_judge_v3_blind (selective_blind)
  P{1..5}_{A,B,AB}_{neutral,selective}              revise_judge_byprobe_v3 (per-probe x arm)
  P{1..5}_{A,B,AB}_selective_blind                  revise_byprobe_blind_v3 (per-probe blind)

n_readers per config:
  pooled_*                  20  (5 probes x 2 groups x 2 slots)
  P{i}_A_*                   2  (A:0, A:1 within probe i)
  P{i}_B_*                   2  (B:0, B:1 within probe i)
  P{i}_AB_*                  4  (A:0, A:1, B:0, B:1 within probe i)

Per story: 1 original + 2 pooled + 1 pooled_blind + 5 probes x 3 arms x 3 variants
(neutral, selective, selective_blind) = 49 files.

Quality flags (column `flag`) are derived purely from word count on every compile
(WRITER_REFUSED if <200 words, TRUNCATED if >=5500, approaching the writer's
8000-token cap). This is reproducible and self-correcting: a retried/fixed revision
clears its flag automatically. The 3 originally-bundled flags all carried
word-count-based notes, so this reproduces them exactly.
"""
from __future__ import annotations
import csv, json, re, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "run_20260531-022438"
OUT = RUN_DIR / "revisions_dataset"
MANIFEST = OUT / "manifest.csv"

PROBES = ("P1", "P2", "P3", "P4", "P5")
ARMS = ("A", "B", "AB")
VARIANTS = ("neutral", "selective")              # pooled scope (writer_judge_v3)
PERPROBE_VARIANTS = ("neutral", "selective", "selective_blind")  # per-probe scope
N_READERS = {"A": 2, "B": 2, "AB": 4}

REFUSAL_MAX_WORDS = 200      # below this, the writer almost certainly refused/empty
TRUNCATION_MIN_WORDS = 5500  # at/above this, likely hit writer max_tokens=8000


def derive_flag(word_count: int) -> tuple[str, str]:
    """Quality flag derived purely from word count, so it is reproducible and never
    goes stale: a retried/fixed revision automatically clears its flag on re-compile.
    The 3 originally-bundled flags all carried word-count-based notes, so this
    reproduces them exactly while staying self-correcting."""
    if word_count < REFUSAL_MAX_WORDS:
        return ("WRITER_REFUSED", "word_count<200 — likely model refusal or empty output")
    if word_count >= TRUNCATION_MIN_WORDS:
        return ("TRUNCATED", f"word_count={word_count} approaches writer max_tokens=8000 cap")
    return ("", "")


def parse_cell_id(story_id: str) -> dict:
    """cell-01-recent-sp-pure__run2 -> {cell, era, perspective, grounding, run}."""
    m = re.match(
        r"^(cell-\d{2})-(recent|middle|distant)-(sp|sys)-(pure|fantastical)(?:__run(\d+))?$",
        story_id,
    )
    if not m:
        raise ValueError(f"unparseable story_id: {story_id}")
    cell, era, persp, ground, run = m.groups()
    return {
        "cell": cell,
        "era": era,
        "perspective": persp,
        "grounding": ground,
        "run": int(run) if run else 1,
    }


def word_count(text: str) -> int:
    return len(text.split())


def copy_revision(src: Path, dst: Path) -> int:
    """Copy with a tiny normalization: strip BOM, ensure trailing newline. Returns word count."""
    text = src.read_text()
    if text.startswith("﻿"):
        text = text[1:]
    if not text.endswith("\n"):
        text = text + "\n"
    dst.write_text(text)
    return word_count(text)


def main():
    meta = json.loads((RUN_DIR / "meta.json").read_text())
    stories = meta["stories"]
    print(f"compiling {len(stories)} stories x 49 files = {len(stories) * 49} expected files")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []

    for sid in stories:
        story_dir = OUT / sid
        story_dir.mkdir(parents=True, exist_ok=True)
        axes = parse_cell_id(sid)

        # --- original
        src = RUN_DIR / "stories" / sid / "story.txt"
        dst = story_dir / "original.md"
        if src.exists():
            wc = copy_revision(src, dst)
            rows.append({
                **axes,
                "story_id": sid,
                "config": "original",
                "scope": "original",
                "consolidator": "",
                "arm": "",
                "probe": "",
                "n_readers": 0,
                "file_path": str(dst.relative_to(RUN_DIR)),
                "word_count": wc,
            })
        else:
            missing.append(("original", str(src)))

        # --- pooled neutral, selective
        for variant in VARIANTS:
            src = RUN_DIR / "writer_judge_v3" / sid / f"revised_{variant}.txt"
            dst = story_dir / f"pooled_{variant}.md"
            if src.exists():
                wc = copy_revision(src, dst)
                rows.append({
                    **axes,
                    "story_id": sid,
                    "config": f"pooled_{variant}",
                    "scope": "pooled",
                    "consolidator": variant,
                    "arm": "pool20",
                    "probe": "all",
                    "n_readers": 20,
                    "file_path": str(dst.relative_to(RUN_DIR)),
                    "word_count": wc,
                })
            else:
                missing.append((f"pooled_{variant}", str(src)))

        # --- pooled blind
        src = RUN_DIR / "writer_judge_v3_blind" / sid / "revised_selective_blind.txt"
        dst = story_dir / "pooled_blind.md"
        if src.exists():
            wc = copy_revision(src, dst)
            rows.append({
                **axes,
                "story_id": sid,
                "config": "pooled_blind",
                "scope": "pooled",
                "consolidator": "selective_blind",
                "arm": "pool20",
                "probe": "all",
                "n_readers": 20,
                "file_path": str(dst.relative_to(RUN_DIR)),
                "word_count": wc,
            })
        else:
            missing.append(("pooled_blind", str(src)))

        # --- per-probe x arm x variant
        for probe in PROBES:
            for arm in ARMS:
                for variant in PERPROBE_VARIANTS:
                    src = (RUN_DIR / "revise_judge_byprobe_v3" / sid / probe
                           / f"revision_{arm}_{variant}.txt")
                    dst = story_dir / f"{probe}_{arm}_{variant}.md"
                    if src.exists():
                        wc = copy_revision(src, dst)
                        rows.append({
                            **axes,
                            "story_id": sid,
                            "config": f"{probe}_{arm}_{variant}",
                            "scope": "per_probe",
                            "consolidator": variant,
                            "arm": arm,
                            "probe": probe,
                            "n_readers": N_READERS[arm],
                            "file_path": str(dst.relative_to(RUN_DIR)),
                            "word_count": wc,
                        })
                    else:
                        missing.append((f"{probe}_{arm}_{variant}/{sid}", str(src)))

    # --- quality flags: derived purely from word count (reproducible, self-correcting)
    for r in rows:
        if r["config"] == "original":
            r["flag"], r["notes"] = "", ""
        else:
            r["flag"], r["notes"] = derive_flag(r["word_count"])

    # --- manifest
    fields = ["story_id", "cell", "era", "perspective", "grounding", "run",
              "config", "scope", "consolidator", "arm", "probe",
              "n_readers", "word_count", "file_path", "flag", "notes"]
    with MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    # --- summary
    by_config_count = {}
    for r in rows:
        by_config_count[r["config"]] = by_config_count.get(r["config"], 0) + 1
    print(f"\ntotal files written: {len(rows)}")
    print(f"manifest: {MANIFEST.relative_to(RUN_DIR)}")
    print(f"unique configs: {len(by_config_count)}")
    print(f"per-story config count (avg): {len(rows) / len(stories):.2f}")
    print(f"missing files: {len(missing)}")
    if missing:
        print("  first 10 missing:")
        for what, where in missing[:10]:
            print(f"    {what}  ->  {where}")
    return rows, missing


if __name__ == "__main__":
    main()
