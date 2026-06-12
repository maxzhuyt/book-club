"""Build V2 persona system prompts from the corpus, grounded in each selected
user's real reading history + two verbatim review exemplars. Group membership
is implicit: a factual genre-breakdown line, never an evaluative label."""
from __future__ import annotations
import json, os
from collections import Counter
from pathlib import Path
import pandas as pd

# Path to the upstream Goodreads scrape; not bundled with this project.
# Set GOODREADS_CORPUS env var or override the default below.
# Only needed if regenerating personas from scratch.
CORPUS = Path(os.environ.get("GOODREADS_CORPUS",
                              Path.home() / "book_club" / "scraping_goodreads"))
HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "personas_v3"
GROUPS = HERE / "groups.json"

_cache = {}


def _load():
    if _cache:
        return _cache
    _cache["ub"] = pd.read_parquet(CORPUS / "user_books.parquet")
    _cache["bk"] = pd.read_parquet(CORPUS / "books.parquet")
    _cache["rv"] = pd.read_parquet(CORPUS / "user_reviews_full.parquet")
    return _cache


def _user_books(uid):
    d = _load()
    ub = d["ub"][d["ub"]["user_id"].astype(str) == str(uid)]
    return ub.merge(d["bk"], on="book_id", how="left")


def build_persona(uid: str) -> str:
    d = _load()
    mb = _user_books(uid)
    n_books = len(mb)
    gc = Counter()
    for gs in mb["genres"].dropna():
        for g in list(gs):
            gc[g] += 1
    top_genres = gc.most_common(6)
    hf = gc.get("Historical Fiction", 0)
    hf_pct = round(100.0 * hf / max(n_books, 1))
    ac = Counter()
    for a in mb["authors"].dropna():
        ac[str(a)] += 1
    top_authors = [a for a, _ in ac.most_common(3)]
    yrs = pd.to_numeric(mb["pub_year"], errors="coerce").dropna()
    median_year = int(yrs.median()) if len(yrs) else None

    rv = d["rv"]
    ur = rv[(rv["user_id"].astype(str) == str(uid)) & rv["text"].notna()].copy()
    ur["text"] = ur["text"].astype(str)
    ur["wc"] = ur["text"].str.split().str.len()
    avg_words = int(ur["wc"].mean()) if len(ur) else 0
    ur = ur.sort_values("rating", ascending=False)
    if len(ur) >= 2:
        exemplars = [ur.iloc[0], ur.iloc[-1]]
    elif len(ur) == 1:
        exemplars = [ur.iloc[0], ur.iloc[0]]
    else:
        exemplars = []

    def book_title(bid):
        row = d["bk"][d["bk"]["book_id"] == bid]
        return str(row["title"].iloc[0]) if len(row) else "a book"

    genre_line = ", ".join(f"{g} ({round(100*c/max(n_books,1))}% of your reading)"
                           for g, c in top_genres)
    lines = []
    lines.append("You are a participant in a reading discussion. Below is your reading "
                 "identity and personal review style. Stay in character — write as this "
                 "reader would write, even when others disagree.\n")
    lines.append("--- YOUR READING IDENTITY ---")
    lines.append(f"• You have read {n_books} books.")
    if median_year:
        lines.append(f"• Your reading centers around books published near {median_year}.")
    if top_authors:
        lines.append(f"• Authors you return to: {', '.join(top_authors)}.")
    lines.append(f"• Your reviews average about {avg_words} words.")
    lines.append("--- END IDENTITY ---\n")
    lines.append("--- GENRE BREAKDOWN (factual; from your shelved books) ---")
    lines.append(f"• Historical Fiction: {hf_pct}% of your reading.")
    lines.append(f"• Your genre mix: {genre_line}.")
    lines.append("--- END GENRE BREAKDOWN ---\n")
    lines.append("--- YOUR VOICE ---")
    lines.append("Two real reviews you wrote. Absorb their style (diction, length, how "
                 "you express praise or criticism); do NOT copy them.\n")
    for i, ex in enumerate(exemplars, 1):
        txt = ex["text"].strip()
        if len(txt) > 2200:
            txt = txt[:2200] + " […]"
        lines.append(f"REVIEW {i} (for \"{book_title(ex['book_id'])}\"):\n{txt}\n")
    lines.append("--- END VOICE ---")
    return "\n".join(lines)


def main():
    OUTDIR.mkdir(exist_ok=True)
    members = json.loads(GROUPS.read_text())["members"]
    index = {}
    for grp in ("A", "B"):
        for m in members[grp]:
            uid = m["user_id"]
            (OUTDIR / f"{uid}.txt").write_text(build_persona(uid))
            index[uid] = {"group": grp, "hf_frac": m["hf_frac"]}
    (OUTDIR / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Wrote {len(index)} persona prompts to {OUTDIR}")


if __name__ == "__main__":
    main()
