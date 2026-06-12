"""Derive two confound-matched reader groups from the Goodreads corpus.

Group A: avid historical-fiction readers.  Group B: low-HF controls, matched to
A on (avg_review_words, analyticity, n_books) so the systematic difference is
HF exposure, not style or volume.
"""
from __future__ import annotations
import json, os, re
from pathlib import Path
import numpy as np
import pandas as pd

# Path to the upstream Goodreads scrape (user_books.parquet, books.parquet,
# user_reviews_full.parquet). Not bundled with this project; set GOODREADS_CORPUS
# env var or override the default below. Only needed if regenerating groups/personas.
CORPUS = Path(os.environ.get("GOODREADS_CORPUS",
                              Path.home() / "book_club" / "scraping_goodreads"))
OUT = Path(__file__).resolve().parent / "groups.json"

# analyticity proxy lexicon (craft/structure vocabulary); rate per 1000 words
ANALYTIC_LEXICON = ["structure", "structural", "prose", "pacing", "pace", "character",
                    "characters", "theme", "thematic", "narrative", "device", "motif",
                    "voice", "imagery", "metaphor", "plot", "tension", "craft"]
_LEX = re.compile(r"\b(" + "|".join(ANALYTIC_LEXICON) + r")\b", re.IGNORECASE)


def _is_hf(genres) -> bool:
    try:
        return "Historical Fiction" in list(genres)
    except TypeError:
        return False


def compute_user_table() -> pd.DataFrame:
    ub = pd.read_parquet(CORPUS / "user_books.parquet")
    bk = pd.read_parquet(CORPUS / "books.parquet")
    rv = pd.read_parquet(CORPUS / "user_reviews_full.parquet")

    bk = bk.assign(is_hf=bk["genres"].apply(_is_hf))
    m = ub.merge(bk[["book_id", "is_hf"]], on="book_id", how="left")
    g = m.groupby("user_id").agg(n_books=("book_id", "size"),
                                 n_hf=("is_hf", "sum")).reset_index()
    g["hf_frac"] = g["n_hf"] / g["n_books"]

    # review-derived stats
    rv = rv.dropna(subset=["text"])
    rv["text"] = rv["text"].astype(str)
    rv["wc"] = rv["text"].str.split().str.len()
    rv["analytic_hits"] = rv["text"].apply(lambda t: len(_LEX.findall(t)))
    rstats = rv.groupby("user_id").agg(
        n_reviews=("review_id", "size"),
        avg_review_words=("wc", "mean"),
        total_words=("wc", "sum"),
        total_hits=("analytic_hits", "sum"),
    ).reset_index()
    rstats["analyticity"] = 1000.0 * rstats["total_hits"] / rstats["total_words"].clip(lower=1)

    df = g.merge(rstats, on="user_id", how="left")
    df["n_reviews"] = df["n_reviews"].fillna(0).astype(int)
    df["avg_review_words"] = df["avg_review_words"].fillna(0.0)
    df["analyticity"] = df["analyticity"].fillna(0.0)
    return df


def select_groups(df: pd.DataFrame, n_per_group: int = 10,
                  hi=0.35, lo=0.05, min_books=50, min_reviews=2) -> dict:
    elig = df[(df["n_reviews"] >= min_reviews) & (df["n_books"] >= min_books)].copy()
    confounds = ["avg_review_words", "analyticity", "n_books"]
    mu = elig[confounds].mean(); sd = elig[confounds].std().replace(0, 1.0)
    z = (elig.set_index("user_id")[confounds] - mu) / sd

    # Group A: top hf_frac among eligible, spanning review-length styles.
    a_pool = elig[elig["hf_frac"] >= hi].sort_values("hf_frac", ascending=False)
    if len(a_pool) < n_per_group:
        a_pool = elig.sort_values("hf_frac", ascending=False)
    a_pool = a_pool.head(max(n_per_group * 2, n_per_group))
    a_sorted = a_pool.sort_values("avg_review_words")
    idx = np.linspace(0, len(a_sorted) - 1, n_per_group).round().astype(int)
    A = a_sorted.iloc[idx]["user_id"].tolist()
    a_mean = z.loc[A].mean().values

    # Group B: low hf_frac, greedily minimize distance of running B-mean to A-mean.
    b_pool = elig[(elig["hf_frac"] <= lo) & (~elig["user_id"].isin(A))].copy()
    chosen, running = [], []
    for _ in range(n_per_group):
        best, best_d = None, 1e9
        for uid in b_pool["user_id"]:
            if uid in chosen:
                continue
            trial = np.array(running + [z.loc[uid].values])
            d = np.linalg.norm(trial.mean(axis=0) - a_mean)
            if d < best_d:
                best_d, best = d, uid
        chosen.append(best); running.append(z.loc[best].values)
    B = chosen
    return {**{u: "A" for u in A}, **{u: "B" for u in B}}


def main():
    df = compute_user_table()
    sel = select_groups(df)
    t = df.set_index("user_id")
    members = {"A": [], "B": []}
    for uid, grp in sel.items():
        r = t.loc[uid]
        members[grp].append({
            "user_id": str(uid), "hf_frac": round(float(r.hf_frac), 3),
            "n_books": int(r.n_books), "n_reviews": int(r.n_reviews),
            "avg_review_words": round(float(r.avg_review_words), 1),
            "analyticity": round(float(r.analyticity), 2),
        })
    balance = {c: {"A": round(float(t.loc[[u for u in sel if sel[u] == 'A'], c].mean()), 2),
                   "B": round(float(t.loc[[u for u in sel if sel[u] == 'B'], c].mean()), 2)}
               for c in ["avg_review_words", "analyticity", "n_books", "hf_frac"]}
    OUT.write_text(json.dumps({"lexicon": ANALYTIC_LEXICON, "members": members,
                               "group_means": balance}, indent=2))
    print("Wrote", OUT)
    print(json.dumps(balance, indent=2))


if __name__ == "__main__":
    main()
