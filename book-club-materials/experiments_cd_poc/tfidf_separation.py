"""Coder-independent A-vs-B separation via TF-IDF cosine.

Does NOT use the DeepSeek coder at all — purely lexical. For each (alpha, probe)
we fit TF-IDF on that slice's pass-2 texts, then compare mean between-group
cosine DISTANCE to mean within-group distance. separation = between - within.
A larger separation under alpha=1 than alpha=0 is coder-independent evidence
that CD pushes Group A and Group B texts further apart.

Permutation test: shuffle A/B labels within each (cell) stratum, recompute
separation, p = fraction of shuffles with separation >= observed.
"""
from __future__ import annotations

import glob
import json
import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent / "run_cd_poc"
OUT = Path(__file__).resolve().parent / "aggregated"
rng = random.Random(17)
N_PERM = 5000


def load_rows():
    rows = []
    for d in sorted(BASE.glob("alpha*/results/*/*/*/agent-*")):
        p2 = d / "pass2.txt"
        if not p2.exists():
            continue
        rows.append({
            "alpha": d.parts[-6], "group": d.parts[-4], "cell": d.parts[-3],
            "probe": d.parts[-2], "uid": d.parts[-1].replace("agent-", ""),
            "text": p2.read_text(),
        })
    return pd.DataFrame(rows)


def mean_pair_dist(idx_a, idx_b, sim, same):
    pairs = (itertools.combinations(idx_a, 2) if same
             else itertools.product(idx_a, idx_b))
    ds = [1.0 - sim[i, j] for i, j in pairs]
    return float(np.mean(ds)) if ds else float("nan")


def separation_for(sub):
    vec = TfidfVectorizer(stop_words="english", max_features=4000,
                          ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(sub["text"].tolist())
    sim = cosine_similarity(X)
    pos = {g: [i for i, gg in enumerate(sub["group"].tolist()) if gg == g]
           for g in ("A", "B")}
    w = (mean_pair_dist(pos["A"], pos["A"], sim, True)
         + mean_pair_dist(pos["B"], pos["B"], sim, True)) / 2
    btw = mean_pair_dist(pos["A"], pos["B"], sim, False)
    return btw - w, btw, w, sim, sub["group"].tolist(), sub["cell"].tolist()


def perm_p(sim, groups, cells, observed):
    idx = list(range(len(groups)))
    by_cell = {}
    for i, c in enumerate(cells):
        by_cell.setdefault(c, []).append(i)
    count = 0
    for _ in range(N_PERM):
        perm = groups[:]
        for c, members in by_cell.items():
            labels = [groups[i] for i in members]
            rng.shuffle(labels)
            for i, l in zip(members, labels):
                perm[i] = l
        posA = [i for i in idx if perm[i] == "A"]
        posB = [i for i in idx if perm[i] == "B"]
        w = (mean_pair_dist(posA, posA, sim, True)
             + mean_pair_dist(posB, posB, sim, True)) / 2
        btw = mean_pair_dist(posA, posB, sim, False)
        if (btw - w) >= observed:
            count += 1
    return count / N_PERM


def main():
    df = load_rows()
    rows = []
    for (alpha, probe), sub in df.groupby(["alpha", "probe"]):
        sub = sub.reset_index(drop=True)
        sep, btw, w, sim, groups, cells = separation_for(sub)
        p = perm_p(sim, groups, cells, sep)
        rows.append({"alpha": alpha, "probe": probe,
                     "between": round(btw, 4), "within": round(w, 4),
                     "separation": round(sep, 4), "p_perm": round(p, 4),
                     "n": len(sub)})
    res = pd.DataFrame(rows).sort_values(["probe", "alpha"])
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "tfidf_separation.csv", index=False)
    print(res.to_string(index=False))
    print("\n=== separation: alpha1 - alpha0 by probe (positive = CD widens) ===")
    piv = res.pivot(index="probe", columns="alpha", values="separation")
    piv["delta"] = piv["alpha1"] - piv["alpha0"]
    print(piv.to_string())


if __name__ == "__main__":
    main()
