"""Analysis for the CD proof-of-concept.

Question: does contrastive decoding (alpha=1) increase A-vs-B group
differentiation relative to baseline (alpha=0), per probe?

Three families of evidence:
1. Coder dimensions (v3 coder): A-B gap per (alpha, probe, dim) with a
   permutation test that shuffles group labels within (cell, probe) strata.
2. convention_type: share of "historical" framing per (alpha, probe, group).
3. Embedding heterogeneity (coder-independent): Qwen3-Embedding-0.6B vectors
   of pass-2 texts; between-group vs within-group cosine separation per
   (cell, probe, alpha).

Outputs: aggregated/ CSVs + printed summary tables.
"""
from __future__ import annotations

import json
import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent / "run_cd_poc"
OUT = Path(__file__).resolve().parent / "aggregated"
DIMS = ["period_specificity", "concreteness", "knowledge_invoked",
        "locations_cited", "anchors"]
N_PERM = 10000
rng = random.Random(17)


def collect() -> pd.DataFrame:
    rows = []
    for d in sorted(BASE.glob("alpha*/results/*/*/*/agent-*")):
        cj = d / "coding.json"
        if not cj.exists():
            continue
        alpha = d.parts[-6]
        group, cell, probe, agent = d.parts[-4], d.parts[-3], d.parts[-2], d.parts[-1]
        coding = json.loads(cj.read_text())
        meta = json.loads((d / "meta.json").read_text())
        for pas in ("pass1", "pass2"):
            c = coding.get(pas, {})
            if "_parse_error" in c:
                continue
            rows.append({
                "alpha": alpha, "group": group, "cell": cell, "probe": probe,
                "uid": agent.replace("agent-", ""), "pass": pas,
                **{k: c.get(k, 0) for k in DIMS},
                "convention_type": c.get("convention_type", "none"),
                "n_tokens": meta[pas]["n_tokens"],
                "text_path": str(d / f"{pas}.txt"),
            })
    return pd.DataFrame(rows)


def perm_test(df: pd.DataFrame, dim: str) -> float:
    """Permutation p for the A-B gap, shuffling group labels within cells."""
    obs = df[df.group == "A"][dim].mean() - df[df.group == "B"][dim].mean()
    cells = {c: sub for c, sub in df.groupby("cell")}
    count = 0
    for _ in range(N_PERM):
        diffs_a, diffs_b = [], []
        for c, sub in cells.items():
            vals = sub[dim].tolist()
            labels = sub["group"].tolist()
            rng.shuffle(labels)
            for v, l in zip(vals, labels):
                (diffs_a if l == "A" else diffs_b).append(v)
        sim = np.mean(diffs_a) - np.mean(diffs_b)
        if abs(sim) >= abs(obs):
            count += 1
    return count / N_PERM


def gaps_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    p2 = df[df["pass"] == "pass2"]
    for (alpha, probe), sub in p2.groupby(["alpha", "probe"]):
        for dim in DIMS:
            a = sub[sub.group == "A"][dim]
            b = sub[sub.group == "B"][dim]
            rows.append({
                "alpha": alpha, "probe": probe, "dim": dim,
                "A_mean": round(a.mean(), 3), "B_mean": round(b.mean(), 3),
                "gap": round(a.mean() - b.mean(), 3),
                "p_perm": perm_test(sub, dim),
                "n_A": len(a), "n_B": len(b),
            })
    return pd.DataFrame(rows)


def convention_table(df: pd.DataFrame) -> pd.DataFrame:
    p2 = df[df["pass"] == "pass2"]
    rows = []
    for (alpha, probe, group), sub in p2.groupby(["alpha", "probe", "group"]):
        n = len(sub)
        rows.append({
            "alpha": alpha, "probe": probe, "group": group, "n": n,
            "historical": round((sub.convention_type == "historical").mean(), 3),
            "generic": round((sub.convention_type == "generic-genre").mean(), 3),
            "mixed": round((sub.convention_type == "mixed").mean(), 3),
            "none": round((sub.convention_type == "none").mean(), 3),
        })
    return pd.DataFrame(rows)


def embed_texts(paths: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer
    mp = "/project/jevans/maxzhuyt/models/Qwen3-Embedding-0.6B"
    tok = AutoTokenizer.from_pretrained(mp, padding_side="left")
    model = AutoModel.from_pretrained(mp, dtype=torch.float32)
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    vecs = []
    with torch.inference_mode():
        for i in range(0, len(paths), 8):
            # Qwen3-Embedding convention: append EOS, last-token pool, L2-normalize
            batch = [Path(p).read_text()[:8000] + tok.eos_token
                     for p in paths[i:i + 8]]
            enc = tok(batch, padding=True, truncation=True, max_length=2048,
                      return_tensors="pt").to(dev)
            out = model(**enc)
            emb = out.last_hidden_state[:, -1, :]   # left padding -> real last tok
            emb = torch.nn.functional.normalize(emb, dim=-1)
            vecs.append(emb.float().cpu())
    return torch.cat(vecs).numpy()


def separation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Between-group minus within-group mean cosine distance per (cell, probe, alpha)."""
    p2 = df[df["pass"] == "pass2"].reset_index(drop=True)
    embs = embed_texts(p2.text_path.tolist())
    rows = []
    for (alpha, cell, probe), sub in p2.groupby(["alpha", "cell", "probe"]):
        ia = sub.index[sub.group == "A"].tolist()
        ib = sub.index[sub.group == "B"].tolist()
        if len(ia) < 2 or len(ib) < 2:
            continue
        def mean_dist(idx1, idx2, same):
            pairs = (itertools.combinations(idx1, 2) if same
                     else itertools.product(idx1, idx2))
            ds = [1 - float(embs[i] @ embs[j]) for i, j in pairs]
            return float(np.mean(ds))
        w_a = mean_dist(ia, ia, True)
        w_b = mean_dist(ib, ib, True)
        btw = mean_dist(ia, ib, False)
        rows.append({
            "alpha": alpha, "cell": cell, "probe": probe,
            "within_A": round(w_a, 4), "within_B": round(w_b, 4),
            "between": round(btw, 4),
            "separation": round(btw - (w_a + w_b) / 2, 4),
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    df = collect()
    print(f"collected {len(df)} coded passes "
          f"({df[df['pass'] == 'pass2'].shape[0]} pass-2)")
    df.drop(columns=["text_path"]).to_csv(OUT / "all_codings.csv", index=False)

    g = gaps_table(df)
    g.to_csv(OUT / "ab_gaps_by_alpha_probe_dim.csv", index=False)
    print("\n=== A-B gaps (pass-2) ===")
    print(g.to_string(index=False))

    c = convention_table(df)
    c.to_csv(OUT / "convention_by_alpha_probe_group.csv", index=False)
    print("\n=== convention_type shares (pass-2) ===")
    print(c.to_string(index=False))

    s = separation_table(df)
    s.to_csv(OUT / "embedding_separation.csv", index=False)
    print("\n=== embedding separation (pass-2) ===")
    print(s.to_string(index=False))
    print("\nmean separation by alpha:")
    print(s.groupby("alpha")["separation"].agg(["mean", "std", "count"]))

    # length dispersion: does CD restore persona-specific verbosity?
    p2 = df[df["pass"] == "pass2"]
    lens = p2.groupby(["alpha", "uid"])["n_tokens"].mean().reset_index()
    disp = lens.groupby("alpha")["n_tokens"].std()
    print("\n=== per-persona mean length std (token dispersion across personas) ===")
    print(disp)


if __name__ == "__main__":
    main()
