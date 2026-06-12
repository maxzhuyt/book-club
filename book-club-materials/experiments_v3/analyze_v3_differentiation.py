"""V3 reader-response differentiation analysis (C1 of the intermediate report).

Loads all 720 reader codings and shows:

(1) probe-level differentiation:
    Do means differ across the 5 probes on each attention dim?
    -> per-dim permutation test (10k perms) on between-probe variance, with effect size eta^2.

(2) probe x group:
    Does the A-B gap (HF vs control reader) depend on probe?
    -> per (probe, dim) permutation test on A-B gap (5k perms), with 95% bootstrap CI.

(3) cell-axis sub-breakdown:
    Means by (probe, era), (probe, perspective), (probe, grounding) so the report
    can note whether the probe-differentiation pattern is uniform across cell types.

(4) convention_type by probe x group:
    Re-tabulates the existing convention_type proportions plus chi-squared on
    (probe, convention_type) independence.

Outputs CSVs to <run>/aggregated_differentiation/ and a JSON summary, plus
text dumps of the permutation/chi-squared results for the report.
"""
from __future__ import annotations
import csv, json, re, sys, time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Iterable

import random

HERE = Path(__file__).resolve().parent
RUN = HERE / "run_20260531-022438"
RESULTS = RUN / "results"
OUT = RUN / "aggregated_differentiation"
OUT.mkdir(parents=True, exist_ok=True)

GRADED = ("period_specificity", "concreteness", "knowledge_invoked",
          "locations_cited", "anchors")
CONV = ("historical", "generic-genre", "mixed", "none")
PROBES = ("P1", "P2", "P3", "P4", "P5")
PROBE_NAME = {"P1": "Plausibility", "P2": "Knowledge-gap", "P3": "Stability",
              "P4": "Convention", "P5": "Salience"}
GROUPS = ("A", "B")

RNG_SEED = 17
RNG = random.Random(RNG_SEED)


def parse_cell_id(sid: str) -> dict:
    m = re.match(
        r"^(cell-\d{2})-(recent|middle|distant)-(sp|sys)-(pure|fantastical)(?:__run(\d+))?$",
        sid,
    )
    cell, era, persp, ground, run = m.groups()
    return {"cell": cell, "era": era, "perspective": persp,
            "grounding": ground, "run": int(run) if run else 1}


def load_rows() -> list[dict]:
    rows = []
    for cj in RESULTS.rglob("coding.json"):
        parts = cj.relative_to(RESULTS).parts  # g/cell/probe/agent-N/coding.json
        if len(parts) < 5:
            continue
        g, sid, probe, ag, _ = parts
        try:
            obj = json.loads(cj.read_text())
        except Exception:
            continue
        axes = parse_cell_id(sid)
        slot = int(ag.split("-")[1])
        for ph in ("pass1", "pass2"):
            blk = obj.get(ph, {})
            if "_parse_error" in blk:
                continue
            r = {"group": g, "story_id": sid, **axes,
                 "probe": probe, "slot": slot, "phase": ph}
            for k in GRADED:
                r[k] = int(blk.get(k, 0))
            ct = blk.get("convention_type", "none")
            r["convention_type"] = ct if ct in CONV else "none"
            rows.append(r)
    return rows


# ============================ STATISTICS ============================

def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def variance(xs, m=None):
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs) if m is None else m
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def between_group_var_eta2(values_by_group: dict) -> tuple[float, float]:
    """Return (eta^2, F-like statistic = SS_between / SS_within)."""
    all_vals = [v for g in values_by_group.values() for v in g]
    if len(all_vals) < 2:
        return 0.0, 0.0
    grand = mean(all_vals)
    ss_total = sum((v - grand) ** 2 for v in all_vals)
    ss_between = sum(len(g) * (mean(g) - grand) ** 2 for g in values_by_group.values())
    ss_within = ss_total - ss_between
    eta2 = ss_between / ss_total if ss_total > 0 else 0.0
    f = ss_between / ss_within if ss_within > 0 else float("inf")
    return eta2, f


def perm_probe_differentiation(rows: list[dict], dim: str, n_perm: int = 10000) -> dict:
    """One-way permutation test: shuffle probe labels and see how often the
    permuted between-probe variance >= observed. Returns p, eta^2, F."""
    obs_by_probe = defaultdict(list)
    for r in rows:
        obs_by_probe[r["probe"]].append(r[dim])
    eta2_obs, f_obs = between_group_var_eta2(obs_by_probe)
    if not eta2_obs:
        return {"eta2": 0.0, "F": 0.0, "p_value": 1.0, "n": len(rows)}
    flat_vals = [r[dim] for r in rows]
    flat_probes = [r["probe"] for r in rows]
    rng = random.Random(RNG_SEED + hash(dim) % 1000)
    count_ge = 0
    for _ in range(n_perm):
        idxs = list(range(len(flat_probes)))
        rng.shuffle(idxs)
        permed = defaultdict(list)
        for i, j in enumerate(idxs):
            permed[flat_probes[i]].append(flat_vals[j])
        e2, _ = between_group_var_eta2(permed)
        if e2 >= eta2_obs:
            count_ge += 1
    return {"eta2": round(eta2_obs, 4), "F": round(f_obs, 3),
            "p_value": (count_ge + 1) / (n_perm + 1),
            "n": len(rows), "n_perm": n_perm}


def perm_ab_gap(values_a: list, values_b: list, n_perm: int = 5000) -> dict:
    """Permutation test on |mean(A) - mean(B)|. Two-sided."""
    obs = mean(values_a) - mean(values_b)
    combined = values_a + values_b
    rng = random.Random(RNG_SEED * 2)
    nA = len(values_a)
    count_ge = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        pa = combined[:nA]
        pb = combined[nA:]
        d = mean(pa) - mean(pb)
        if abs(d) >= abs(obs):
            count_ge += 1
    return {"gap": round(obs, 3), "nA": nA, "nB": len(values_b),
            "p_value": (count_ge + 1) / (n_perm + 1)}


def bootstrap_ci(values: list, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    m_obs = mean(values)
    rng = random.Random(RNG_SEED * 3 + len(values))
    boots = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(mean(sample))
    boots.sort()
    lo = boots[int(n_boot * alpha / 2)]
    hi = boots[int(n_boot * (1 - alpha / 2))]
    return round(m_obs, 3), round(lo, 3), round(hi, 3)


def chi_squared_independence(table: list[list[int]]) -> tuple[float, int]:
    """Chi-squared test on a contingency table. Returns (chi2, df)."""
    rsum = [sum(r) for r in table]
    csum = [sum(table[r][c] for r in range(len(table))) for c in range(len(table[0]))]
    total = sum(rsum)
    if not total:
        return 0.0, 0
    chi2 = 0.0
    for r in range(len(table)):
        for c in range(len(table[0])):
            exp = rsum[r] * csum[c] / total
            if exp > 0:
                chi2 += (table[r][c] - exp) ** 2 / exp
    df = (len(table) - 1) * (len(table[0]) - 1)
    return chi2, df


def chi2_pvalue_perm(table: list[list[int]], n_perm: int = 5000) -> dict:
    """Approximate p-value for table independence via row-shuffled permutations.
    Each row keeps its marginal; columns are permuted within each row."""
    chi2_obs, df = chi_squared_independence(table)
    rng = random.Random(RNG_SEED * 4)
    rows = [list(r) for r in table]
    flat_cells = []
    for r in rows:
        for c, v in enumerate(r):
            flat_cells.extend([c] * v)
    count_ge = 0
    rsum = [sum(r) for r in rows]
    ncols = len(rows[0])
    for _ in range(n_perm):
        rng.shuffle(flat_cells)
        permed = []
        idx = 0
        for n in rsum:
            row = [0] * ncols
            for v in flat_cells[idx:idx + n]:
                row[v] += 1
            idx += n
            permed.append(row)
        c2p, _ = chi_squared_independence(permed)
        if c2p >= chi2_obs:
            count_ge += 1
    return {"chi2": round(chi2_obs, 3), "df": df,
            "p_value": (count_ge + 1) / (n_perm + 1)}


# ============================ AGGREGATION ============================

def agg_probe_group_dim(rows: list[dict]) -> list[dict]:
    out = []
    for probe in PROBES:
        for g in GROUPS:
            sub = [r for r in rows if r["probe"] == probe and r["group"] == g]
            for dim in GRADED:
                vals = [r[dim] for r in sub]
                m, lo, hi = bootstrap_ci(vals)
                out.append({"probe": probe, "probe_name": PROBE_NAME[probe],
                            "group": g, "dim": dim, "n": len(vals),
                            "mean": m, "ci_lo": lo, "ci_hi": hi})
    return out


def agg_probe_dim(rows: list[dict]) -> list[dict]:
    out = []
    for probe in PROBES:
        sub = [r for r in rows if r["probe"] == probe]
        for dim in GRADED:
            vals = [r[dim] for r in sub]
            m, lo, hi = bootstrap_ci(vals)
            out.append({"probe": probe, "probe_name": PROBE_NAME[probe],
                        "dim": dim, "n": len(vals),
                        "mean": m, "ci_lo": lo, "ci_hi": hi})
    return out


def agg_probe_axis(rows: list[dict], axis: str) -> list[dict]:
    out = []
    levels = sorted({r[axis] for r in rows})
    for probe in PROBES:
        for lev in levels:
            sub = [r for r in rows if r["probe"] == probe and r[axis] == lev]
            for dim in GRADED:
                vals = [r[dim] for r in sub]
                if vals:
                    m, lo, hi = bootstrap_ci(vals)
                else:
                    m, lo, hi = 0, 0, 0
                out.append({"probe": probe, "probe_name": PROBE_NAME[probe],
                            "axis": axis, "level": lev, "dim": dim,
                            "n": len(vals), "mean": m,
                            "ci_lo": lo, "ci_hi": hi})
    return out


def agg_convention(rows: list[dict]) -> list[dict]:
    """Per (probe, group): convention_type proportions."""
    out = []
    for probe in PROBES:
        for g in GROUPS:
            sub = [r for r in rows if r["probe"] == probe and r["group"] == g]
            n = len(sub)
            counts = Counter(r["convention_type"] for r in sub)
            row = {"probe": probe, "probe_name": PROBE_NAME[probe],
                   "group": g, "n": n}
            for ct in CONV:
                row[ct] = round(counts[ct] / n, 3) if n else 0.0
                row[f"{ct}_n"] = counts[ct]
            out.append(row)
    return out


def write_csv(path: Path, rows: list[dict], header: list[str]):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in header})


# ============================ MAIN ============================

def main():
    rows = load_rows()
    print(f"loaded {len(rows)} reader-coding rows (pass1 + pass2 across {len(set(r['story_id'] for r in rows))} stories)")
    print(f"  groups: {dict(Counter(r['group'] for r in rows))}")
    print(f"  phases: {dict(Counter(r['phase'] for r in rows))}")
    print(f"  probes: {dict(Counter(r['probe'] for r in rows))}")

    # Filter to pass2 for the headline analysis (the probe-conditioned response).
    # Aggregations on pass1 are also produced for the lift comparison.
    p2 = [r for r in rows if r["phase"] == "pass2"]
    p1 = [r for r in rows if r["phase"] == "pass1"]
    print(f"\nusing {len(p2)} pass2 rows for headline differentiation analyses")

    # 1) probe-level differentiation: permutation test per dim
    print("\n=== (1) probe-level differentiation (pass2 only) ===")
    diff_rows = []
    for dim in GRADED:
        res = perm_probe_differentiation(p2, dim, n_perm=10000)
        print(f"  {dim:22s}  eta2={res['eta2']:.3f}  F={res['F']:.2f}  p={res['p_value']:.4f}  n={res['n']}")
        diff_rows.append({"dim": dim, **res})
    write_csv(OUT / "probe_differentiation.csv", diff_rows,
              ["dim", "eta2", "F", "p_value", "n", "n_perm"])

    # 1b) probe-level differentiation computed WITHIN each group (for Fig 1 a/b/c:
    # full = A+B pooled, then Group A only, then Group B only). Shows the probe
    # effect is not an artifact of group composition — it holds inside each group.
    print("\n=== (1b) probe differentiation within group (full / A / B) ===")
    bg_rows = []
    for scope, sub in (("full", p2),
                       ("A", [r for r in p2 if r["group"] == "A"]),
                       ("B", [r for r in p2 if r["group"] == "B"])):
        for dim in GRADED:
            res = perm_probe_differentiation(sub, dim, n_perm=10000)
            bg_rows.append({"scope": scope, "dim": dim, **res})
        print(f"  scope={scope:4s} n={len(sub)}")
    write_csv(OUT / "probe_differentiation_by_group.csv", bg_rows,
              ["scope", "dim", "eta2", "F", "p_value", "n", "n_perm"])

    # 2) probe x group means + per-(probe, dim) A-B gap permutation
    pg = agg_probe_group_dim(p2)
    write_csv(OUT / "probe_group_dim_pass2.csv", pg,
              ["probe", "probe_name", "group", "dim", "n", "mean", "ci_lo", "ci_hi"])

    pg_p1 = agg_probe_group_dim(p1)
    write_csv(OUT / "probe_group_dim_pass1.csv", pg_p1,
              ["probe", "probe_name", "group", "dim", "n", "mean", "ci_lo", "ci_hi"])

    print("\n=== (2) A-B gap by (probe, dim), pass2 ===")
    ab_rows = []
    for probe in PROBES:
        for dim in GRADED:
            va = [r[dim] for r in p2 if r["probe"] == probe and r["group"] == "A"]
            vb = [r[dim] for r in p2 if r["probe"] == probe and r["group"] == "B"]
            res = perm_ab_gap(va, vb, n_perm=5000)
            ab_rows.append({"probe": probe, "probe_name": PROBE_NAME[probe],
                            "dim": dim, **res})
            print(f"  {probe} {PROBE_NAME[probe]:14s} {dim:22s}  gap={res['gap']:+.3f}  p={res['p_value']:.4f}")
    write_csv(OUT / "ab_gap_by_probe_dim.csv", ab_rows,
              ["probe", "probe_name", "dim", "gap", "nA", "nB", "p_value"])

    # 3) probe x cell-axis sub-breakdowns
    for axis in ("era", "perspective", "grounding"):
        out = agg_probe_axis(p2, axis)
        write_csv(OUT / f"probe_x_{axis}_dim_pass2.csv", out,
                  ["probe", "probe_name", "axis", "level", "dim",
                   "n", "mean", "ci_lo", "ci_hi"])
    print("\n=== (3) probe x cell-axis breakdowns written for era/perspective/grounding ===")

    # 4) convention_type by probe x group + chi-sq independence by probe
    conv = agg_convention(p2)
    header = ["probe", "probe_name", "group", "n",
              "historical", "generic-genre", "mixed", "none",
              "historical_n", "generic-genre_n", "mixed_n", "none_n"]
    write_csv(OUT / "convention_by_probe_group_pass2.csv", conv, header)

    print("\n=== (4) convention_type independence by probe (pass2) ===")
    chi_rows = []
    # for each probe, test if conv distribution differs A vs B
    for probe in PROBES:
        sub = [r for r in p2 if r["probe"] == probe]
        ctA = Counter(r["convention_type"] for r in sub if r["group"] == "A")
        ctB = Counter(r["convention_type"] for r in sub if r["group"] == "B")
        table = [[ctA[c] for c in CONV], [ctB[c] for c in CONV]]
        res = chi2_pvalue_perm(table, n_perm=5000)
        chi_rows.append({"test": f"AvsB|{probe}", "probe": probe, **res})
        print(f"  A vs B | {probe} {PROBE_NAME[probe]:14s}  chi2={res['chi2']:.2f}  df={res['df']}  p={res['p_value']:.4f}")
    # And test if conv distribution differs across the 5 probes (pooled groups)
    table = [[Counter(r["convention_type"] for r in p2 if r["probe"] == probe)[c] for c in CONV]
             for probe in PROBES]
    res = chi2_pvalue_perm(table, n_perm=5000)
    chi_rows.append({"test": "across-5-probes", "probe": "all", **res})
    print(f"  across 5 probes (pooled groups)  chi2={res['chi2']:.2f}  df={res['df']}  p={res['p_value']:.4f}")
    write_csv(OUT / "convention_chi2.csv", chi_rows,
              ["test", "probe", "chi2", "df", "p_value"])

    # JSON summary
    summary = {
        "n_pass2_rows": len(p2),
        "n_pass1_rows": len(p1),
        "n_stories": len(set(r["story_id"] for r in rows)),
        "n_per_group": dict(Counter(r["group"] for r in p2)),
        "n_per_probe": dict(Counter(r["probe"] for r in p2)),
        "probe_differentiation": {r["dim"]: {"eta2": r["eta2"], "p_value": r["p_value"]}
                                  for r in diff_rows},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwritten to: {OUT}")


if __name__ == "__main__":
    main()
