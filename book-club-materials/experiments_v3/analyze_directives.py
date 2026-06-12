"""C2 analysis: pooling-strategy differentiation in consolidator directives.

Reads <run>/directive_codings/manifest.csv (produced by code_directives.py) and
computes:

  (1) variant differentiation (within pooled scope):
        pooled_neutral vs pooled_selective vs pooled_selective_blind
        on commitment / specificity / craft_vocab / conflict_resolution mix.
  (2) variant differentiation (within per-probe scope):
        per_probe neutral vs selective vs selective_blind (all three variants).
  (3) arm differentiation (per-probe only):
        A-only vs B-only vs AB-joint within each variant.
  (4) rubric-leak check at POOLED scope (selective vs selective_blind on craft_vocab):
        if selective_blind << selective, the rubric leak is real.
  (4b) rubric-leak check at PER-PROBE scope, overall and broken out by probe:
        does the leak (craft_vocab drop) and the surviving commitment hold per probe?
  (5) probe-level differentiation in directive structure (per-probe scope).
  (6) conflict_resolution mix chi-square by variant (pooled + per-probe).
  (7) edit_emphasis composition by variant, by probe, and probe x variant
        (cut/expand/voice_shift/structural/mixed/unclear) — previously coded but
        never aggregated.
  (8) [optional, needs code_directive_extras.py] invention / grounding rate:
        validity check — share of directive moves with no reader support, by variant.
  (9) [optional, needs code_directive_extras.py] craft camp (period vs generic):
        which side the directive takes, by variant and by arm (the §4.4 mechanism).

Outputs CSVs under <run>/directive_aggregates/ and a JSON summary.
"""
from __future__ import annotations
import csv, json, random, sys, time
from pathlib import Path
from collections import Counter, defaultdict

HERE = Path(__file__).resolve().parent
RUN = HERE / "run_20260531-022438"
MANIFEST = RUN / "directive_codings" / "manifest.csv"
OUT = RUN / "directive_aggregates"
OUT.mkdir(parents=True, exist_ok=True)

NUMERIC_DIMS = ("editorial_commitment", "specificity", "takeaway_count",
                "craft_vocab_count")
BINARY_DIMS = ("reader_attribution", "probe_attribution", "conflict_acknowledged")
CATEG_DIMS = ("edit_emphasis", "conflict_resolution")
RESOLUTION_CATS = ("commit", "aggregate", "punt", "none")
EMPHASIS_CATS = ("cut", "expand", "voice_shift", "structural", "mixed", "unclear")
PROBES = ("P1", "P2", "P3", "P4", "P5")
ARMS = ("A", "B", "AB", "pool20")
VARIANTS_ALL = ("neutral", "selective", "selective_blind")

RNG_SEED = 31


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def load_rows() -> list[dict]:
    rows = []
    with MANIFEST.open() as f:
        for r in csv.DictReader(f):
            if r["status"] not in ("ok", "cached"):
                continue
            for d in NUMERIC_DIMS:
                try:
                    r[d] = int(r[d]) if r[d] != "" else None
                except (TypeError, ValueError):
                    r[d] = None
            for d in BINARY_DIMS:
                try:
                    r[d] = int(r[d]) if r[d] != "" else None
                except (TypeError, ValueError):
                    r[d] = None
            rows.append(r)
    return rows


def perm_diff_two_groups(va, vb, n_perm=5000):
    """Permutation test on |mean(va) - mean(vb)|."""
    obs = mean(va) - mean(vb)
    combined = list(va) + list(vb)
    nA = len(va)
    rng = random.Random(RNG_SEED + len(va) + len(vb))
    count_ge = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        d = mean(combined[:nA]) - mean(combined[nA:])
        if abs(d) >= abs(obs):
            count_ge += 1
    return {"gap": round(obs, 3), "nA": len(va), "nB": len(vb),
            "meanA": round(mean(va), 3), "meanB": round(mean(vb), 3),
            "p_value": (count_ge + 1) / (n_perm + 1)}


def chi_squared(table):
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


def chi2_pvalue_perm(table, n_perm=5000):
    chi2_obs, df = chi_squared(table)
    rng = random.Random(RNG_SEED * 2)
    rsum = [sum(r) for r in table]
    ncols = len(table[0])
    flat = []
    for r in table:
        for c, v in enumerate(r):
            flat.extend([c] * v)
    count_ge = 0
    for _ in range(n_perm):
        rng.shuffle(flat)
        permed = []
        idx = 0
        for n in rsum:
            row = [0] * ncols
            for v in flat[idx:idx + n]:
                row[v] += 1
            idx += n
            permed.append(row)
        c2, _ = chi_squared(permed)
        if c2 >= chi2_obs:
            count_ge += 1
    return {"chi2": round(chi2_obs, 3), "df": df,
            "p_value": (count_ge + 1) / (n_perm + 1)}


def write_csv(path, rows, header):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in header})


# ============================ ANALYSES ============================

def analyze(rows):
    pooled = [r for r in rows if r["scope"] == "pooled"]
    per_probe = [r for r in rows if r["scope"] == "per_probe"]
    print(f"loaded {len(rows)} directive codings  (pooled={len(pooled)}, per_probe={len(per_probe)})")

    # (1) variant differentiation — pooled scope, on numeric dims
    print("\n=== (1) pooled-scope: variant means on each numeric dim ===")
    means_pooled = []
    for variant in VARIANTS_ALL:
        sub = [r for r in pooled if r["variant"] == variant]
        rec = {"scope": "pooled", "variant": variant, "n": len(sub)}
        for d in NUMERIC_DIMS:
            rec[d] = round(mean([r[d] for r in sub if r[d] is not None]), 3)
        means_pooled.append(rec)
        print(f"  {variant:18s} n={len(sub):3d}  "
              f"commit={rec['editorial_commitment']:.2f}  "
              f"specif={rec['specificity']:.2f}  "
              f"takeaways={rec['takeaway_count']:.2f}  "
              f"craft_vocab={rec['craft_vocab_count']:.2f}")
    write_csv(OUT / "pooled_variant_means.csv", means_pooled,
              ["scope", "variant", "n"] + list(NUMERIC_DIMS))

    # pairwise permutation tests on key dims
    print("\n=== (2) pooled-scope: pairwise variant differences ===")
    diff_rows = []
    pairs = [("neutral", "selective"),
             ("neutral", "selective_blind"),
             ("selective", "selective_blind")]
    for d in NUMERIC_DIMS:
        for va_name, vb_name in pairs:
            va_vals = [r[d] for r in pooled if r["variant"] == va_name and r[d] is not None]
            vb_vals = [r[d] for r in pooled if r["variant"] == vb_name and r[d] is not None]
            res = perm_diff_two_groups(va_vals, vb_vals)
            diff_rows.append({"scope": "pooled", "dim": d,
                              "left": va_name, "right": vb_name, **res})
            star = "*" if res["p_value"] < 0.05 else ""
            print(f"  {d:22s}  {va_name:18s} vs {vb_name:18s}  "
                  f"meanL={res['meanA']:.2f} meanR={res['meanB']:.2f}  "
                  f"gap={res['gap']:+.3f}  p={res['p_value']:.4f} {star}")
    write_csv(OUT / "pooled_pairwise_diffs.csv", diff_rows,
              ["scope", "dim", "left", "right", "meanA", "meanB",
               "gap", "nA", "nB", "p_value"])

    # (3) per-probe scope: variant means + neutral-vs-selective gap
    print("\n=== (3) per_probe-scope: variant means + neutral-vs-selective ===")
    means_pp = []
    for variant in VARIANTS_ALL:
        sub = [r for r in per_probe if r["variant"] == variant]
        rec = {"scope": "per_probe", "variant": variant, "n": len(sub)}
        for d in NUMERIC_DIMS:
            rec[d] = round(mean([r[d] for r in sub if r[d] is not None]), 3)
        means_pp.append(rec)
        print(f"  {variant:10s} n={len(sub):4d}  "
              f"commit={rec['editorial_commitment']:.2f}  "
              f"specif={rec['specificity']:.2f}  "
              f"takeaways={rec['takeaway_count']:.2f}  "
              f"craft_vocab={rec['craft_vocab_count']:.2f}")
    write_csv(OUT / "perprobe_variant_means.csv", means_pp,
              ["scope", "variant", "n"] + list(NUMERIC_DIMS))

    pp_diff_rows = []
    pp_pairs = [("neutral", "selective"),
                ("neutral", "selective_blind"),
                ("selective", "selective_blind")]
    for d in NUMERIC_DIMS:
        for va_name, vb_name in pp_pairs:
            va = [r[d] for r in per_probe if r["variant"] == va_name and r[d] is not None]
            vb = [r[d] for r in per_probe if r["variant"] == vb_name and r[d] is not None]
            res = perm_diff_two_groups(va, vb)
            pp_diff_rows.append({"scope": "per_probe", "dim": d,
                                 "left": va_name, "right": vb_name, **res})
            star = "*" if res["p_value"] < 0.05 else ""
            print(f"  {d:22s}  {va_name:15s} vs {vb_name:15s}  "
                  f"meanL={res['meanA']:.2f} meanR={res['meanB']:.2f}  "
                  f"gap={res['gap']:+.3f}  p={res['p_value']:.4f} {star}")
    write_csv(OUT / "perprobe_pairwise_diffs.csv", pp_diff_rows,
              ["scope", "dim", "left", "right", "meanA", "meanB",
               "gap", "nA", "nB", "p_value"])

    # (4) arm differentiation — per-probe scope only
    print("\n=== (4) per_probe arm differentiation, by variant ===")
    arm_means = []
    for variant in VARIANTS_ALL:
        for arm in ("A", "B", "AB"):
            sub = [r for r in per_probe if r["variant"] == variant and r["arm"] == arm]
            rec = {"scope": "per_probe", "variant": variant, "arm": arm, "n": len(sub)}
            for d in NUMERIC_DIMS:
                rec[d] = round(mean([r[d] for r in sub if r[d] is not None]), 3)
            arm_means.append(rec)
            print(f"  variant={variant:10s} arm={arm:4s} n={len(sub):4d}  "
                  f"commit={rec['editorial_commitment']:.2f}  "
                  f"takeaways={rec['takeaway_count']:.2f}  "
                  f"specif={rec['specificity']:.2f}")
    write_csv(OUT / "perprobe_arm_variant_means.csv", arm_means,
              ["scope", "variant", "arm", "n"] + list(NUMERIC_DIMS))

    # AB vs A and AB vs B permutation tests on takeaway_count, commitment
    arm_diff_rows = []
    for variant in VARIANTS_ALL:
        for L, R in (("A", "AB"), ("B", "AB"), ("A", "B")):
            for d in ("editorial_commitment", "takeaway_count"):
                va = [r[d] for r in per_probe
                      if r["variant"] == variant and r["arm"] == L and r[d] is not None]
                vb = [r[d] for r in per_probe
                      if r["variant"] == variant and r["arm"] == R and r[d] is not None]
                res = perm_diff_two_groups(va, vb)
                arm_diff_rows.append({"variant": variant, "dim": d,
                                      "left": L, "right": R, **res})
    write_csv(OUT / "perprobe_arm_pairwise.csv", arm_diff_rows,
              ["variant", "dim", "left", "right", "meanA", "meanB",
               "gap", "nA", "nB", "p_value"])

    # (5) rubric-leak check: selective vs selective_blind on craft_vocab (pooled)
    print("\n=== (5) rubric-leak check: selective vs selective_blind on craft_vocab ===")
    sel = [r["craft_vocab_count"] for r in pooled
           if r["variant"] == "selective" and r["craft_vocab_count"] is not None]
    sb = [r["craft_vocab_count"] for r in pooled
          if r["variant"] == "selective_blind" and r["craft_vocab_count"] is not None]
    res_craft = perm_diff_two_groups(sel, sb)
    print(f"  selective craft_vocab mean: {res_craft['meanA']:.2f}  n={res_craft['nA']}")
    print(f"  selective_blind craft_vocab mean: {res_craft['meanB']:.2f}  n={res_craft['nB']}")
    print(f"  difference: {res_craft['gap']:+.3f}  p={res_craft['p_value']:.4f}")

    # (5b) rubric-leak check at PER-PROBE scope (selective vs selective_blind).
    # Mirrors the pooled check above but on the per-probe consolidations. For the
    # leak to be "contained" the same way pooled is, craft_vocab should drop sharply
    # from selective -> selective_blind while editorial_commitment / specificity /
    # takeaway_count stay flat. We report it overall and broken out by probe.
    print("\n=== (5b) PER-PROBE rubric-leak: selective vs selective_blind ===")
    leak_rows = []

    def _vals(src, variant, dim):
        return [r[dim] for r in src if r["variant"] == variant and r[dim] is not None]

    # overall (all per-probe directives, both arms pooled)
    res_craft_pp = perm_diff_two_groups(_vals(per_probe, "selective", "craft_vocab_count"),
                                        _vals(per_probe, "selective_blind", "craft_vocab_count"))
    print(f"  [ALL probes] craft_vocab  selective={res_craft_pp['meanA']:.2f} "
          f"blind={res_craft_pp['meanB']:.2f}  gap={res_craft_pp['gap']:+.3f}  "
          f"p={res_craft_pp['p_value']:.4f}  (n={res_craft_pp['nA']}/{res_craft_pp['nB']})")
    for dim in ("craft_vocab_count", "editorial_commitment", "specificity", "takeaway_count"):
        res = perm_diff_two_groups(_vals(per_probe, "selective", dim),
                                   _vals(per_probe, "selective_blind", dim))
        leak_rows.append({"probe": "ALL", "dim": dim,
                          "selective": res["meanA"], "selective_blind": res["meanB"],
                          "gap": res["gap"], "nA": res["nA"], "nB": res["nB"],
                          "p_value": res["p_value"]})

    # broken out per probe
    for probe in PROBES:
        pp = [r for r in per_probe if r["probe"] == probe]
        for dim in ("craft_vocab_count", "editorial_commitment", "specificity", "takeaway_count"):
            res = perm_diff_two_groups(_vals(pp, "selective", dim),
                                       _vals(pp, "selective_blind", dim))
            leak_rows.append({"probe": probe, "dim": dim,
                              "selective": res["meanA"], "selective_blind": res["meanB"],
                              "gap": res["gap"], "nA": res["nA"], "nB": res["nB"],
                              "p_value": res["p_value"]})
        cv = next(x for x in leak_rows if x["probe"] == probe and x["dim"] == "craft_vocab_count")
        cm = next(x for x in leak_rows if x["probe"] == probe and x["dim"] == "editorial_commitment")
        star = "*" if cv["p_value"] < 0.05 else ""
        print(f"  {probe}  craft_vocab sel={cv['selective']:.2f} blind={cv['selective_blind']:.2f} "
              f"gap={cv['gap']:+.3f} p={cv['p_value']:.4f}{star}   |   "
              f"commit sel={cm['selective']:.2f} blind={cm['selective_blind']:.2f} "
              f"gap={cm['gap']:+.3f}")
    write_csv(OUT / "rubric_leak_perprobe.csv", leak_rows,
              ["probe", "dim", "selective", "selective_blind", "gap",
               "nA", "nB", "p_value"])

    # (6) probe-level differentiation (per-probe scope)
    print("\n=== (6) per_probe: probe-level differentiation in directive structure ===")
    probe_means = []
    for probe in PROBES:
        for variant in VARIANTS_ALL:
            sub = [r for r in per_probe if r["probe"] == probe and r["variant"] == variant]
            rec = {"probe": probe, "variant": variant, "n": len(sub)}
            for d in NUMERIC_DIMS:
                rec[d] = round(mean([r[d] for r in sub if r[d] is not None]), 3)
            probe_means.append(rec)
    write_csv(OUT / "perprobe_probe_variant_means.csv", probe_means,
              ["probe", "variant", "n"] + list(NUMERIC_DIMS))

    # (7) conflict_resolution mix by variant (chi-sq)
    print("\n=== (7) conflict_resolution mix by variant ===")
    resolution_rows = []
    chi_rows = []
    # pooled
    p_table = []
    for variant in VARIANTS_ALL:
        sub = [r for r in pooled if r["variant"] == variant]
        counts = Counter(r["conflict_resolution"] for r in sub)
        n = len(sub)
        rec = {"scope": "pooled", "variant": variant, "n": n}
        for cat in RESOLUTION_CATS:
            rec[cat] = round(counts[cat] / n, 3) if n else 0.0
            rec[f"{cat}_n"] = counts[cat]
        resolution_rows.append(rec)
        p_table.append([counts[c] for c in RESOLUTION_CATS])
        print(f"  pooled / {variant:18s} n={n:3d}  " +
              " ".join(f"{cat}={counts[cat]/n*100:.0f}%" for cat in RESOLUTION_CATS))
    chi = chi2_pvalue_perm(p_table)
    chi_rows.append({"scope": "pooled", "test": "conflict_resolution_x_variant", **chi})
    print(f"    chi2={chi['chi2']:.2f}  df={chi['df']}  p={chi['p_value']:.4f}")

    # per-probe (neutral vs selective)
    pp_table = []
    for variant in VARIANTS_ALL:
        sub = [r for r in per_probe if r["variant"] == variant]
        counts = Counter(r["conflict_resolution"] for r in sub)
        n = len(sub)
        rec = {"scope": "per_probe", "variant": variant, "n": n}
        for cat in RESOLUTION_CATS:
            rec[cat] = round(counts[cat] / n, 3) if n else 0.0
            rec[f"{cat}_n"] = counts[cat]
        resolution_rows.append(rec)
        pp_table.append([counts[c] for c in RESOLUTION_CATS])
        print(f"  per_probe / {variant:18s} n={n:4d}  " +
              " ".join(f"{cat}={counts[cat]/n*100:.0f}%" for cat in RESOLUTION_CATS))
    chi = chi2_pvalue_perm(pp_table)
    chi_rows.append({"scope": "per_probe", "test": "conflict_resolution_x_variant", **chi})
    print(f"    chi2={chi['chi2']:.2f}  df={chi['df']}  p={chi['p_value']:.4f}")

    header = (["scope", "variant", "n"]
              + list(RESOLUTION_CATS)
              + [f"{c}_n" for c in RESOLUTION_CATS])
    write_csv(OUT / "conflict_resolution_by_variant.csv", resolution_rows, header)
    write_csv(OUT / "conflict_resolution_chi2.csv", chi_rows,
              ["scope", "test", "chi2", "df", "p_value"])

    # (8) binary attribution rates by variant
    print("\n=== (8) attribution rates by variant ===")
    attr_rows = []
    for scope, src in (("pooled", pooled), ("per_probe", per_probe)):
        for variant in VARIANTS_ALL:
            sub = [r for r in src if r["variant"] == variant]
            n = len(sub)
            rec = {"scope": scope, "variant": variant, "n": n}
            for d in BINARY_DIMS:
                rec[d] = round(mean([r[d] for r in sub if r[d] is not None]), 3)
            attr_rows.append(rec)
            print(f"  {scope:10s} / {variant:18s} n={n:4d}  "
                  f"reader_attr={rec['reader_attribution']:.2f}  "
                  f"probe_attr={rec['probe_attribution']:.2f}  "
                  f"conflict_ack={rec['conflict_acknowledged']:.2f}")
    write_csv(OUT / "binary_dims_by_variant.csv", attr_rows,
              ["scope", "variant", "n"] + list(BINARY_DIMS))

    # (9) edit_emphasis composition — coded by code_directives.py but not previously
    # aggregated. What KIND of revision each configuration pushes (cut / expand /
    # voice_shift / structural / mixed / unclear), by variant and by probe.
    print("\n=== (9) edit_emphasis mix by variant ===")
    emph_rows = []
    emph_chi_rows = []
    for scope, src in (("pooled", pooled), ("per_probe", per_probe)):
        table = []
        present = [v for v in VARIANTS_ALL if any(r["variant"] == v for r in src)]
        for variant in present:
            sub = [r for r in src if r["variant"] == variant]
            counts = Counter(r["edit_emphasis"] for r in sub)
            n = len(sub)
            rec = {"scope": scope, "variant": variant, "n": n}
            for cat in EMPHASIS_CATS:
                rec[cat] = round(counts[cat] / n, 3) if n else 0.0
                rec[f"{cat}_n"] = counts[cat]
            emph_rows.append(rec)
            table.append([counts[c] for c in EMPHASIS_CATS])
            print(f"  {scope:10s} / {variant:18s} n={n:4d}  " +
                  " ".join(f"{cat}={counts[cat]/n*100:.0f}%" for cat in EMPHASIS_CATS if counts[cat]))
        chi = chi2_pvalue_perm(table)
        emph_chi_rows.append({"scope": scope, "test": "edit_emphasis_x_variant", **chi})
        print(f"    chi2={chi['chi2']:.2f}  df={chi['df']}  p={chi['p_value']:.4f}")

    # edit_emphasis by probe (per-probe scope, variants pooled) — does the PROBE
    # change the kind of edit, the way it changes reader attention?
    print("\n=== (9b) edit_emphasis mix by probe (per-probe scope) ===")
    probe_table = []
    for probe in PROBES:
        sub = [r for r in per_probe if r["probe"] == probe]
        counts = Counter(r["edit_emphasis"] for r in sub)
        n = len(sub)
        rec = {"scope": "per_probe", "variant": f"probe_{probe}", "n": n}
        for cat in EMPHASIS_CATS:
            rec[cat] = round(counts[cat] / n, 3) if n else 0.0
            rec[f"{cat}_n"] = counts[cat]
        emph_rows.append(rec)
        probe_table.append([counts[c] for c in EMPHASIS_CATS])
        print(f"  {probe}  n={n:4d}  " +
              " ".join(f"{cat}={counts[cat]/n*100:.0f}%" for cat in EMPHASIS_CATS if counts[cat]))
    chi = chi2_pvalue_perm(probe_table)
    emph_chi_rows.append({"scope": "per_probe", "test": "edit_emphasis_x_probe", **chi})
    print(f"    chi2={chi['chi2']:.2f}  df={chi['df']}  p={chi['p_value']:.4f}")

    emph_header = (["scope", "variant", "n"]
                   + list(EMPHASIS_CATS)
                   + [f"{c}_n" for c in EMPHASIS_CATS])
    write_csv(OUT / "edit_emphasis_mix.csv", emph_rows, emph_header)
    write_csv(OUT / "edit_emphasis_chi2.csv", emph_chi_rows,
              ["scope", "test", "chi2", "df", "p_value"])

    # (9c) edit_emphasis full cross: probe x variant (per-probe scope)
    print("\n=== (9c) edit_emphasis by probe x variant (per-probe) ===")
    emph_pv_rows = []
    for probe in PROBES:
        for variant in VARIANTS_ALL:
            sub = [r for r in per_probe if r["probe"] == probe and r["variant"] == variant]
            counts = Counter(r["edit_emphasis"] for r in sub)
            n = len(sub)
            rec = {"probe": probe, "variant": variant, "n": n}
            for cat in EMPHASIS_CATS:
                rec[cat] = round(counts[cat] / n, 3) if n else 0.0
                rec[f"{cat}_n"] = counts[cat]
            emph_pv_rows.append(rec)
            if n:
                print(f"  {probe} / {variant:18s} n={n:3d}  " +
                      " ".join(f"{cat}={counts[cat]/n*100:.0f}%"
                               for cat in EMPHASIS_CATS if counts[cat]))
    write_csv(OUT / "edit_emphasis_probe_variant.csv", emph_pv_rows,
              ["probe", "variant", "n"] + list(EMPHASIS_CATS)
              + [f"{c}_n" for c in EMPHASIS_CATS])

    # (10)+(11) RELATIONAL codings (invention + craft camp), only if
    # code_directive_extras.py has been run.
    res_inv_pp = None
    extras_manifest = RUN / "directive_extras_codings" / "manifest.csv"
    if extras_manifest.exists():
        ex = []
        with extras_manifest.open() as f:
            for r in csv.DictReader(f):
                if r["status"] not in ("ok", "cached"):
                    continue
                try:
                    r["invention_rate"] = (float(r["invention_rate"])
                                           if r["invention_rate"] != "" else None)
                except ValueError:
                    r["invention_rate"] = None
                for k in ("n_moves", "n_grounded", "n_invented", "camp_strength"):
                    try:
                        r[k] = int(r[k]) if r[k] != "" else None
                    except (TypeError, ValueError):
                        r[k] = None
                ex.append(r)
        ex_pooled = [r for r in ex if r["scope"] == "pooled"]
        ex_pp = [r for r in ex if r["scope"] == "per_probe"]

        # (10) invention / grounding — validity check
        print(f"\n=== (10) invention / grounding (validity check), n={len(ex)} ===")
        inv_rows = []
        for scope, src in (("pooled", ex_pooled), ("per_probe", ex_pp)):
            for variant in VARIANTS_ALL:
                sub = [r for r in src if r["variant"] == variant]
                n = len(sub)
                if not n:
                    continue
                rec = {"scope": scope, "variant": variant, "n": n,
                       "invention_rate": round(mean([r["invention_rate"] for r in sub
                                                     if r["invention_rate"] is not None]), 3),
                       "n_moves": round(mean([r["n_moves"] for r in sub
                                              if r["n_moves"] is not None]), 2),
                       "n_invented": round(mean([r["n_invented"] for r in sub
                                                 if r["n_invented"] is not None]), 2)}
                inv_rows.append(rec)
                print(f"  {scope:10s}/{variant:18s} n={n:4d}  "
                      f"invention_rate={rec['invention_rate']:.3f}  "
                      f"moves={rec['n_moves']:.1f}  invented={rec['n_invented']:.2f}")
        write_csv(OUT / "invention_by_variant.csv", inv_rows,
                  ["scope", "variant", "n", "invention_rate", "n_moves", "n_invented"])
        inv_diffs = []
        for scope, src in (("pooled", ex_pooled), ("per_probe", ex_pp)):
            for L, Rr in (("neutral", "selective"), ("selective", "selective_blind"),
                          ("neutral", "selective_blind")):
                va = [r["invention_rate"] for r in src
                      if r["variant"] == L and r["invention_rate"] is not None]
                vb = [r["invention_rate"] for r in src
                      if r["variant"] == Rr and r["invention_rate"] is not None]
                if not va or not vb:
                    continue
                res = perm_diff_two_groups(va, vb)
                inv_diffs.append({"scope": scope, "left": L, "right": Rr, **res})
                if scope == "per_probe" and L == "neutral" and Rr == "selective":
                    res_inv_pp = res
        write_csv(OUT / "invention_diffs.csv", inv_diffs,
                  ["scope", "left", "right", "meanA", "meanB", "gap",
                   "nA", "nB", "p_value"])

        # (11) craft camp — which side the directive takes
        print("\n=== (11) craft camp (period vs generic) by variant ===")
        CAMPS = ("period", "generic", "balanced", "none")
        camp_rows, camp_chi = [], []
        for scope, src in (("pooled", ex_pooled), ("per_probe", ex_pp)):
            table = []
            present = [v for v in VARIANTS_ALL if any(r["variant"] == v for r in src)]
            for variant in present:
                sub = [r for r in src if r["variant"] == variant]
                n = len(sub)
                counts = Counter(r["camp"] for r in sub)
                rec = {"scope": scope, "variant": variant, "n": n}
                for c in CAMPS:
                    rec[c] = round(counts[c] / n, 3) if n else 0.0
                    rec[f"{c}_n"] = counts[c]
                camp_rows.append(rec)
                table.append([counts[c] for c in CAMPS])
                print(f"  {scope:10s}/{variant:18s} n={n:4d}  " +
                      " ".join(f"{c}={counts[c]/n*100:.0f}%" for c in CAMPS if counts[c]))
            chi = chi2_pvalue_perm(table)
            camp_chi.append({"scope": scope, "test": "camp_x_variant", **chi})
            print(f"    chi2={chi['chi2']:.2f}  df={chi['df']}  p={chi['p_value']:.4f}")
        write_csv(OUT / "camp_by_variant.csv", camp_rows,
                  ["scope", "variant", "n"] + list(CAMPS) + [f"{c}_n" for c in CAMPS])

        # camp by arm (per-probe) — the §4.4 mechanism: does B-only feed -> generic?
        print("\n  -- (11b) camp by arm (per-probe), within variant --")
        camp_arm_rows = []
        for variant in VARIANTS_ALL:
            table = []
            for arm in ("A", "B", "AB"):
                sub = [r for r in ex_pp if r["variant"] == variant and r["arm"] == arm]
                n = len(sub)
                if not n:
                    continue
                counts = Counter(r["camp"] for r in sub)
                rec = {"variant": variant, "arm": arm, "n": n}
                for c in CAMPS:
                    rec[c] = round(counts[c] / n, 3) if n else 0.0
                    rec[f"{c}_n"] = counts[c]
                camp_arm_rows.append(rec)
                table.append([counts[c] for c in CAMPS])
                print(f"    {variant:18s} arm={arm:3s} n={n:3d}  " +
                      " ".join(f"{c}={counts[c]/n*100:.0f}%" for c in CAMPS if counts[c]))
            if len(table) >= 2:
                chi = chi2_pvalue_perm(table)
                camp_chi.append({"scope": f"per_probe:{variant}", "test": "camp_x_arm", **chi})
                print(f"      chi2={chi['chi2']:.2f}  df={chi['df']}  p={chi['p_value']:.4f}")
        write_csv(OUT / "camp_by_arm.csv", camp_arm_rows,
                  ["variant", "arm", "n"] + list(CAMPS) + [f"{c}_n" for c in CAMPS])
        write_csv(OUT / "camp_chi2.csv", camp_chi,
                  ["scope", "test", "chi2", "df", "p_value"])
    else:
        print("\n(10/11) extras codings not found — run code_directive_extras.py "
              "to enable invention + camp analysis")

    # summary json
    summary = {
        "n_directives_coded": len(rows),
        "n_pooled": len(pooled),
        "n_per_probe": len(per_probe),
        "headline": {
            "pooled_commit_neutral":  next(r for r in means_pooled if r["variant"] == "neutral")["editorial_commitment"],
            "pooled_commit_selective": next(r for r in means_pooled if r["variant"] == "selective")["editorial_commitment"],
            "pooled_commit_blind":    next(r for r in means_pooled if r["variant"] == "selective_blind")["editorial_commitment"],
            "pooled_craft_selective": next(r for r in means_pooled if r["variant"] == "selective")["craft_vocab_count"],
            "pooled_craft_blind":     next(r for r in means_pooled if r["variant"] == "selective_blind")["craft_vocab_count"],
            "rubric_leak_check": res_craft,
            "rubric_leak_check_perprobe": res_craft_pp,
            "invention_perprobe_neutral_vs_selective": res_inv_pp,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwritten to: {OUT}")


def main():
    rows = load_rows()
    analyze(rows)


if __name__ == "__main__":
    main()
