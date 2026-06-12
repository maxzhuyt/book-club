"""Generate the figures used in REPORT_V3_INTERMEDIATE.md.

Reads aggregated_differentiation/* (reader-response codings) and
directive_aggregates/* (directive codings), writes PNGs to
report_v3/figures_intermediate/.

Design standard (applied uniformly):
  * 200 dpi, white background
  * Okabe–Ito colorblind-safe palette; consistent color meaning across figures
    (period/historical = blue, generic = vermillion, etc.)
  * legible fonts (>=11.5pt), plain-English axis labels (no coder jargon)
  * concise, informative titles: one bold message line + one light stat line
  * constrained layout (no overlapping titles / legends / ticks)

Run with a python that has matplotlib + numpy:
    /Users/maxzhu/anaconda3/bin/python make_intermediate_figures.py
(On Midway use any venv with matplotlib + numpy.)
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / "run_20260531-022438"
RD = RUN / "aggregated_differentiation"
DD = RUN / "directive_aggregates"
FIG = HERE / "report_v3" / "figures_intermediate"
FIG.mkdir(parents=True, exist_ok=True)

# ----------------------------- shared style -----------------------------

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 13,
    "axes.titlesize": 13.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 11.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

INK = "#222222"
SUB = "#555555"

# Okabe–Ito based, with stable semantic meaning across the report
C_BLUE = "#0072B2"      # period / historical / "A" readers / SELECTIVE
C_SKY = "#56B4E9"       # SELECTIVE-BLIND (selective minus the rubric)
C_ORANGE = "#E69F00"    # "B" readers / punt
C_VERM = "#D55E00"      # generic / cut
C_GREEN = "#009E73"     # balanced / aggregate / expand
C_PURPLE = "#CC79A7"    # structural
C_GREY = "#9b9b9b"      # NEUTRAL
C_GREYL = "#cfcfcf"     # none / mixed-neutral

GROUP_COLOR = {"A": C_BLUE, "B": C_ORANGE}
GROUP_LBL = {"A": "A — historical-fiction readers", "B": "B — controls"}

VAR_ORDER = ["neutral", "selective", "selective_blind"]
VAR_COLOR = {"neutral": C_GREY, "selective": C_BLUE, "selective_blind": C_SKY}
VAR_LBL = {"neutral": "NEUTRAL", "selective": "SELECTIVE", "selective_blind": "SELECTIVE-BLIND"}
# two-line form for x-axis tick labels (never tilt; wrap instead)
VAR_LBL2 = {"neutral": "NEUTRAL", "selective": "SELECTIVE", "selective_blind": "SELECTIVE-\nBLIND"}
# SELECTIVE-BLIND is the default committing consolidator; SELECTIVE appears ONLY in the
# rubric-leak validation figures. Everywhere else we contrast NEUTRAL vs SELECTIVE-BLIND.
VAR2 = ["neutral", "selective_blind"]

CONV = ("historical", "generic-genre", "mixed", "none")
CONV_COLOR = {"historical": C_BLUE, "generic-genre": C_VERM, "mixed": C_GREEN, "none": C_GREYL}
CONV_LBL = {"historical": "historical", "generic-genre": "generic-genre",
            "mixed": "mixed", "none": "none"}

CAMP = ("period", "generic", "balanced", "none")
CAMP_COLOR = {"period": C_BLUE, "generic": C_VERM, "balanced": C_GREEN, "none": C_GREYL}
CAMP_LBL = {"period": "period (historical craft)", "generic": "generic (story craft)",
            "balanced": "balanced", "none": "none"}

EMPH = ("cut", "expand", "voice_shift", "structural", "mixed", "unclear")
EMPH_COLOR = {"cut": C_VERM, "expand": C_GREEN, "voice_shift": C_BLUE,
              "structural": C_PURPLE, "mixed": C_GREYL, "unclear": "#7f7f7f"}
EMPH_LBL = {"cut": "cut (compress)", "expand": "expand (add)", "voice_shift": "voice shift",
            "structural": "structural", "mixed": "mixed", "unclear": "unclear"}

RES = ("commit", "aggregate", "punt", "none")
RES_COLOR = {"commit": C_BLUE, "aggregate": C_GREEN, "punt": C_ORANGE, "none": C_GREYL}

PROBES = ("P1", "P2", "P3", "P4", "P5")
PROBE_NAME = {"P1": "Plausibility", "P2": "Knowledge-gap", "P3": "Stability",
              "P4": "Convention", "P5": "Salience"}
PROBE_KEY = "  ·  ".join(f"{p} {PROBE_NAME[p]}" for p in PROBES)

DIMS = ("period_specificity", "concreteness", "knowledge_invoked",
        "locations_cited", "anchors")
DIM_FRIENDLY = {"period_specificity": "period specificity", "concreteness": "concreteness",
                "knowledge_invoked": "outside knowledge", "locations_cited": "locations cited",
                "anchors": "historical anchors"}
DIM_2LINE = {"period_specificity": "period\nspecificity", "concreteness": "concreteness",
             "knowledge_invoked": "outside\nknowledge", "locations_cited": "locations\ncited",
             "anchors": "historical\nanchors"}


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def titles(fig, main, sub=None, top=0.88, bottom=0.0, main_size=15):
    """Bold message line + optional light stat line, placed in a reserved band at
    the TOP so they never collide with axis tick labels. `bottom` reserves a band
    at the foot of the figure for a fig.legend / below-axis labels."""
    try:
        fig.get_layout_engine().set(rect=(0.0, bottom, 1.0, top - bottom))
    except Exception:
        pass
    fig.suptitle(main, fontsize=main_size, fontweight="bold", color=INK, y=0.992, va="top")
    if sub:
        fig.text(0.5, top + (1.0 - top) * 0.30, sub, ha="center", va="center",
                 fontsize=11, color=SUB)


def save(fig, name):
    out = FIG / name
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out.name}")


def stacked(ax, xpos, fracs_by_cat, cats, colors, width=0.62, min_lbl=0.08):
    bottom = np.zeros(len(xpos))
    for c in cats:
        vals = np.array([fracs_by_cat[i].get(c, 0.0) for i in range(len(xpos))])
        ax.bar(xpos, vals, width, bottom=bottom, color=colors[c],
               edgecolor="white", linewidth=0.8)
        for i, v in enumerate(vals):
            if v >= min_lbl:
                ax.text(xpos[i], bottom[i] + v / 2, f"{round(v*100)}%",
                        ha="center", va="center", color="white",
                        fontsize=10.5, fontweight="bold")
        bottom += vals
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])


# ============================ C1: PROBES × GROUPS ============================

def fig_eta2():
    """Three shared-axis panels: probe η² for the full sample, then within each
    group (a/b/c). Horizontal bars with a shared y-axis put the dimension labels
    on the left panel only — an efficient use of space."""
    by = {(r["scope"], r["dim"]): fnum(r["eta2"])
          for r in read_csv(RD / "probe_differentiation_by_group.csv")}
    order = sorted(DIMS, key=lambda d: by[("full", d)], reverse=True)  # biggest on top
    xmax = max(by.values()) * 1.25
    panels = [("full", "(a)  Full  (A + B) · n = 720", C_GREY),
              ("A",    "(b)  Group A · n = 360",        GROUP_COLOR["A"]),
              ("B",    "(c)  Group B · n = 360",        GROUP_COLOR["B"])]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9), layout="constrained", sharey=True)
    y = np.arange(len(order))
    for ax, (scope, title, color) in zip(axes, panels):
        vals = [by[(scope, d)] for d in order]
        ax.barh(y, vals, color=color, height=0.66)
        ax.set_yticks(y)
        ax.set_yticklabels([DIM_FRIENDLY[d] for d in order])
        ax.set_xlim(0, xmax)
        ax.set_xlabel("η²  (variance explained by probe)")
        ax.set_title(title, fontsize=12.5)
        ax.grid(axis="x", alpha=0.25)
        for i, v in enumerate(vals):
            ax.text(v + xmax * 0.02, i, f"{v:.2f}", va="center", fontsize=10.5, color=INK)
    axes[0].invert_yaxis()  # shared axis → flips all three; largest dimension on top
    titles(fig, "Probes shape what readers notice",
           "η² = share of an attention dimension's variance explained by probe · "
           "one-way permutation test (10k) · every bar p ≤ 0.001", main_size=16)
    save(fig, "fig_c1_eta2.png")


def fig_probe_dim_heatmap():
    rows = read_csv(RD / "probe_group_dim_pass2.csv")
    by = defaultdict(list)
    for r in rows:
        by[(r["probe"], r["dim"])].append((fnum(r["mean"]), fnum(r["n"])))
    mat = np.zeros((len(PROBES), len(DIMS)))
    for i, p in enumerate(PROBES):
        for j, d in enumerate(DIMS):
            vals = by[(p, d)]
            tot = sum(n for _, n in vals)
            mat[i, j] = sum(m * n for m, n in vals) / tot if tot else 0
    # The five dimensions are two different units, so they get two panels with
    # their own colour scales: 0–5 rubric ratings (left) vs mean counts per
    # response (right). Putting them on one shared scale would be misleading
    # (a count like "anchors=5.62" is not a 5-out-of-5 rating).
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11.0, 4.8), layout="constrained",
        gridspec_kw={"width_ratios": [3, 2]})
    score_dims, count_dims = DIMS[:3], DIMS[3:]

    matL = mat[:, :3]
    axL.imshow(matL, cmap="Blues", aspect="auto", vmin=0, vmax=5)
    axL.set_xticks(range(3)); axL.set_xticklabels([DIM_2LINE[d] for d in score_dims])
    axL.set_yticks(range(5)); axL.set_yticklabels([f"{p}  {PROBE_NAME[p]}" for p in PROBES])
    axL.set_title("mean rating  (0–5 rubric)", fontsize=13)
    for i in range(5):
        for j in range(3):
            v = matL[i, j]
            axL.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if v > 3.5 else INK, fontsize=11.5)

    matR = mat[:, 3:]
    vmaxR = matR.max()
    axR.imshow(matR, cmap="Blues", aspect="auto", vmin=0, vmax=vmaxR)
    axR.set_xticks(range(2)); axR.set_xticklabels([DIM_2LINE[d] for d in count_dims])
    axR.set_yticks([])
    axR.set_title("mean count per response", fontsize=13)
    for i in range(5):
        for j in range(2):
            v = matR[i, j]
            axR.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if v > vmaxR * 0.62 else INK, fontsize=11.5)

    titles(fig, "Each probe elicits a distinct attention profile",
           "Mean over the 144 responses per probe (720 codings) · "
           "left: 0–5 rubric ratings · right: mean count of items named per response")
    save(fig, "fig_c1_probe_dim_heatmap.png")


def fig_probe_group_dim():
    """2×3 grid (5 dims + a legend/key cell): A vs B by probe, with significance marks."""
    rows = read_csv(RD / "probe_group_dim_pass2.csv")
    gap = {(r["probe"], r["dim"]): fnum(r["p_value"])
           for r in read_csv(RD / "ab_gap_by_probe_dim.csv")}
    by = defaultdict(dict)
    for r in rows:
        by[(r["probe"], r["dim"])][r["group"]] = (fnum(r["mean"]), fnum(r["ci_lo"]), fnum(r["ci_hi"]))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), layout="constrained")
    axes = axes.ravel()
    x = np.arange(len(PROBES))
    w = 0.38
    for k, d in enumerate(DIMS):
        ax = axes[k]
        mA = [by[(p, d)].get("A", (0, 0, 0))[0] for p in PROBES]
        mB = [by[(p, d)].get("B", (0, 0, 0))[0] for p in PROBES]
        eA = [[mA[i] - by[(PROBES[i], d)].get("A", (0, 0, 0))[1] for i in range(5)],
              [by[(PROBES[i], d)].get("A", (0, 0, 0))[2] - mA[i] for i in range(5)]]
        eB = [[mB[i] - by[(PROBES[i], d)].get("B", (0, 0, 0))[1] for i in range(5)],
              [by[(PROBES[i], d)].get("B", (0, 0, 0))[2] - mB[i] for i in range(5)]]
        ax.bar(x - w/2, mA, w, yerr=eA, capsize=2, color=GROUP_COLOR["A"],
               error_kw=dict(lw=1, ecolor="#444"))
        ax.bar(x + w/2, mB, w, yerr=eB, capsize=2, color=GROUP_COLOR["B"],
               error_kw=dict(lw=1, ecolor="#444"))
        ax.set_xticks(x)
        ax.set_xticklabels(PROBES)
        ax.set_title(DIM_FRIENDLY[d], fontsize=15)
        ax.grid(axis="y", alpha=0.25)
        datamax = max(max(mA[i] + eA[1][i] for i in range(5)),
                      max(mB[i] + eB[1][i] for i in range(5)))
        ytop = datamax * 1.34 + 0.1
        ax.set_ylim(0, ytop)
        for i in range(5):
            if gap.get((PROBES[i], d), 1) < 0.05:
                yb = max(mA[i] + eA[1][i], mB[i] + eB[1][i]) + ytop * 0.05
                xa, xb = i - w / 2, i + w / 2
                ax.plot([xa, xa, xb, xb], [yb, yb + ytop * 0.02, yb + ytop * 0.02, yb],
                        lw=1.4, color="#333333", clip_on=False, zorder=6)
                ax.scatter([i], [yb + ytop * 0.085], marker="*", s=200, color=C_VERM,
                           edgecolor="white", linewidth=0.6, clip_on=False, zorder=7)
    # legend / key in the 6th cell
    lg = axes[5]
    lg.axis("off")
    handles = [Patch(facecolor=GROUP_COLOR["A"], label=GROUP_LBL["A"]),
               Patch(facecolor=GROUP_COLOR["B"], label=GROUP_LBL["B"]),
               plt.Line2D([0], [0], marker="*", color="none", markerfacecolor=C_VERM,
                          markeredgecolor="white", markersize=20,
                          label="A vs B differ (p < .05)")]
    lg.legend(handles=handles, loc="upper center", frameon=False, fontsize=12,
              handlelength=1.4, borderaxespad=0.2)
    lg.text(0.5, 0.40, PROBE_KEY, ha="center", va="top", fontsize=10.5, color=SUB, wrap=True)
    fig.supylabel("mean per probed response", fontsize=12)
    titles(fig, "HF-reader 'disposition' surfaces on the control probes (P3, P5)",
           "A vs B by probe, each attention dimension · 95% CIs · "
           "vermillion star + bracket = significant A–B gap (5,000-perm)", main_size=18)
    save(fig, "fig_c1_probe_group_dim.png")


def fig_convention_type():
    rows = read_csv(RD / "convention_by_probe_group_pass2.csv")
    by = {(r["probe"], r["group"]): {c: fnum(r[c]) for c in CONV} for r in rows}
    fig, ax = plt.subplots(figsize=(11.0, 4.8), layout="constrained")
    xpos, labels, fracs = [], [], []
    for i, p in enumerate(PROBES):
        for j, g in enumerate(("A", "B")):
            xpos.append(i * 1.5 + j * 0.62)
            labels.append(g)
            fracs.append(by[(p, g)])
    xpos = np.array(xpos)
    stacked(ax, xpos, fracs, CONV, CONV_COLOR, width=0.58)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("share of probed responses")
    for i, p in enumerate(PROBES):
        ax.text(i * 1.5 + 0.31, -0.12, f"{p}  {PROBE_NAME[p]}", ha="center", va="top",
                fontsize=11, color=INK, fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_xlim(-0.5, xpos[-1] + 0.5)
    titles(fig, "Group A imports 'historical' framing even on the control probes",
           "Convention type readers invoked · blue = historical, vermillion = generic-genre · "
           "B's generic-genre signature spikes on P3 & P5", top=0.84, bottom=0.16)
    handles = [Patch(facecolor=CONV_COLOR[c], label=CONV_LBL[c]) for c in CONV]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=4, frameon=False)
    save(fig, "fig_c1_convention_type.png")


# ============================ C2: CONSOLIDATOR ============================

def fig_pooled_variant_means():
    rows = {r["variant"]: r for r in read_csv(DD / "pooled_variant_means.csv")}
    dims = ("editorial_commitment", "specificity", "takeaway_count")
    lbls = ("editorial commitment\n(0–5)", "specificity\n(0–5)", "takeaways\n(count)")
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.2), layout="constrained")
    for k, d in enumerate(dims):
        ax = axes[k]
        vals = [fnum(rows[v][d]) for v in VAR2]
        ax.bar(range(len(VAR2)), vals, width=0.6, color=[VAR_COLOR[v] for v in VAR2])
        ax.set_xticks(range(len(VAR2)))
        ax.set_xticklabels([VAR_LBL2[v] for v in VAR2], fontsize=11)
        ax.set_title(lbls[k], fontsize=12)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(vals) * 1.2)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.03, f"{v:.2f}", ha="center", fontsize=11)
    titles(fig, "The default consolidator commits where NEUTRAL aggregates",
           "Pooled directives, NEUTRAL vs SELECTIVE-BLIND (the default) · n = 72")
    save(fig, "fig_c2_pooled_variant_means.png")


def fig_resolution_mix():
    rows = read_csv(DD / "conflict_resolution_by_variant.csv")
    order = [("pooled", v) for v in VAR2] + [("per_probe", v) for v in VAR2]
    d = {(r["scope"], r["variant"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(9.0, 4.6), layout="constrained")
    xpos = np.array([0, 1, 2.4, 3.4])
    fracs = [{c: fnum(d[k][c]) for c in RES} for k in order]
    stacked(ax, xpos, fracs, RES, RES_COLOR, width=0.74)
    ax.set_xticks(xpos)
    ax.set_xticklabels([VAR_LBL2[v] for _, v in order])
    ax.set_ylabel("share of directives")
    ax.text(0.5, 1.05, "POOLED  (n=36 each)", ha="center", fontsize=11, color=SUB, fontweight="bold")
    ax.text(2.9, 1.05, "PER-PROBE  (n=540 each)", ha="center", fontsize=11, color=SUB, fontweight="bold")
    ax.set_ylim(0, 1.13)
    ax.set_xlim(-0.6, 4.0)
    titles(fig, "When readers disagree: the default consolidator commits, NEUTRAL punts",
           "How each directive resolves reader conflict, by consolidator and scope",
           top=0.86, bottom=0.13)
    handles = [Patch(facecolor=RES_COLOR[c], label=c) for c in RES]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=4, frameon=False)
    save(fig, "fig_c2_resolution_mix.png")


def fig_perprobe_arm_variant():
    rows = read_csv(DD / "perprobe_arm_variant_means.csv")
    arms = ("A", "B", "AB")
    by = {(r["variant"], r["arm"]): r for r in rows}
    dims = ("editorial_commitment", "takeaway_count")
    lbls = ("editorial commitment (0–5)", "takeaways (count)")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), layout="constrained")
    x = np.arange(len(arms))
    w = 0.36
    for k, d in enumerate(dims):
        ax = axes[k]
        top = 0
        for j, v in enumerate(VAR2):
            vals = [fnum(by[(v, a)][d]) for a in arms]
            ax.bar(x + (j - 0.5) * w, vals, w, color=VAR_COLOR[v], label=VAR_LBL[v])
            top = max(top, max(vals))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a}-arm" for a in arms])
        ax.set_title(lbls[k], fontsize=12)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, top * 1.28)  # headroom so the legend never sits on a bar
    axes[0].legend(frameon=False, fontsize=11, loc="upper left", borderaxespad=0.4)
    titles(fig, "Wider reader pool (AB) adds takeaways without diluting commitment",
           "Per-probe directives by reader arm · NEUTRAL vs SELECTIVE-BLIND · A = 2 readers, B = 2, AB = 4")
    save(fig, "fig_c2_perprobe_arm_variant.png")


def fig_rubric_leak():
    res = json.loads((DD / "summary.json").read_text())["headline"]["rubric_leak_check"]
    means = [res["meanA"], res["meanB"]]
    drop = 100 * (means[0] - means[1]) / means[0] if means[0] else 0
    fig, ax = plt.subplots(figsize=(5.8, 4.2), layout="constrained")
    ax.bar(["SELECTIVE", "SELECTIVE-BLIND"], means, color=[C_BLUE, C_SKY], width=0.6)
    for i, m in enumerate(means):
        ax.text(i, m + max(means) * 0.03, f"{m:.2f}", ha="center", fontsize=13, fontweight="bold")
    ax.set_ylabel("CRAFT_GUIDE vocab per directive")
    ax.set_ylim(0, max(means) * 1.2)
    ax.grid(axis="y", alpha=0.25)
    titles(fig, f"Stripping the rubric cuts in-output vocab by {drop:.0f}%",
           f"Pooled directives (n = 36 each) · permutation p = {res['p_value']:.4f}")
    save(fig, "fig_c2_rubric_leak.png")


def fig_rubric_leak_perprobe():
    """NEW: per-probe rubric leak — vocab collapses, commitment unchanged."""
    rows = read_csv(DD / "rubric_leak_perprobe.csv")
    d = {(r["probe"], r["dim"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), layout="constrained")
    x = np.arange(len(PROBES))
    w = 0.38
    # left: craft vocab collapses
    ax = axes[0]
    sel = [fnum(d[(p, "craft_vocab_count")]["selective"]) for p in PROBES]
    bli = [fnum(d[(p, "craft_vocab_count")]["selective_blind"]) for p in PROBES]
    ax.bar(x - w/2, sel, w, color=C_BLUE, label="SELECTIVE")
    ax.bar(x + w/2, bli, w, color=C_SKY, label="SELECTIVE-BLIND")
    ax.set_xticks(x); ax.set_xticklabels(PROBES)
    ax.set_ylabel("occurrences per directive")
    ax.set_title("CRAFT_GUIDE vocabulary  —  collapses", fontsize=12.5)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=11)
    ax.set_ylim(0, max(sel) * 1.18)
    # right: commitment unchanged
    ax = axes[1]
    sel = [fnum(d[(p, "editorial_commitment")]["selective"]) for p in PROBES]
    bli = [fnum(d[(p, "editorial_commitment")]["selective_blind"]) for p in PROBES]
    ax.bar(x - w/2, sel, w, color=C_BLUE, label="SELECTIVE")
    ax.bar(x + w/2, bli, w, color=C_SKY, label="SELECTIVE-BLIND")
    ax.set_xticks(x); ax.set_xticklabels(PROBES)
    ax.set_ylabel("editorial commitment (0–5)")
    ax.set_title("Editorial commitment  —  unchanged", fontsize=12.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 5.6)
    titles(fig, "Stripping the rubric removes the vocabulary, not the commitment",
           "Per-probe SELECTIVE vs SELECTIVE-BLIND (n = 108/probe each) · vocab −81%, commitment flat (5.0 = 5.0)")
    save(fig, "fig_c2_rubric_leak_perprobe.png")


def fig_edit_emphasis():
    """NEW: what kind of edit — by consolidator (NEUTRAL vs default) and by probe
    under the default consolidator (SELECTIVE-BLIND)."""
    by = {(r["scope"], r["variant"]): r for r in read_csv(DD / "edit_emphasis_mix.csv")}
    pv = {(r["probe"], r["variant"]): r for r in read_csv(DD / "edit_emphasis_probe_variant.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), layout="constrained")
    # left: by variant (per-probe) — NEUTRAL vs SELECTIVE-BLIND
    ax = axes[0]
    xpos = np.arange(len(VAR2))
    fr = [{c: fnum(by[("per_probe", v)][c]) for c in EMPH} for v in VAR2]
    stacked(ax, xpos, fr, EMPH, EMPH_COLOR, width=0.55, min_lbl=0.06)
    ax.set_xticks(xpos); ax.set_xticklabels([VAR_LBL2[v] for v in VAR2], fontsize=11)
    ax.set_ylabel("share of directives")
    ax.set_title("By consolidator  —  committing means cutting", fontsize=12.5)
    # right: by probe, under the default consolidator (SELECTIVE-BLIND)
    ax = axes[1]
    xpos = np.arange(5)
    fr = [{c: fnum(pv[(p, "selective_blind")][c]) for c in EMPH} for p in PROBES]
    stacked(ax, xpos, fr, EMPH, EMPH_COLOR, width=0.66, min_lbl=0.06)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{p}\n{PROBE_NAME[p]}" for p in PROBES], fontsize=10.5)
    ax.set_title("By probe (default consolidator)  —  each favors a different operation", fontsize=12)
    handles = [Patch(facecolor=EMPH_COLOR[c], label=EMPH_LBL[c]) for c in EMPH]
    titles(fig, "What kind of edit the directive pushes",
           "Per-probe directives · left: NEUTRAL vs SELECTIVE-BLIND · right: SELECTIVE-BLIND by probe",
           top=0.88, bottom=0.15)
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.02))
    save(fig, "fig_c2_edit_emphasis.png")


def fig_invention():
    """NEW: validity check — invention low, the default consolidator makes more
    (grounded) moves than NEUTRAL."""
    by = {(r["scope"], r["variant"]): r for r in read_csv(DD / "invention_by_variant.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained")
    x = np.arange(len(VAR2))
    w = 0.36
    scopes = (("per_probe", "per-probe"), ("pooled", "pooled"))
    # left: invention rate
    ax = axes[0]
    for j, (sc, lbl) in enumerate(scopes):
        vals = [fnum(by[(sc, v)]["invention_rate"]) * 100 for v in VAR2]
        bars = ax.bar(x + (j - 0.5) * w, vals, w,
                      color=[VAR_COLOR[v] for v in VAR2],
                      alpha=1.0 if j == 0 else 0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.0f}%", ha="center", fontsize=10.5)
    ax.set_xticks(x); ax.set_xticklabels([VAR_LBL2[v] for v in VAR2], fontsize=11)
    ax.set_ylabel("invented moves (%)")
    ax.set_ylim(0, 26)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Invention rate stays low (≈ 90% grounded)", fontsize=12)
    ax.text(0.02, 0.97, "solid = per-probe · faded = pooled", transform=ax.transAxes,
            fontsize=10, color=SUB, va="top")
    # right: n_moves
    ax = axes[1]
    vals = [fnum(by[("per_probe", v)]["n_moves"]) for v in VAR2]
    ax.bar(x, vals, 0.6, color=[VAR_COLOR[v] for v in VAR2])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.12, f"{v:.1f}", ha="center", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels([VAR_LBL2[v] for v in VAR2], fontsize=11)
    ax.set_ylabel("editorial moves per directive")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("The default makes ~2× the moves", fontsize=12)
    titles(fig, "Validity check: the default consolidator's commitment is reader-grounded",
           "SELECTIVE-BLIND makes ~2× NEUTRAL's moves, still ~90% grounded — extra moves are signal NEUTRAL punts")
    save(fig, "fig_c2_invention.png")


def fig_camp():
    """NEW: which craft camp the directive takes; arm doesn't move it (§4.4)."""
    cv = {(r["scope"], r["variant"]): r for r in read_csv(DD / "camp_by_variant.csv")}
    ca = {(r["variant"], r["arm"]): r for r in read_csv(DD / "camp_by_arm.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    # left: by variant (per-probe) — all three: shows the rubric tilts the camp
    # (SELECTIVE vs BLIND), which is leak-adjacent and feeds the §4.4 discussion.
    ax = axes[0]
    xpos = np.arange(len(VAR_ORDER))
    fr = [{c: fnum(cv[("per_probe", v)][c]) for c in CAMP} for v in VAR_ORDER]
    stacked(ax, xpos, fr, CAMP, CAMP_COLOR, width=0.62, min_lbl=0.07)
    ax.set_xticks(xpos); ax.set_xticklabels([VAR_LBL2[v] for v in VAR_ORDER], fontsize=10.5)
    ax.set_ylabel("share of directives")
    ax.set_title("By consolidator  —  SELECTIVE picks period", fontsize=12.5)
    # right: by arm under SELECTIVE (the B-wins phenomenon is SELECTIVE-specific, §4.4)
    ax = axes[1]
    arms = ("A", "B", "AB")
    xpos = np.arange(3)
    fr = [{c: fnum(ca[("selective", a)][c]) for c in CAMP} for a in arms]
    stacked(ax, xpos, fr, CAMP, CAMP_COLOR, width=0.6, min_lbl=0.07)
    ax.set_xticks(xpos); ax.set_xticklabels([f"{a}-arm" for a in arms], fontsize=11)
    ax.set_title("Under SELECTIVE  —  arm doesn't move it (n.s.)", fontsize=12.5)
    handles = [Patch(facecolor=CAMP_COLOR[c], label=CAMP_LBL[c]) for c in CAMP]
    titles(fig, "Which craft camp the editor takes — and why it's not the B-wins story",
           "SELECTIVE picks period 3× more than NEUTRAL; under SELECTIVE the reader arm doesn't "
           "shift it (n.s.) — rules out one §4.4 story",
           top=0.85, bottom=0.15)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.02))
    save(fig, "fig_c2_camp.png")


def main():
    print("=== C1: probes × groups ===")
    fig_eta2()
    fig_probe_dim_heatmap()
    fig_probe_group_dim()
    fig_convention_type()
    print("\n=== C2: consolidator ===")
    fig_pooled_variant_means()
    fig_resolution_mix()
    fig_perprobe_arm_variant()
    fig_rubric_leak()
    fig_rubric_leak_perprobe()
    fig_edit_emphasis()
    fig_invention()
    fig_camp()
    print(f"\nAll figures written to {FIG}")


if __name__ == "__main__":
    main()
