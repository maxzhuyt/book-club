"""Aggregate V2: per-probe group means on each attention dimension (A vs B, pass-2),
the A-B gap (Stability is the priors-independent control), convention_type proportions
(historical vs generic-genre), pass1->pass2 lift, and figures."""
from __future__ import annotations
import json, sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import probes

NUM_DIMS = ["period_specificity", "concreteness", "knowledge_invoked", "locations_cited", "anchors"]
GRADED = ["period_specificity", "concreteness", "knowledge_invoked"]  # 0-5
PROBE_NAME = {k: v["name"] for k, v in probes.PROBES.items()}
ORDER = list(probes.PROBES.keys())  # P1..P5
CONV_LABELS = ["historical", "generic-genre", "mixed", "none"]


def load_rows(run_dir):
    rows = []
    for d in glob.glob(f"{run_dir}/results/*/*/*/agent-*"):
        mp, cp = Path(d) / "meta.json", Path(d) / "coding.json"
        if not (mp.exists() and cp.exists()):
            continue
        m = json.loads(mp.read_text())
        if m.get("status") != "ok":
            continue
        c = json.loads(cp.read_text())
        for phase in ("pass1", "pass2"):
            cc = c.get(phase, {})
            r = {"group": m["group"], "cell": m["cell"], "probe": m["probe"], "phase": phase,
                 "convention_type": cc.get("convention_type", "none")}
            for dim in NUM_DIMS:
                v = cc.get(dim, np.nan)
                r[dim] = v if isinstance(v, (int, float)) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(glob.glob(str(HERE / "run_*")))[-1]
    df = load_rows(run_dir)
    p2 = df[df.phase == "pass2"]
    aggdir = Path(run_dir) / "aggregated"; aggdir.mkdir(exist_ok=True)

    # (1) per-probe group means + (2) A-B gaps
    agg = p2.groupby(["probe", "group"])[NUM_DIMS].mean().round(2)
    agg.to_csv(aggdir / "by_probe_group.csv")
    gaps = []
    for pk in ORDER:
        for dim in NUM_DIMS:
            a = p2[(p2.probe == pk) & (p2.group == "A")][dim].mean()
            b = p2[(p2.probe == pk) & (p2.group == "B")][dim].mean()
            gaps.append({"probe": pk, "name": PROBE_NAME[pk], "type": probes.PROBES[pk]["type"],
                         "dim": dim, "A": round(a, 2), "B": round(b, 2),
                         "gap_A_minus_B": round(a - b, 2)})
    gdf = pd.DataFrame(gaps); gdf.to_csv(aggdir / "ab_gaps.csv", index=False)

    # (3) convention_type proportions per probe x group
    conv = (p2.groupby(["probe", "group"])["convention_type"]
            .value_counts(normalize=True).unstack(fill_value=0.0).round(2))
    for lab in CONV_LABELS:
        if lab not in conv.columns:
            conv[lab] = 0.0
    conv = conv[CONV_LABELS]
    conv.to_csv(aggdir / "convention_type.csv")

    # (4) pass1 -> pass2 lift (both groups pooled)
    lift = []
    for pk in ORDER:
        for dim in NUM_DIMS:
            p1 = df[(df.probe == pk) & (df.phase == "pass1")][dim].mean()
            p2m = df[(df.probe == pk) & (df.phase == "pass2")][dim].mean()
            lift.append({"probe": pk, "name": PROBE_NAME[pk], "dim": dim,
                         "pass1": round(p1, 2), "pass2": round(p2m, 2),
                         "lift": round(p2m - p1, 2)})
    pd.DataFrame(lift).to_csv(aggdir / "elicitation_lift.csv", index=False)

    # console summary
    print("=== A-B gap (priors-dependent dims: period_specificity+knowledge_invoked+anchors) ===")
    key = ["period_specificity", "knowledge_invoked", "anchors"]
    for pk in ORDER:
        s = sum(gdf[(gdf.probe == pk) & (gdf.dim == d)]["gap_A_minus_B"].iloc[0] for d in key)
        print(f"  {PROBE_NAME[pk]:16} ({probes.PROBES[pk]['type']:20}) richer-for-A by {s:+.1f}")
    print("\n=== convention_type: share 'historical' vs 'generic-genre' (pass-2) ===")
    for pk in ORDER:
        for g in ("A", "B"):
            try:
                row = conv.loc[(pk, g)]
                print(f"  {PROBE_NAME[pk]:16} {g}: historical={row['historical']:.2f} "
                      f"generic-genre={row['generic-genre']:.2f} mixed={row['mixed']:.2f}")
            except KeyError:
                pass

    # Figure 1: A vs B grouped bars on headline dims (0-5 graded)
    sns.set_theme(style="whitegrid", context="talk")
    head = ["period_specificity", "knowledge_invoked", "anchors"]
    fig, axes = plt.subplots(1, len(head), figsize=(6.2 * len(head), 6))
    xlab = [PROBE_NAME[k] for k in ORDER]
    for ax, dim in zip(axes, head):
        A = [p2[(p2.probe == k) & (p2.group == "A")][dim].mean() for k in ORDER]
        B = [p2[(p2.probe == k) & (p2.group == "B")][dim].mean() for k in ORDER]
        x = np.arange(len(ORDER)); w = 0.38
        ax.bar(x - w / 2, A, w, label="Group A (HF readers)", color="#2a6f9e")
        ax.bar(x + w / 2, B, w, label="Group B (control)", color="#d1813d")
        ax.set_xticks(x); ax.set_xticklabels(xlab, rotation=20, ha="right")
        ax.set_title(dim.replace("_", " "))
        if dim in GRADED:
            ax.set_ylim(0, 5)
    axes[0].set_ylabel("mean (pass-2)")
    axes[-1].legend(loc="upper right", fontsize=13)
    figdir = HERE / "report_v2" / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(figdir / "fig_ab_attention.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: convention_type composition per group, on the Convention probe (P4)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    groups = ["A", "B"]; ypos = np.arange(len(groups))[::-1]
    colors = {"historical": "#2a6f9e", "generic-genre": "#d1813d",
              "mixed": "#9a8cbf", "none": "#cccccc"}
    left = np.zeros(len(groups))
    for lab in CONV_LABELS:
        vals = []
        for g in groups:
            try:
                vals.append(100 * conv.loc[("P4", g)][lab])
            except KeyError:
                vals.append(0.0)
        ax.barh(ypos, vals, left=left, color=colors[lab], label=lab)
        left += np.array(vals)
    ax.set_yticks(ypos); ax.set_yticklabels(["Group A (HF)", "Group B (control)"])
    ax.set_xlabel("share of responses (%)"); ax.set_xlim(0, 100)
    ax.set_title("Convention probe: which playbook readers invoke")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=12)
    fig.savefig(figdir / "fig_convention_type.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigures -> {figdir}")
    print(f"CSVs    -> {aggdir}")


if __name__ == "__main__":
    main()
