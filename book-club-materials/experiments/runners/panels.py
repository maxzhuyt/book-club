"""Persona panels for each condition. Documented in DESIGN.md §3."""

# C0 — single mainstream reader
C0_PANEL = ["sam"]

# C1 — diverse-but-neutral; spans emotional / period / mainstream / structural / pacing axes
C1_PANEL = ["mara", "hugo", "sam", "nadia", "pete"]

# C2 — consensus: overlapping moderate styles
C2_PANEL = ["sam", "theo", "elena", "iris", "june"]

# C3 — adversarial: extremes on every axis
C3_PANEL = ["mara", "hugo", "pete", "nadia", "rohan"]

PANELS = {
    "C0": C0_PANEL,
    "C1a": C1_PANEL,
    "C1b": C1_PANEL,
    "C1c": C1_PANEL,
    "C1d": C1_PANEL,
    "C2":  C2_PANEL,
    "C3":  C3_PANEL,
}

PROBES = {
    "C1a": ("Plausibility",
            "Was there anything in this passage that didn't quite fit the period as you understand it? "
            "Be specific about what struck you."),
    "C1b": ("Knowledge-gap",
            "Was there anything you wanted more context for, or wished the narration had explained "
            "more directly?"),
    "C1c": ("Stability",
            "Did your sense of the world remain stable as you read, or did anything pull you out of it? "
            "If something pulled you out, can you say what?"),
    "C1d": ("Convention",
            "What did you bring to this passage from your own reading — what conventions or "
            "expectations were you using to make sense of it?"),
}

CONDITION_LABELS = {
    "C0":  "Single-Reader baseline",
    "C1a": "Book-Club Probe · Plausibility",
    "C1b": "Book-Club Probe · Knowledge-gap",
    "C1c": "Book-Club Probe · Stability",
    "C1d": "Book-Club Probe · Convention",
    "C2":  "Workshop · Consensus",
    "C3":  "Workshop · Adversarial",
}
