"""V3 probes. Five attentional probes, each a system-prompt PRIME paired with a
moderator ELICITATION (verbatim from experiments_v3/v3.md). P5 (Salience) is a
priors-symmetric control. P3 (Stability) is acknowledged not perfectly priors-
independent (see v3.md summary)."""

GENERIC_PASS1 = (
    "Read the passage above. Before any specific question: what did you notice, "
    "and what was your experience of reading it? Speak in your own voice."
)

PROBES = {
    "P1": {
        "name": "Plausibility", "type": "priors-dependent",
        "prime": (
            "As you read the following passage, identify where the historical material "
            "aligns with what you know of the period and where it does not. Notice two "
            "kinds of mismatches: (a) places where historical figures or events drawn "
            "from the actual past have properties or behaviors that don't match the "
            "historical record; (b) places where invented elements clash with the "
            "period — anachronistic, out of place, generic, or unmotivated by the "
            "world's logic. What you identify may be specific or it may be a general "
            "sense that something doesn't sit right without being easily localized.\n\n"
            "If the passage involves deliberately counterfactual or fantastical "
            "elements, focus on whether the world holds together on its own declared "
            "terms rather than whether it matches actual history."
        ),
        "elicitation": (
            "Did anything in the passage not match what you would expect of this "
            "historical setting? If something stood out, was it about something the "
            "passage referred to from real history, or about something the passage "
            "invented? If you can't tell which, or if the mismatch is hard to "
            "localize, that's worth reporting too."
        ),
    },
    "P2": {
        "name": "Knowledge-gap", "type": "priors-dependent",
        "prime": (
            "As you read the following passage, identify how the narration handles "
            "what it explains and what it leaves unsaid. Notice three kinds of "
            "moments: (a) places where the text leaves out something the scene needed "
            "for coherence; (b) places where the text supplies more than it needed "
            "to — explaining something a competent reader of the period would infer, "
            "or breaking a silence the world's logic should have kept; (c) places "
            "where the text is complete on its own terms but invites curiosity about "
            "the actual history beyond it.\n\n"
            "If the passage involves deliberately counterfactual or fantastical "
            "elements, focus on whether the world's own specifications hold together "
            "rather than whether they match actual history."
        ),
        "elicitation": (
            "Were there moments where the narration handled explanation in a way "
            "that stood out — by leaving out something the scene needed, by supplying "
            "more than necessary, or by leaving open a question about the actual "
            "history beyond the text? Point to specific places, and try to say which "
            "of the three the moment was. If it's ambiguous between them, that's "
            "worth reporting."
        ),
    },
    "P3": {
        "name": "Stability", "type": "priors-independent",
        "prime": (
            "As you read the following passage, identify any places where the writing "
            "calls attention to itself — where the prose, the framing, or the "
            "construction becomes visible as writing rather than receding behind the "
            "events being narrated. This might happen because of an unusual phrasing, "
            "an apparent author intrusion, a stylistic shift, a structural break, a "
            "passage that reads more like commentary than scene, or other features "
            "that foreground the text as text."
        ),
        "elicitation": (
            "Were there places in the passage where the writing called attention to "
            "itself as writing or counterfactual fiction? If so, what produced the "
            "effect — a particular phrase, a shift in register, an intrusion of "
            "authorial commentary, a structural feature? It is also informative to "
            "report that the writing remained transparent throughout — that the "
            "passage moved without drawing attention to its own construction."
        ),
    },
    "P4": {
        "name": "Convention", "type": "priors-dependent",
        "prime": (
            "As you read the following passage, identify moments where it either "
            "follows what scenes of this kind typically do — moving through familiar "
            "patterns and beats — or breaks from typical patterns. Notice both: "
            "places where the passage follows expected moves for this kind of scene, "
            "and places where it departs from them, either by doing something "
            "unexpected or by skipping something this kind of scene would typically "
            "include."
        ),
        "elicitation": (
            "Were there moments in the passage that followed familiar patterns for "
            "this kind of scene? Were there moments that departed from those "
            "patterns — either by doing something unexpected, or by skipping "
            "something this kind of scene would typically include? Point to specific "
            "places in the passage."
        ),
    },
    "P5": {
        "name": "Salience", "type": "priors-symmetric",
        "prime": (
            "As you read the following passage, identify which parts of it you would "
            "highlight as most worth attending to if you were preparing a fellow "
            "reader to engage with it. These might be moments that are vivid, that "
            "advance the action, that reveal something about a character, that "
            "establish the world, or that have other distinctive importance. Think of "
            "yourself as marking up the passage to guide another reader's focus."
        ),
        "elicitation": (
            "If you were marking up this passage to guide a fellow reader's "
            "attention, which parts would you highlight? Identify two or three "
            "parts, and say briefly what makes each of them worth the reader's focus."
        ),
    },
}
