from __future__ import annotations

from scoring import ScoreBreakdown

_DIMENSION_LABELS = {
    "technical_competence": "technical competence",
    "domain_fit": "domain fit",
    "career_quality": "career trajectory",
    "ownership": "ownership",
    "production_readiness": "production readiness",
    "leadership": "leadership",
    "learning_velocity": "learning velocity",
    "behavioral_reliability": "behavioral reliability",
    "hiring_risk": "lower hiring risk",
}


def _join_clauses(items: list[str], max_n: int) -> str:
    items = [i.strip() for i in items if i and i.strip()]
    items = items[:max_n]

    if not items:
        return ""

    if len(items) == 1:
        return items[0]

    return ", ".join(items[:-1]) + ", and " + items[-1]


def _standout_dimension(dims: dict) -> str | None:

    if not dims:
        return None

    valid = {k: v for k, v in dims.items() if k in _DIMENSION_LABELS}

    if not valid:
        return None

    best = max(valid, key=valid.get)

    if valid[best] < 0.78:
        return None

    return _DIMENSION_LABELS[best]


def build_reasoning(
    breakdown: ScoreBreakdown,
    rank: int,
    cohort_note=None,
):

    if breakdown.is_honeypot:

        why = "; ".join(breakdown.honeypot_reasons)

        return (
            "Excluded from recommendation because the profile contains "
            f"internal inconsistencies ({why})."
        )

    positives = _join_clauses(
        breakdown.reasons,
        max_n=3,
    )

    concerns = _join_clauses(
        breakdown.concerns,
        max_n=2,
    )

    sentence = f"Ranked #{rank}. "

    if positives:
        sentence += positives[0].upper() + positives[1:] + "."
    else:
        sentence += (
            "Profile demonstrates an overall match to the hiring "
            "requirements."
        )

    standout = _standout_dimension(
        breakdown.recruiter_dimensions
    )

    if standout:

        sentence += (
            f" Strongest evidence is in {standout}."
        )

    behavioral = breakdown.behavioral_multiplier

    if behavioral >= 0.95:

        sentence += (
            " Behavioral signals indicate the candidate is highly "
            "engaged and likely available."
        )

    elif behavioral <= 0.60:

        sentence += (
            " Behavioral signals suggest reduced hiring readiness."
        )

    if concerns:

        sentence += (
            " Recruiter should verify: "
            + concerns[0].upper()
            + concerns[1:]
            + "."
        )

    if cohort_note is not None:

        sentence += (
            f" Compared with the closest competing profile "
            f"({cohort_note.twin_id}), the deciding factor was "
            f"{cohort_note.deciding_label}."
        )

    if breakdown.confidence_label == "low":

        sentence += (
            " Recommendation confidence is limited because the profile "
            "contains relatively little supporting evidence."
        )

    return sentence


def build_short_summary(
    breakdown: ScoreBreakdown,
):

    if breakdown.is_honeypot:
        return "Excluded (honeypot)"

    if breakdown.reasons:
        return breakdown.reasons[0][:120]

    return "Overall recruiter fit"


def build_dimension_summary(
    breakdown: ScoreBreakdown,
):

    parts = []

    for k, v in breakdown.recruiter_dimensions.items():

        parts.append(f"{k}={v:.2f}")

    return (
        " ".join(parts)
        + f" confidence={breakdown.confidence_label}"
    )