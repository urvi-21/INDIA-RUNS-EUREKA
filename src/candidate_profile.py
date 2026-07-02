"""
candidate_profile.py
=====================
The single Candidate Understanding layer.

PHASE 2 GOAL (see PHASE2_REPORT.md for full detail): give the pipeline a
CandidateProfile the same way Phase 1 gave it a HiringProfile -- one
structured object built ONCE per candidate that organizes every piece of
raw evidence the schema contains (career timeline, responsibilities,
companies, skills, education, certifications, assessments, behavioral /
recruiter-interaction / GitHub signals, completeness, availability).

CandidateProfile is deliberately evidence-only:
  - It never computes a recruiter SCORE (no 0-1 "fit" numbers live here).
  - It never decides a final CAPABILITY (that is explicitly Phase 3 scope --
    "Evidence-Based Capability Inference" -- and stays out of this file).
  - Where it does derive something beyond a raw field (e.g. "promotion_count",
    "domain_consistency", "career_entirely_services_firms"), that derived
    value is still evidence about the candidate's history, not a judgment
    about how good a fit they are. All of scoring.py's weights, WEIGHTS,
    RECRUITER_DIMENSION_WEIGHTS, and capability_inference's evidence-weighted
    0-1 capability scores are UNCHANGED and untouched by this module.

Everything here was previously computed piecemeal, and independently, inside
features.py (career_history was re-sorted three separate times by three
separate functions; _extract_recruiter_profile() re-looped career_history a
fourth time for startup/product-company signal; capability_inference.py
re-built its own career/skills/title text blobs a fifth time). This module
is now the ONE place career_history, skills, education, and behavioral
signals are read out of the raw candidate dict and organized -- every
extractor in features.py, and capability_inference.py's evidence-source text
builder, now consume the CandidateProfile object instead of re-deriving the
same things from the raw dict.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from config import (
    SENIORITY_LADDER, CORE_TITLE_KEYWORDS, ADJACENT_TITLE_KEYWORDS,
    ARCHITECTURE_TITLE_KEYWORDS, STALE_IC_MONTHS_THRESHOLD,
    LEADERSHIP_RANK_THRESHOLD, SHORT_TENURE_MONTHS_THRESHOLD,
    STABLE_TENURE_MONTHS_THRESHOLD, DATASET_REFERENCE_DATE,
)
import capability_inference


def _parse_date(s: str | None) -> dt.date | None:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Career-sequence helpers (moved here from features.py verbatim -- these are
# about interpreting a single career_history entry / title string, which is
# squarely "career understanding," not feature scoring). features.py no
# longer defines its own copies; it reads the results off CandidateProfile.
# ---------------------------------------------------------------------------

def _seniority_rank(title: str) -> int:
    """Highest seniority-ladder rank found in a title string, or -1 if none
    of the ladder keywords appear (e.g. a non-technical title)."""
    t = (title or "").lower()
    best = -1
    for kw, rank in SENIORITY_LADDER:
        if kw in t and rank > best:
            best = rank
    return best


def _title_tier(title: str) -> str:
    """Return 'core', 'adjacent', or 'other' based on title keyword match."""
    t = (title or "").lower()
    if any(k in t for k in CORE_TITLE_KEYWORDS):
        return "core"
    if any(k in t for k in ADJACENT_TITLE_KEYWORDS):
        return "adjacent"
    return "other"


def _is_architecture_title(title: str) -> bool:
    t = (title or "").lower()
    return any(k in t for k in ARCHITECTURE_TITLE_KEYWORDS)


# Achievement evidence: description sentences that *look* like a quantified
# result rather than a plain responsibility statement. This is intentionally
# a light, auditable heuristic (a number/percent alongside an outcome verb)
# -- it flags a sentence as "possible achievement evidence" for a recruiter
# to read, it does not score or weight it in any way.
_ACHIEVEMENT_VERBS = (
    "increased", "decreased", "reduced", "improved", "grew", "boosted",
    "cut", "saved", "generated", "launched", "shipped", "scaled",
    "accelerated", "optimized", "delivered", "drove", "achieved",
)


def _extract_achievement_snippets(description: str) -> list[str]:
    if not description:
        return []
    snippets = []
    for sentence in _split_sentences(description):
        low = sentence.lower()
        has_number = any(ch.isdigit() for ch in sentence) or "%" in sentence
        has_verb = any(v in low for v in _ACHIEVEMENT_VERBS)
        if has_number and has_verb:
            snippets.append(sentence.strip())
    return snippets


def _split_sentences(text: str) -> list[str]:
    # Simple, dependency-free sentence splitter -- good enough for short
    # recruiter-style bullet/paragraph descriptions; not meant to be a full
    # NLP sentence tokenizer.
    parts = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in ".;\n":
            parts.append("".join(buf))
            buf = []
    if buf:
        parts.append("".join(buf))
    return [p.strip(" .;\n") for p in parts if p.strip(" .;\n")]


@dataclass
class CareerEntry:
    """One career_history record, enriched with career-sequence evidence
    derived purely from this entry's own fields (no cross-entry context)."""

    company: str
    title: str
    start_date: str
    end_date: str | None
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str

    # Career-sequence evidence
    seniority_rank: int
    title_tier: str
    is_architecture_title: bool

    # Responsibility evidence: which capability-ontology phrases this role's
    # title+description text touches (e.g. "Built recommendation platform"
    # -> ["retrieval_systems", "ranking_systems", "production_ml"]). These
    # are EVIDENCE TAGS, not scores -- capability_inference's own weighted
    # 0-1 scoring is untouched and lives only in capability_inference.py.
    evidence_tags: list[str] = field(default_factory=list)
    achievement_snippets: list[str] = field(default_factory=list)


@dataclass
class SkillEntry:
    name: str
    proficiency: str
    endorsements: int
    duration_months: int


@dataclass
class CandidateProfile:
    """Canonical, evidence-only representation of a candidate. Built once
    per candidate by `build()` below; every downstream extractor reads from
    this object rather than re-deriving the same things from the raw JSON."""

    candidate_id: str

    # ---- Identity / current snapshot -------------------------------------
    anonymized_name: str = ""
    headline: str = ""
    summary: str = ""
    location: str = ""
    country: str = ""
    current_title: str = ""
    current_title_tier: str = "other"
    current_company: str = ""
    current_company_size: str = ""
    current_industry: str = ""
    years_of_experience: float = 0.0

    # ---- Career timeline (evidence) ---------------------------------------
    career_history: list[CareerEntry] = field(default_factory=list)       # ascending (oldest first)
    career_history_desc: list[CareerEntry] = field(default_factory=list)  # descending (most recent first)
    most_recent_role: CareerEntry | None = None
    total_career_months: float = 0.0
    n_roles: int = 0
    companies: list[str] = field(default_factory=list)          # unique employers, first-seen order
    employers_lower: list[str] = field(default_factory=list)    # one entry per role, lowercased

    # ---- Company characteristics (evidence) --------------------------------
    career_entirely_services_firms: bool = False
    startup_months: float = 0.0
    product_company_months: float = 0.0

    # ---- Career progression evidence (careers-as-sequences) ---------------
    promotion_count: int = 0
    promotion_velocity: float = 0.0
    n_company_transitions: int = 0
    avg_tenure_months: float = 0.0
    tenure_stability: str = "unknown"
    leadership_emergence: bool = False
    leadership_is_recent: bool = False
    highest_seniority_rank: int = -1
    domain_consistency: float = 0.0
    specialization_drift: str = "unknown"
    is_stale_ic: bool = False

    # ---- Responsibility / domain evidence ----------------------------------
    # Union of every career entry's evidence_tags -- "which capability
    # domains does this candidate's history touch at all," as plain evidence
    # for a recruiter/Phase-3 capability layer to read, not a score.
    domains_worked_in: list[str] = field(default_factory=list)

    # ---- Skills evidence ----------------------------------------------------
    skills: list[SkillEntry] = field(default_factory=list)
    n_skills_total: int = 0

    # ---- Education / certifications / assessments (evidence) --------------
    education: list[dict] = field(default_factory=list)
    best_education_tier: str = "unknown"
    certifications: list[dict] = field(default_factory=list)
    languages: list[dict] = field(default_factory=list)
    skill_assessment_scores: dict = field(default_factory=dict)

    # ---- Behavioral / recruiter-interaction / GitHub signals (evidence) ---
    redrob_signals: dict = field(default_factory=dict)
    last_active_date: str | None = None
    days_inactive: int = 99999
    open_to_work_flag: bool = False
    willing_to_relocate: bool = False
    notice_period_days: int = 180
    recruiter_response_rate: float = 0.0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = -1.0
    avg_response_time_hours: float = 168.0
    saved_by_recruiters_30d: int = 0
    search_appearance_30d: int = 0
    profile_views_received_30d: int = 0
    endorsements_received: int = 0
    connection_count: int = 0
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False
    github_activity_score: float = -1.0
    profile_completeness_score: float = 0.0

    # ---- Profile completeness evidence -------------------------------------
    completeness_signals: dict = field(default_factory=dict)

    # ---- Free-text evidence blobs (built once, reused by capability
    # inference and the semantic-similarity text builder instead of each
    # rebuilding their own copy from the raw candidate dict) --------------
    career_text: str = ""
    skills_text: str = ""
    title_text: str = ""


def _build_career_entries(raw_history: list[dict]) -> list[CareerEntry]:
    entries = []
    for h in raw_history:
        title = h.get("title", "")
        description = h.get("description", "")
        entries.append(CareerEntry(
            company=h.get("company", ""),
            title=title,
            start_date=h.get("start_date", ""),
            end_date=h.get("end_date"),
            duration_months=h.get("duration_months", 0) or 0,
            is_current=h.get("is_current", False),
            industry=h.get("industry", ""),
            company_size=h.get("company_size", ""),
            description=description,
            seniority_rank=_seniority_rank(title),
            title_tier=_title_tier(title),
            is_architecture_title=_is_architecture_title(title),
            evidence_tags=capability_inference.match_capability_keys(
                f"{title} {description}"
            ),
            achievement_snippets=_extract_achievement_snippets(description),
        ))
    return entries


def _career_progression_evidence(history_asc: list[CareerEntry]) -> dict:
    """Sequence-level evidence computed from the chronologically-ordered
    career entries. Logic is unchanged from Phase 1's
    features.extract_trajectory_features -- only its *location* moved, so
    scoring.py's downstream consumption is byte-identical."""
    n = len(history_asc)
    if n == 0:
        return dict(
            promotion_count=0, promotion_velocity=0.0,
            n_company_transitions=0, avg_tenure_months=0.0,
            tenure_stability="unknown", leadership_emergence=False,
            leadership_is_recent=False, domain_consistency=0.0,
            specialization_drift="unknown", highest_seniority_rank=-1,
        )

    ranks = [e.seniority_rank for e in history_asc]
    promotions = sum(
        1 for i in range(1, n)
        if ranks[i] > ranks[i - 1] and ranks[i - 1] >= 0
    )
    total_months = sum(e.duration_months for e in history_asc) or 1
    promotion_velocity = promotions / (total_months / 36.0)

    employers = [e.company.strip().lower() for e in history_asc]
    n_transitions = max(0, len({e for e in employers}) - 1) if employers else 0

    avg_tenure = total_months / n
    if avg_tenure >= STABLE_TENURE_MONTHS_THRESHOLD:
        stability = "stable"
    elif avg_tenure <= SHORT_TENURE_MONTHS_THRESHOLD:
        stability = "frequent_mover"
    else:
        stability = "moderate"

    highest_rank = max(ranks) if ranks else -1
    leadership_emergence = highest_rank >= LEADERSHIP_RANK_THRESHOLD
    leadership_is_recent = ranks[-1] >= LEADERSHIP_RANK_THRESHOLD

    tiers = [e.title_tier for e in history_asc]
    current_tier = tiers[-1]
    domain_consistency = sum(1 for t in tiers if t == current_tier) / n

    tier_rank = {"other": 0, "adjacent": 1, "core": 2}
    third = max(1, n // 3)
    early_score = sum(tier_rank[t] for t in tiers[:third]) / third
    late_score = sum(tier_rank[t] for t in tiers[-third:]) / third
    if late_score - early_score >= 0.5:
        drift = "narrowing_to_domain"
    elif early_score - late_score >= 0.5:
        drift = "broadening_away_from_domain"
    else:
        drift = "stable"

    return dict(
        promotion_count=promotions,
        promotion_velocity=promotion_velocity,
        n_company_transitions=n_transitions,
        avg_tenure_months=avg_tenure,
        tenure_stability=stability,
        leadership_emergence=leadership_emergence,
        leadership_is_recent=leadership_is_recent,
        domain_consistency=domain_consistency,
        specialization_drift=drift,
        highest_seniority_rank=highest_rank,
    )


def _company_characteristics(history_asc: list[CareerEntry]) -> dict:
    """Product-vs-services / startup-vs-established evidence, derived once
    from structured company_size/industry fields. Previously duplicated
    inside features._extract_recruiter_profile(), which independently
    looped career_history a second time to compute the same thing."""
    employers = [e.company.strip().lower() for e in history_asc]
    all_services = bool(employers) and all(
        any(sf in emp for sf in _SERVICES_FIRMS()) for emp in employers
    )

    startup_months = 0.0
    product_company_months = 0.0
    for e in history_asc:
        size = (e.company_size or "").lower()
        industry = (e.industry or "").lower()
        months = e.duration_months
        if "startup" in size or "1-50" in size or "11-50" in size:
            startup_months += months
        if "product" in industry or "software" in industry or "saas" in industry:
            product_company_months += months

    return dict(
        career_entirely_services_firms=all_services,
        employers_lower=employers,
        startup_months=startup_months,
        product_company_months=product_company_months,
    )


def _SERVICES_FIRMS():
    from config import SERVICES_FIRMS
    return SERVICES_FIRMS


def _completeness_signals(candidate: dict, profile_completeness_score: float) -> dict:
    profile = candidate.get("profile", {})
    return {
        "has_summary": bool(profile.get("summary")),
        "has_headline": bool(profile.get("headline")),
        "has_career_history": len(candidate.get("career_history", [])) > 0,
        "has_skills": len(candidate.get("skills", [])) > 0,
        "has_education": len(candidate.get("education", [])) > 0,
        "redrob_completeness_ge_50": profile_completeness_score >= 50,
    }


def build(candidate: dict) -> CandidateProfile:
    """Build the CandidateProfile: the single, evidence-only representation
    of a candidate that every downstream extractor consumes."""

    profile_dict = candidate.get("profile", {})
    raw_history = candidate.get("career_history", [])
    raw_skills = candidate.get("skills", [])
    sig = candidate.get("redrob_signals", {})

    # ---- Career timeline: built ONCE, both orderings derived from it ----
    # Ascending sort is stable, so ties preserve raw career_history order.
    # Sorting that ascending result again with reverse=True is *also*
    # stable, so ties preserve the order of its input (which already
    # preserved raw order) -- net effect is identical to sorting the raw
    # list directly with reverse=True, which is what Phase 1's
    # extract_title_features / extract_experience_features each did
    # independently. This is now computed once and both orderings are
    # derived from the same underlying CareerEntry objects.
    history_asc = _build_career_entries(
        sorted(raw_history, key=lambda h: h.get("start_date", ""))
    )
    history_desc = sorted(history_asc, key=lambda e: e.start_date, reverse=True)
    most_recent_role = history_desc[0] if history_desc else None

    is_stale_ic = bool(
        most_recent_role
        and most_recent_role.is_architecture_title
        and most_recent_role.duration_months >= STALE_IC_MONTHS_THRESHOLD
    )

    total_career_months = sum(e.duration_months for e in history_asc)
    companies_seen, companies = set(), []
    for e in history_asc:
        if e.company not in companies_seen:
            companies_seen.add(e.company)
            companies.append(e.company)

    progression = _career_progression_evidence(history_asc)
    company_chars = _company_characteristics(history_asc)

    domains_worked_in = sorted({
        tag for e in history_asc for tag in e.evidence_tags
    })

    # ---- Skills evidence --------------------------------------------------
    skills = [
        SkillEntry(
            name=s.get("name", ""),
            proficiency=s.get("proficiency", ""),
            endorsements=s.get("endorsements", 0) or 0,
            duration_months=s.get("duration_months", 0) or 0,
        )
        for s in raw_skills
    ]

    # ---- Education ----------------------------------------------------------
    education = candidate.get("education", [])
    tier_rank = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0}
    best_tier = "unknown"
    if education:
        best = max(education, key=lambda e: tier_rank.get(e.get("tier", "unknown"), 0))
        best_tier = best.get("tier", "unknown")

    # ---- Behavioral / recruiter-interaction / GitHub / availability -------
    profile_completeness_score = sig.get("profile_completeness_score", 0) or 0
    last_active = _parse_date(sig.get("last_active_date"))
    days_inactive = (
        (DATASET_REFERENCE_DATE - last_active).days if last_active else 99999
    )

    # ---- Free-text evidence blobs (single build, reused everywhere) -------
    career_text = " ".join(f"{e.title} {e.description}" for e in history_asc)
    skills_text = " ".join(s.name for s in skills)
    title_text = profile_dict.get("current_title", "")

    return CandidateProfile(
        candidate_id=candidate.get("candidate_id", ""),
        anonymized_name=profile_dict.get("anonymized_name", ""),
        headline=profile_dict.get("headline", ""),
        summary=profile_dict.get("summary", ""),
        location=profile_dict.get("location", ""),
        country=profile_dict.get("country", ""),
        current_title=profile_dict.get("current_title", ""),
        current_title_tier=_title_tier(profile_dict.get("current_title", "")),
        current_company=profile_dict.get("current_company", ""),
        current_company_size=profile_dict.get("current_company_size", ""),
        current_industry=profile_dict.get("current_industry", ""),
        years_of_experience=profile_dict.get("years_of_experience", 0.0),

        career_history=history_asc,
        career_history_desc=history_desc,
        most_recent_role=most_recent_role,
        total_career_months=total_career_months,
        n_roles=len(history_asc),
        companies=companies,
        employers_lower=company_chars["employers_lower"],

        career_entirely_services_firms=company_chars["career_entirely_services_firms"],
        startup_months=company_chars["startup_months"],
        product_company_months=company_chars["product_company_months"],

        promotion_count=progression["promotion_count"],
        promotion_velocity=progression["promotion_velocity"],
        n_company_transitions=progression["n_company_transitions"],
        avg_tenure_months=progression["avg_tenure_months"],
        tenure_stability=progression["tenure_stability"],
        leadership_emergence=progression["leadership_emergence"],
        leadership_is_recent=progression["leadership_is_recent"],
        highest_seniority_rank=progression["highest_seniority_rank"],
        domain_consistency=progression["domain_consistency"],
        specialization_drift=progression["specialization_drift"],
        is_stale_ic=is_stale_ic,

        domains_worked_in=domains_worked_in,

        skills=skills,
        n_skills_total=len(skills),

        education=education,
        best_education_tier=best_tier,
        certifications=candidate.get("certifications", []) or [],
        languages=candidate.get("languages", []) or [],
        skill_assessment_scores=sig.get("skill_assessment_scores", {}) or {},

        redrob_signals=sig,
        last_active_date=sig.get("last_active_date"),
        days_inactive=days_inactive,
        open_to_work_flag=sig.get("open_to_work_flag", False),
        willing_to_relocate=sig.get("willing_to_relocate", False),
        notice_period_days=sig.get("notice_period_days", 180),
        recruiter_response_rate=sig.get("recruiter_response_rate", 0.0),
        interview_completion_rate=sig.get("interview_completion_rate", 0.0),
        offer_acceptance_rate=sig.get("offer_acceptance_rate", -1),
        avg_response_time_hours=sig.get("avg_response_time_hours", 168),
        saved_by_recruiters_30d=sig.get("saved_by_recruiters_30d", 0),
        search_appearance_30d=sig.get("search_appearance_30d", 0),
        profile_views_received_30d=sig.get("profile_views_received_30d", 0),
        endorsements_received=sig.get("endorsements_received", 0),
        connection_count=sig.get("connection_count", 0),
        verified_email=sig.get("verified_email", False),
        verified_phone=sig.get("verified_phone", False),
        linkedin_connected=sig.get("linkedin_connected", False),
        github_activity_score=sig.get("github_activity_score", -1),
        profile_completeness_score=profile_completeness_score,

        completeness_signals=_completeness_signals(candidate, profile_completeness_score),

        career_text=career_text,
        skills_text=skills_text,
        title_text=title_text,
    )
