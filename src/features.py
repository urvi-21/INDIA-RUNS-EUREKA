

from __future__ import annotations
from typing import Any
import re
from collections import defaultdict
from config import (
    CORE_SKILLS, SUPPORT_SKILLS, OFF_DOMAIN_AI_SKILLS,
    RECENT_AI_ONLY_MONTHS_THRESHOLD,
    PREFERRED_LOCATIONS, WELCOME_LOCATIONS,
    HONEYPOT_CAREER_OVERRUN_MONTHS, HONEYPOT_MIN_ZERO_DURATION_EXPERT_SKILLS,
    INACTIVE_DAYS_THRESHOLD,
    SKILL_CLUSTERS, SKILL_ADJACENT_CREDIT_FRACTION,
)
import capability_inference
import candidate_profile
from candidate_profile import CandidateProfile


# ---------------------------------------------------------------------------
# Skill intelligence: build a term -> cluster lookup once at import time so
# per-candidate matching stays O(1) per skill (no scanning all clusters per
# skill at runtime -- important for the 100K-candidate / 5-minute budget).
# ---------------------------------------------------------------------------
def _build_skill_cluster_index() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    synonym_index: dict[str, list[str]] = {}
    adjacent_index: dict[str, list[str]] = {}
    for cluster_id, cluster in SKILL_CLUSTERS.items():
        for term in cluster["synonyms"]:
            synonym_index.setdefault(term, []).append(cluster_id)
        for term in cluster["adjacent"]:
            adjacent_index.setdefault(term, []).append(cluster_id)
    return synonym_index, adjacent_index


_SKILL_SYNONYM_INDEX, _SKILL_ADJACENT_INDEX = _build_skill_cluster_index()
_CORE_WEIGHTED_CLUSTERS = {cid for cid, c in SKILL_CLUSTERS.items() if c["weight_in_core"]}


def extract_title_features(profile: CandidateProfile) -> dict:
    """
    PHASE 2: title tier and the stale-IC check ("most recent role is
    architecture/lead AND has lasted longer than the threshold") used to
    re-sort career_history and re-run the title-keyword checks here. Both
    are now pure evidence fields computed once by candidate_profile.build()
    (current_title_tier, is_stale_ic) -- this function just reads them off
    the CandidateProfile. Values are unchanged.
    """
    return {
        "title_tier": profile.current_title_tier,
        "is_stale_ic": profile.is_stale_ic,
        "current_title": profile.current_title,
    }


def extract_skill_features(profile: CandidateProfile) -> dict:
    skills = profile.skills
    core_trust = 0.0
    support_trust = 0.0
    off_domain_trust = 0.0
    cluster_trust = 0.0
    core_hits, support_hits, off_domain_hits, cluster_hits = [], [], [], []
    matched_cluster_ids: set[str] = set()

    for s in skills:
        name = s.name.strip().lower()
        endorsements = s.endorsements or 0
        duration = s.duration_months or 0
        # "trust" weight: a skill claimed with 0 endorsements and 0 months
        # used contributes ~nothing; a skill with real tenure and social
        # proof contributes a lot. This is the core defense against
        # keyword-stuffing (JD: "that's a trap we've explicitly built in").
        trust = (1 + min(endorsements, 50) / 10) * (1 + min(duration, 60) / 12)

        if name in CORE_SKILLS:
            core_trust += trust
            core_hits.append(name)
        elif name in SUPPORT_SKILLS:
            support_trust += trust
            support_hits.append(name)
        elif name in OFF_DOMAIN_AI_SKILLS:
            off_domain_trust += trust
            off_domain_hits.append(name)

        # Skill-cluster intelligence: a skill that isn't a literal CORE_SKILLS
        # string but is a *synonym* of one (e.g. "Retrieval Augmented
        # Generation" for "rag") earns full trust into that cluster; an
        # *adjacent* technology (e.g. "LangGraph" -> AI Orchestration cluster,
        # which itself is adjacent-only to core retrieval skills) earns a
        # discounted fraction. Each cluster counts a candidate only once
        # (matched_cluster_ids) so ten synonyms of the same concept don't
        # multiply credit -- this preserves the anti-keyword-stuffing
        # property of the original design while recognizing real equivalence.
        for cluster_id in _SKILL_SYNONYM_INDEX.get(name, []):
            if cluster_id in matched_cluster_ids:
                continue
            if SKILL_CLUSTERS[cluster_id]["weight_in_core"]:
                cluster_trust += trust
                matched_cluster_ids.add(cluster_id)
                cluster_hits.append(f"{name}\u2192{SKILL_CLUSTERS[cluster_id]['canonical']}")
        for cluster_id in _SKILL_ADJACENT_INDEX.get(name, []):
            if cluster_id in matched_cluster_ids:
                continue
            if SKILL_CLUSTERS[cluster_id]["weight_in_core"]:
                cluster_trust += trust * SKILL_ADJACENT_CREDIT_FRACTION
                matched_cluster_ids.add(cluster_id)
                cluster_hits.append(f"{name}\u2192{SKILL_CLUSTERS[cluster_id]['canonical']} (adjacent)")

    return {
        "core_skill_trust": core_trust,
        "support_skill_trust": support_trust,
        "off_domain_skill_trust": off_domain_trust,
        "cluster_skill_trust": cluster_trust,
        "core_skill_hits": core_hits,
        "support_skill_hits": support_hits,
        "cluster_skill_hits": cluster_hits,
        "n_skills_total": len(skills),
    }


def extract_experience_features(profile: CandidateProfile) -> dict:
    yoe = profile.years_of_experience
    total_career_months = profile.total_career_months

    # Recent-AI-only check: does the candidate's AI/ML-relevant experience
    # come almost entirely from the most recent stretch (< threshold months)
    # with no earlier ML/IR signal? Proxy: look at career entries beyond the
    # most recent N months and see if any show ML/IR keywords in title or
    # description. Uses the CandidateProfile's already-sorted descending
    # career timeline instead of re-sorting career_history here.
    recent_months_acc = 0
    earlier_has_ml_signal = False
    ml_desc_keywords = (
        "machine learning", "retrieval", "ranking", "recommend", "embedding",
        "nlp", "search", "ml model", "vector", "information retrieval",
    )
    for e in profile.career_history_desc:
        recent_months_acc += e.duration_months
        text = (e.title + " " + e.description).lower()
        has_ml = any(k in text for k in ml_desc_keywords)
        if recent_months_acc > RECENT_AI_ONLY_MONTHS_THRESHOLD and has_ml:
            earlier_has_ml_signal = True

    return {
        "years_of_experience": yoe,
        "total_career_months": total_career_months,
        "earlier_has_ml_signal": earlier_has_ml_signal,
    }


def extract_company_features(profile: CandidateProfile) -> dict:
    return {
        "career_entirely_services_firms": profile.career_entirely_services_firms,
        "employers": profile.employers_lower,
    }


def extract_location_features(profile: CandidateProfile) -> dict:
    location = profile.location.lower()
    country = profile.country.strip()
    is_india = country.strip().lower() == "india"

    if any(p in location for p in PREFERRED_LOCATIONS):
        loc_tier = "preferred"
    elif any(w in location for w in WELCOME_LOCATIONS):
        loc_tier = "welcome"
    elif is_india:
        loc_tier = "india_other"
    else:
        loc_tier = "outside_india"

    return {
        "location_tier": loc_tier,
        "is_india": is_india,
        "raw_location": profile.location,
        "country": country,
    }


def extract_education_features(profile: CandidateProfile) -> dict:
    return {"best_tier": profile.best_education_tier}


def extract_behavioral_features(profile: CandidateProfile) -> dict:
    return {
        "days_inactive": profile.days_inactive,
        "is_inactive_180d": profile.days_inactive >= INACTIVE_DAYS_THRESHOLD,
        "open_to_work_flag": profile.open_to_work_flag,
        "recruiter_response_rate": profile.recruiter_response_rate,
        "interview_completion_rate": profile.interview_completion_rate,
        "offer_acceptance_rate": profile.offer_acceptance_rate,
        "notice_period_days": profile.notice_period_days,
        "willing_to_relocate": profile.willing_to_relocate,
        "github_activity_score": profile.github_activity_score,
        "profile_completeness_score": profile.profile_completeness_score,
        "saved_by_recruiters_30d": profile.saved_by_recruiters_30d,
        "search_appearance_30d": profile.search_appearance_30d,
    }


def detect_honeypot(profile: CandidateProfile, exp_features: dict) -> dict:
    """
    Flags internally-inconsistent ("honeypot") profiles using only fields
    present in the schema -- no external ground truth is used or assumed.
    See submission_spec.md Section 7 and redrob_signals_doc.md for the two
    documented example patterns this function operationalizes.
    """
    reasons = []

    overrun_months = exp_features["total_career_months"] - (
        exp_features["years_of_experience"] * 12
    )
    if overrun_months > HONEYPOT_CAREER_OVERRUN_MONTHS:
        reasons.append(
            f"career_history totals {exp_features['total_career_months']:.0f}mo "
            f"vs stated {exp_features['years_of_experience']:.1f}y experience"
        )

    zero_duration_experts = sum(
        1 for s in profile.skills
        if s.proficiency == "expert" and (s.duration_months or 0) == 0
    )
    if zero_duration_experts >= HONEYPOT_MIN_ZERO_DURATION_EXPERT_SKILLS:
        reasons.append(
            f"{zero_duration_experts} skills marked 'expert' with 0 months used"
        )

    return {"is_honeypot": bool(reasons), "honeypot_reasons": reasons}


def extract_trajectory_features(profile: CandidateProfile) -> dict:
    """
    Career-trajectory signals computed from career_history alone, treated as
    a *sequence* rather than independent entries (JD weakness #4: "career
    history is currently treated as independent entries").

    PHASE 2: the sequence math (promotion velocity, tenure stability,
    leadership emergence/recency, domain consistency, specialization drift)
    used to live here, operating on a career_history list this function
    re-sorted itself. It's now computed once in
    candidate_profile._career_progression_evidence() as part of building the
    CandidateProfile, and this function just reads the result off the
    profile -- values are unchanged.
    """
    return {
        "promotion_count": profile.promotion_count,
        "promotion_velocity": profile.promotion_velocity,
        "n_company_transitions": profile.n_company_transitions,
        "avg_tenure_months": profile.avg_tenure_months,
        "tenure_stability": profile.tenure_stability,
        "leadership_emergence": profile.leadership_emergence,
        "leadership_is_recent": profile.leadership_is_recent,
        "domain_consistency": profile.domain_consistency,
        "specialization_drift": profile.specialization_drift,
        "highest_seniority_rank": profile.highest_seniority_rank,
    }


def extract_completeness_features(profile: CandidateProfile) -> dict:
    """
    Profile-completeness fraction used only for internal confidence
    estimation (scoring.py / confidence layer) -- never a ranking input
    itself, to avoid rewarding candidates simply for filling out more
    fields. Cheap O(1) check per candidate.

    PHASE 2: the underlying booleans are now CandidateProfile evidence
    (profile.completeness_signals), computed once alongside the rest of the
    profile instead of being re-derived here from the raw dict.
    """
    signals = profile.completeness_signals
    return {"completeness_fraction": sum(1 for v in signals.values() if v) / len(signals)}

# ---------------------------------------------------------------------------
# RECRUITER CAPABILITY INTELLIGENCE
#
# PHASE 1: the keyword-pattern capability detector that used to live here
# (_CAPABILITY_PATTERNS / _detect_capabilities) has been removed. It was a
# duplicate of the never-wired CapabilityInference module in
# capability_inference.py. There is now exactly ONE capability reasoning
# pipeline -- capability_inference.infer_capabilities() -- and this module
# calls it instead of re-implementing it.
#
# PHASE 2: _extract_recruiter_profile() used to independently re-loop
# career_history a second time (a third and fourth time counting
# capability_inference's own text-blob builder and extract_trajectory_
# features' sequence pass) to total up startup/product-company months from
# company_size/industry. That total is now computed once, in
# candidate_profile.build(), as company-characteristics evidence
# (profile.startup_months / profile.product_company_months) -- this
# function just normalizes it into the same 0-1 signal as before.
# ---------------------------------------------------------------------------


def _extract_recruiter_profile(profile: CandidateProfile) -> dict:
    """
    High-level recruiter profile signals, normalized from CandidateProfile's
    company-characteristics evidence (startup/product-company months).
    """
    return {
        "startup_experience": min(profile.startup_months / 24.0, 1.0),
        "product_company_experience": min(profile.product_company_months / 36.0, 1.0),
    }


def extract_capability_features(profile: CandidateProfile) -> dict:
    """
    Single entry point for recruiter-capability features. Evidence-based
    capabilities (retrieval_systems, ranking_systems, ... open_source) come
    from capability_inference.infer_capability_evidence() -- the one
    capability reasoning pipeline (PHASE 3: career/achievement/progression/
    company-context/assessment evidence first, technology mentions folded
    in only as a bounded, secondary strengthener). Career-context signals
    (startup/product-company experience) come from CandidateProfile's
    company-characteristics evidence via _extract_recruiter_profile() -- a
    distinct, non-duplicated concern.

    Beyond the flat 0-1 scores scoring.py has always consumed, this also
    surfaces the recruiter-readable reasoning behind each capability
    (`capability_reasoning`, `capability_headline`) and the full structured
    evidence (`capability_evidence_summary`) for the explanation JSON --
    so a recruiter reading the output sees WHY a capability was credited,
    not just a number.
    """
    capability_evidence = capability_inference.infer_capability_evidence(profile)
    capability_scores = {cap: ev.confidence for cap, ev in capability_evidence.items()}
    recruiter_profile = _extract_recruiter_profile(profile)

    capability_reasoning = {
        cap: ev.reasoning_sentence for cap, ev in capability_evidence.items() if ev.reasoning_sentence
    }
    # The single strongest, best-evidenced capability -- used by
    # score_skills_trust as a recruiter-style headline instead of a raw
    # skill-name dump ("Hands-on with FAISS, Pinecone, Weaviate").
    headline_cap = max(
        capability_evidence.values(),
        key=lambda ev: (ev.confidence, ev.evidence_count),
        default=None,
    )
    capability_headline = (
        headline_cap.reasoning_sentence
        if headline_cap and headline_cap.confidence >= 0.5 and headline_cap.reasoning_sentence
        else ""
    )

    capability_evidence_summary = {
        cap: {
            "confidence": ev.confidence,
            "evidence_count": ev.evidence_count,
            "evidence_strength": ev.evidence_strength,
            "supporting_positions": ev.supporting_positions,
            "supporting_projects": ev.supporting_projects,
            "supporting_evidence": ev.supporting_evidence,
            "technology_signals": ev.technology_signals,
        }
        for cap, ev in capability_evidence.items()
    }

    return {
        **capability_scores,
        **recruiter_profile,
        "capability_reasoning": capability_reasoning,
        "capability_headline": capability_headline,
        "capability_evidence_summary": capability_evidence_summary,
    }

def extract_behavior_intelligence(profile: CandidateProfile) -> dict:
    """
    Convert the raw Redrob behavioural signals into recruiter-level concepts.

    These are NOT replacements for the original signals.
    They are higher-level recruiter concepts derived from multiple signals.

    PHASE 2: reads `profile.redrob_signals` -- the same raw signals dict,
    but sourced from the CandidateProfile that was already built once for
    this candidate -- instead of indexing the raw candidate dict directly a
    second time. Recency is now taken from `profile.days_inactive`
    (computed once, alongside the rest of the profile) instead of
    re-parsing `last_active_date` here.
    """

    sig = profile.redrob_signals

    # ---------- Availability ----------
    availability = 0.0

    if sig.get("open_to_work_flag", False):
        availability += 0.40

    notice = sig.get("notice_period_days", 180)
    availability += max(0.0, (180 - notice) / 180.0) * 0.30

    if profile.last_active_date:
        availability += max(0.0, (180 - profile.days_inactive) / 180.0) * 0.30

    availability = min(availability, 1.0)

    # ---------- Recruiter Confidence ----------
    recruiter_confidence = (
        0.30 * sig.get("recruiter_response_rate", 0.0)
        + 0.25 * sig.get("interview_completion_rate", 0.0)
        + 0.20 * (1.0 if sig.get("verified_email", False) else 0.0)
        + 0.15 * (1.0 if sig.get("verified_phone", False) else 0.0)
        + 0.10 * (1.0 if sig.get("linkedin_connected", False) else 0.0)
    )

    recruiter_confidence = min(recruiter_confidence, 1.0)

    # ---------- Reliability ----------
    offer_rate = sig.get("offer_acceptance_rate", -1)

    if offer_rate < 0:
        offer_rate = 0.5

    response_rate = sig.get("recruiter_response_rate", 0.0)
    interview_rate = sig.get("interview_completion_rate", 0.0)

    avg_response_time = sig.get("avg_response_time_hours", 168)

    response_speed = max(0.0, (168 - min(avg_response_time, 168)) / 168)

    reliability = (
        0.35 * interview_rate
        + 0.30 * response_rate
        + 0.20 * offer_rate
        + 0.15 * response_speed
    )

    reliability = min(reliability, 1.0)

    # ---------- Market Demand ----------
    recruiter_saves = min(sig.get("saved_by_recruiters_30d", 0), 50) / 50
    profile_views = min(sig.get("profile_views_received_30d", 0), 200) / 200
    search_hits = min(sig.get("search_appearance_30d", 0), 500) / 500
    endorsements = min(sig.get("endorsements_received", 0), 100) / 100

    market_demand = (
        0.35 * recruiter_saves
        + 0.25 * search_hits
        + 0.20 * profile_views
        + 0.20 * endorsements
    )

    market_demand = min(market_demand, 1.0)

    # ---------- Platform Engagement ----------
    github = sig.get("github_activity_score", -1)
    github = max(github, 0) / 100

    applications = min(sig.get("applications_submitted_30d", 0), 30) / 30
    completeness = sig.get("profile_completeness_score", 0) / 100
    connections = min(sig.get("connection_count", 0), 500) / 500

    platform_engagement = (
        0.35 * github
        + 0.25 * applications
        + 0.20 * completeness
        + 0.20 * connections
    )

    platform_engagement = min(platform_engagement, 1.0)

    # ---------- Hiring Readiness ----------
    hiring_readiness = (
        0.40 * availability
        + 0.30 * recruiter_confidence
        + 0.20 * reliability
        + 0.10 * platform_engagement
    )

    hiring_readiness = min(hiring_readiness, 1.0)

    return {
        "availability_score": availability,
        "recruiter_confidence": recruiter_confidence,
        "reliability_score": reliability,
        "market_demand_score": market_demand,
        "platform_engagement": platform_engagement,
        "hiring_readiness": hiring_readiness,
    }
    
def extract_all_features(candidate: dict, hiring_profile=None,) -> dict:
    """
    Single entry point: candidate JSON dict -> flat feature dict.

    PHASE 2: builds the CandidateProfile ONCE here -- the single structured,
    evidence-only representation of the candidate -- and every extractor
    below reads from it instead of re-deriving the same things
    (career_history sorting, text-blob construction, company-characteristic
    loops, etc.) from the raw dict independently. The flat feature dict this
    function returns is unchanged in shape and values: scoring.py, ranker.py,
    and the existing test suite all continue to consume exactly the same
    keys they did before.
    """
    profile = candidate_profile.build(candidate)

    title_f = extract_title_features(profile)
    skill_f = extract_skill_features(profile)
    exp_f = extract_experience_features(profile)
    company_f = extract_company_features(profile)
    loc_f = extract_location_features(profile)
    edu_f = extract_education_features(profile)
    behav_f = extract_behavioral_features(profile)
    traj_f = extract_trajectory_features(profile)
    capability_f = extract_capability_features(profile)
    behavior_intelligence = extract_behavior_intelligence(profile)
    completeness_f = extract_completeness_features(profile)
    honeypot_f = detect_honeypot(profile, exp_f)

    feats = {
        "candidate_id": profile.candidate_id,
        **title_f, **skill_f, **exp_f, **company_f,
        **loc_f, **edu_f, **behav_f, **traj_f, **capability_f, **behavior_intelligence, **completeness_f, **honeypot_f,
    }

    # jd_alignment reads capability scores already present in `feats`, so it
    # must be computed after `feats` exists, then folded in.
    feats.update(extract_jd_alignment(feats, hiring_profile))

    # Keep a reference to source text fields needed by the semantic layer.
    feats["_text_for_embedding"] = _build_text_blob(profile, skill_f)
    return feats

def extract_jd_alignment(candidate_features, hiring_profile):
    """
    Score how well a candidate's already-computed capability features align
    with the HiringProfile's required/preferred capability bullets.

    PHASE 1 FIX: this previously converted each free-text requirement bullet
    (e.g. "3+ years building retrieval or ranking systems") into a
    snake_case dict key via `.lower().replace(" ", "_")` and looked that up
    directly in candidate_features -- which never matches any real feature
    key (those are things like "retrieval_systems", not
    "3+_years_building_retrieval_or_ranking_systems"), so this always
    silently evaluated to 0.0 for every candidate. It now resolves each
    bullet to the canonical capability key(s) it actually refers to (via
    capability_inference.match_capability_keys, the same ontology used to
    score candidates) before looking anything up, so a requirement bullet
    that doesn't map to a known capability is skipped rather than silently
    contributing a meaningless zero.
    """

    if hiring_profile is None:
        return {}

    def _avg_alignment(requirement_bullets: list[str]) -> float:
        matched_scores = []
        for bullet in requirement_bullets:
            for key in capability_inference.match_capability_keys(bullet):
                matched_scores.append(candidate_features.get(key, 0.0))
        return sum(matched_scores) / len(matched_scores) if matched_scores else 0.0

    return {
        "required_capability_match": _avg_alignment(hiring_profile.required_capabilities),
        "preferred_capability_match": _avg_alignment(hiring_profile.preferred_capabilities),
    }
    
def _build_text_blob(profile: CandidateProfile, skill_f: dict | None = None) -> str:
    """
    Builds the free text used for semantic similarity against the JD.
    Deliberately weighted toward career_history descriptions and summary,
    NOT the skills array -- per the JD's own framing, narrative > keyword
    list. Title and headline included for lexical anchoring.

    Concept expansion: when a candidate's matched skill clusters include a
    canonical concept name that may not appear verbatim in their free text
    (e.g. they list the skill "rag" but never write "Retrieval Augmented
    Generation" in a sentence), we append the cluster's canonical phrase.
    This lets the TF-IDF+SVD semantic layer "see" the equivalence too, not
    just the explicit skill-match scorer -- closing the gap between
    skill-intelligence and career-narrative semantic similarity without
    requiring a heavier embedding model.

    PHASE 2: career_history entries are now read off CandidateProfile in
    chronological (ascending) order -- consistent with how the rest of the
    pipeline now treats career history as an ordered sequence -- instead of
    whatever order the raw JSON array happened to be in.
    """
    parts = [
        profile.headline,
        profile.summary,
        profile.current_title,
    ]
    for e in profile.career_history:
        parts.append(e.title)
        parts.append(e.description)

    if skill_f and skill_f.get("cluster_skill_hits"):
        canonical_terms = {
            hit.split("\u2192", 1)[1].split(" (adjacent)")[0]
            for hit in skill_f["cluster_skill_hits"]
        }
        parts.append(" ".join(sorted(canonical_terms)))

    return " ".join(p for p in parts if p)
