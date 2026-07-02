from __future__ import annotations
import math
from dataclasses import dataclass, field

from config import (
    WEIGHTS, EXPERIENCE_BAND_MIN, EXPERIENCE_BAND_MAX, EXPERIENCE_SOFT_MARGIN,
    BEHAVIORAL_MULTIPLIER_MIN, BEHAVIORAL_MULTIPLIER_MAX,
    NOTICE_PERIOD_IDEAL_DAYS, HONEYPOT_SCORE_FLOOR,
    RECRUITER_DIMENSION_WEIGHTS, CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD, CONFIDENCE_MIN_COMPLETENESS_FOR_HIGH,
    BEHAVIORAL_RECENCY_HALFLIFE_DAYS,
)


@dataclass
class ScoreBreakdown:
    candidate_id: str
    fit_score: float
    behavioral_multiplier: float
    final_score: float
    component_scores: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    concerns: list = field(default_factory=list)
    is_honeypot: bool = False
    honeypot_reasons: list = field(default_factory=list)
    # Recruiter-intelligence layer (internal/explainability only -- not part
    # of the validator-required CSV columns).
    recruiter_dimensions: dict = field(default_factory=dict)
    confidence_score: float = 0.0
    confidence_label: str = "unknown"
    # PHASE 3: structured, evidence-based capability reasoning (see
    # capability_inference.CapabilityEvidence) -- one entry per capability
    # with confidence, supporting evidence, evidence count/strength, and
    # supporting positions/projects. Internal/explainability only, same as
    # recruiter_dimensions -- not part of the validator-required CSV.
    capability_evidence: dict = field(default_factory=dict)


def _lower_first(s: str) -> str:
    """Reason clauses in this codebase are lowercase-led, comma-joined
    fragments without their own trailing period (build_reasoning() appends
    exactly one period at the end of the whole joined sentence and
    capitalizes only its very first character). Capability reasoning
    sentences are stored as capitalized, period-terminated standalone
    sentences (they're also surfaced verbatim in the explanations JSON), so
    when folding one into the reasons list, lowercase the lead character and
    drop the trailing period."""
    if not s:
        return s
    s = s.rstrip(".")
    return s[0].lower() + s[1:] if s else s


def _triangular_band_score(value: float, lo: float, hi: float, margin: float) -> float:
    
    if lo <= value <= hi:
        return 1.0
    if value < lo:
        return max(0.0, 1 - (lo - value) / margin)
    return max(0.0, 1 - (value - hi) / margin)


def score_title_seniority(feats: dict) -> tuple[float, list[str], list[str]]:
    reasons, concerns = [], []
    tier = feats["title_tier"]
    if tier == "core":
        s = 1.0
        reasons.append(f"current title '{feats['current_title']}' is a direct match for the role")
    elif tier == "adjacent":
        s = 0.55
        reasons.append(f"current title '{feats['current_title']}' is adjacent (not a direct ML/AI title)")
    else:
        s = 0.05
        concerns.append(f"current title '{feats['current_title']}' has no ML/AI/IR signal")

    if feats["is_stale_ic"]:
        s *= 0.4
        concerns.append("has been in an architecture/lead role 18+ months without recent production coding")

    return s, reasons, concerns


def score_career_narrative(semantic_similarity: float) -> tuple[float, list[str], list[str]]:
    """
    Semantic similarity measures whether the overall career trajectory aligns
    with the hiring intent rather than literal keyword overlap.
    """

    s = max(0.0, min(1.0, semantic_similarity))

    reasons = []
    concerns = []

    if s >= 0.80:
        reasons.append("career evidence strongly aligns with the hiring requirements")

    elif s >= 0.60:
        reasons.append("career trajectory aligns well with the required capabilities")

    elif s >= 0.40:
        reasons.append("career history shows partial alignment with the role")

    elif s < 0.20:
        concerns.append("limited evidence supporting the required capabilities")

    return s, reasons, concerns


def score_skills_trust(feats: dict) -> tuple[float, list[str], list[str]]:
    core = feats["core_skill_trust"]
    support = feats["support_skill_trust"]
    off_domain = feats["off_domain_skill_trust"]
    cluster = feats.get("cluster_skill_trust", 0.0)

    # cluster_skill_trust captures equivalent/adjacent technologies (RAG,
    # AI Orchestration, etc.) the literal CORE_SKILLS/SUPPORT_SKILLS lists
    # would otherwise miss -- weighted like core skills since clusters are
    # already discounted at the source for adjacency (see
    # SKILL_ADJACENT_CREDIT_FRACTION in config.py).
    raw = core + cluster + 0.4 * support
    s = 1 - pow(2.71828, -raw / 15.0)  # smooth saturating curve, asymptotes to 1

    reasons, concerns = [], []
    if feats["core_skill_hits"]:
        # PHASE 3: lead with what the candidate has demonstrated (the
        # strongest evidence-backed capability), and mention specific tools
        # only as a trailing, supporting aside -- a recruiter says "built
        # production retrieval systems (hands-on with FAISS, Pinecone)," not
        # "hands-on with core skills: faiss, pinecone, weaviate."
        sample = ", ".join(feats["core_skill_hits"][:3])
        headline = feats.get("capability_headline", "")
        if headline and "hands-on with" in headline:
            # The capability reasoning sentence already names its own
            # supporting technology evidence -- don't repeat the clause.
            reasons.append(_lower_first(headline))
        elif headline:
            reasons.append(f"{_lower_first(headline)} (hands-on with {sample})")
        else:
            reasons.append(f"demonstrated hands-on depth across core technical skills ({sample})")
    if feats.get("cluster_skill_hits"):
        sample = ", ".join(h.split("\u2192", 1)[1] for h in feats["cluster_skill_hits"][:2])
        reasons.append(f"equivalent/adjacent technology coverage in {sample}")
    if off_domain > core and off_domain > 5:
        concerns.append("skill list is weighted toward vision/speech rather than NLP/IR/ranking")
    if feats["n_skills_total"] > 10 and core < 3 and cluster < 3 and off_domain < 3:
        concerns.append("long skill list but little endorsed/long-tenure depth in any of them")

    return max(0.0, min(1.0, s)), reasons, concerns


def score_experience_years(feats: dict) -> tuple[float, list[str], list[str]]:
    yoe = feats["years_of_experience"]
    s = _triangular_band_score(
        yoe, EXPERIENCE_BAND_MIN, EXPERIENCE_BAND_MAX, EXPERIENCE_SOFT_MARGIN
    )
    reasons, concerns = [], []
    if EXPERIENCE_BAND_MIN <= yoe <= EXPERIENCE_BAND_MAX:
        reasons.append(f"{yoe:.1f} years of experience sits inside the JD's 5-9y band")
    elif yoe < EXPERIENCE_BAND_MIN:
        concerns.append(f"only {yoe:.1f} years of experience, below the 5-9y band")
    else:
        concerns.append(f"{yoe:.1f} years of experience, above the 5-9y band (not disqualifying per JD)")

    if not feats["earlier_has_ml_signal"] and feats["title_tier"] in ("core", "adjacent"):
        concerns.append("ML/AI signal appears recent only; limited evidence of pre-LLM-era production ML")

    return s, reasons, concerns


def score_career_trajectory(feats: dict) -> tuple[float, list[str], list[str]]:
    """
    Scores career_history as a *sequence*: promotion velocity, tenure
    stability, leadership emergence, domain consistency, specialization
    direction. Each sub-signal is a simple, auditable 0-1 contribution;
    nothing here depends on data outside career_history/profile fields
    already present in the schema.
    """
    reasons, concerns = [], []

    # Promotion velocity: 1 promotion per ~3 years is treated as a strong,
    # fully-credited signal; scaled and capped so a single early promotion
    # in a long career isn't over-rewarded relative to sustained growth.
    promo_s = min(1.0, feats["promotion_velocity"] / 1.0)
    if feats["promotion_count"] >= 2:
        reasons.append(f"{feats['promotion_count']} title-level promotions across career history")
    elif feats["promotion_count"] == 0 and feats["highest_seniority_rank"] >= 2:
        concerns.append("no detected promotions across recorded career history")

    # Stability: stable/moderate tenure is rewarded; frequent-mover is a
    # soft concern only (per JD: never a hard disqualifier on its own).
    stability_score = {"stable": 1.0, "moderate": 0.75, "frequent_mover": 0.4, "unknown": 0.5}
    stab_s = stability_score[feats["tenure_stability"]]
    if feats["tenure_stability"] == "stable":
        reasons.append(f"average tenure of {feats['avg_tenure_months']:.0f} months per role shows stability")
    elif feats["tenure_stability"] == "frequent_mover":
        concerns.append(f"average tenure of only {feats['avg_tenure_months']:.0f} months per role")

    # Leadership: recent leadership is the strongest signal; past-but-not-
    # recent leadership still earns partial credit (a former lead now in an
    # IC role is not penalized, just not boosted as much).
    if feats["leadership_is_recent"]:
        lead_s = 1.0
        reasons.append("currently in a leadership/ownership-scoped role")
    elif feats["leadership_emergence"]:
        lead_s = 0.6
    else:
        lead_s = 0.3

    # Domain consistency + specialization direction.
    domain_s = feats["domain_consistency"]
    if feats["specialization_drift"] == "narrowing_to_domain":
        domain_s = min(1.0, domain_s + 0.15)
        reasons.append("career trajectory shows increasing specialization toward this domain")
    elif feats["specialization_drift"] == "broadening_away_from_domain":
        domain_s = max(0.0, domain_s - 0.15)
        concerns.append("recent roles have drifted away from the core domain")

    s = 0.30 * promo_s + 0.25 * stab_s + 0.25 * lead_s + 0.20 * domain_s
    return max(0.0, min(1.0, s)), reasons, concerns


def score_company_quality(feats: dict) -> tuple[float, list[str], list[str]]:
    reasons, concerns = [], []
    if feats["career_entirely_services_firms"]:
        concerns.append("entire career history is at services/consulting firms (TCS/Infosys/Wipro-type)")
        return 0.1, reasons, concerns
    return 0.8, reasons, concerns


def score_location(feats: dict) -> tuple[float, list[str], list[str]]:
    reasons, concerns = [], []
    tier = feats["location_tier"]
    mapping = {
        "preferred": 1.0, "welcome": 0.85, "india_other": 0.55, "outside_india": 0.15,
    }
    s = mapping[tier]
    if tier == "preferred":
        reasons.append(f"based in {feats['raw_location']}, matching the JD's preferred locations")
    elif tier == "outside_india":
        concerns.append(f"based outside India ({feats['raw_location']}); JD does not sponsor visas")
    return s, reasons, concerns


def score_education(feats: dict) -> tuple[float, list[str], list[str]]:
    mapping = {"tier_1": 1.0, "tier_2": 0.7, "tier_3": 0.5, "tier_4": 0.35, "unknown": 0.4}
    return mapping.get(feats["best_tier"], 0.4), [], []

def score_recruiter_capabilities(feats: dict):

    reasons = []
    concerns = []

    capability_scores = [
        feats.get("retrieval_systems", 0),
        feats.get("ranking_systems", 0),
        feats.get("evaluation_frameworks", 0),
        feats.get("vector_databases", 0),
        feats.get("production_ml", 0),
        feats.get("llm_engineering", 0),
        feats.get("distributed_systems", 0),
        feats.get("startup_experience", 0),
        feats.get("product_company_experience", 0),
        feats.get("leadership", 0),
        feats.get("hands_on_engineering", 0),
        feats.get("open_source", 0),
    ]

    # PHASE 1: fold in HiringProfile requirement-alignment when a
    # HiringProfile was actually built for this run (features.extract_
    # jd_alignment only adds these keys when a hiring_profile was passed
    # into extract_all_features). Previously these were computed and then
    # never read anywhere -- "inferred but ignored" -- gated on presence
    # so behavior is byte-identical to before when no HiringProfile is
    # available (e.g. every existing unit test).
    if "required_capability_match" in feats:
        capability_scores.append(feats["required_capability_match"])
    if "preferred_capability_match" in feats:
        capability_scores.append(0.5 * feats["preferred_capability_match"])

    score = sum(capability_scores) / len(capability_scores)

    # PHASE 3: prefer the evidence-backed, recruiter-style sentence
    # capability_inference already built for this candidate (e.g. "Built
    # production retrieval and search systems across 3 independent
    # roles/signals (hands-on with FAISS, Pinecone).") over a generic
    # templated line -- it names what was actually demonstrated, not just
    # that a threshold was crossed. Falls back to the old generic phrasing
    # only if no reasoning sentence was produced (e.g. edge-case candidates
    # with almost no evidence anywhere).
    cap_reasoning = feats.get("capability_reasoning", {})

    if feats.get("required_capability_match", 0) >= 0.60:
        reasons.append("directly matches the JD's stated required capabilities")

    if feats.get("production_ml", 0) >= 0.60:
        reasons.append(_lower_first(cap_reasoning.get("production_ml", "")) or "strong production engineering evidence")

    if feats.get("retrieval_systems", 0) >= 0.60:
        reasons.append(_lower_first(cap_reasoning.get("retrieval_systems", "")) or "demonstrated information retrieval expertise")

    if feats.get("ranking_systems", 0) >= 0.60:
        reasons.append(_lower_first(cap_reasoning.get("ranking_systems", "")) or "experience building ranking or recommendation systems")

    if feats.get("evaluation_frameworks", 0) >= 0.60:
        reasons.append(_lower_first(cap_reasoning.get("evaluation_frameworks", "")) or "experience designing evaluation methodologies")

    if feats.get("vector_databases", 0) >= 0.60:
        reasons.append(_lower_first(cap_reasoning.get("vector_databases", "")) or "demonstrated production vector search infrastructure")

    if feats.get("llm_engineering", 0) >= 0.60:
        reasons.append(_lower_first(cap_reasoning.get("llm_engineering", "")) or "delivered production LLM applications")

    if feats.get("leadership", 0) >= 0.60:
        reasons.append(_lower_first(cap_reasoning.get("leadership", "")) or "led or owned engineering initiatives")

    if feats.get("product_company_experience", 0) >= 0.60:
        reasons.append("strong product engineering background")

    if feats.get("startup_experience", 0) >= 0.60:
        reasons.append("startup execution experience")

    if feats.get("hands_on_engineering", 0) < 0.30:
        concerns.append("limited evidence of recent hands-on engineering")

    if score < 0.35:
        concerns.append("limited evidence supporting the required technical capabilities")

    return score, reasons, concerns

def compute_behavioral_multiplier(feats: dict) -> tuple[float, list[str], list[str]]:

    reasons = []
    concerns = []

    # Existing behavioral intelligence
    availability = feats.get("availability_score", 0.5)
    confidence = feats.get("recruiter_confidence", 0.5)
    reliability = feats.get("reliability_score", 0.5)
    hiring = feats.get("hiring_readiness", 0.5)
    demand = feats.get("market_demand_score", 0.5)

    score = (
        0.30 * availability +
        0.25 * reliability +
        0.20 * confidence +
        0.15 * hiring +
        0.10 * demand
    )

    # ---------- Existing recruiter rules ----------

    if feats["is_inactive_180d"]:
        score *= 0.75
        concerns.append(
            f"inactive for {feats['days_inactive']} days"
        )

    if not feats["open_to_work_flag"]:
        score *= 0.90
        concerns.append("not currently open to work")

    if feats["notice_period_days"] <= NOTICE_PERIOD_IDEAL_DAYS:
        reasons.append(
            f"{feats['notice_period_days']}-day notice period"
        )
    else:
        score *= 0.95

    if feats["availability_score"] > 0.75:
        reasons.append("high hiring availability")

    if feats["reliability_score"] > 0.75:
        reasons.append("strong interview reliability")

    if feats["recruiter_confidence"] > 0.75:
        reasons.append("trusted recruiter engagement")

    if feats["market_demand_score"] > 0.70:
        reasons.append("high recruiter demand")

    score = max(
        BEHAVIORAL_MULTIPLIER_MIN,
        min(BEHAVIORAL_MULTIPLIER_MAX, score)
    )

    return score, reasons, concerns


def compute_recruiter_dimensions(component_scores: dict, behav_mult: float) -> dict:
    
    pool = dict(component_scores)
    pool["_behavioral_multiplier"] = behav_mult
    # Inverse-of-fit / inverse-of-behavioral feed "hiring_risk": risk rises
    # as the underlying fit and reachability signals fall.
    avg_fit = sum(WEIGHTS[k] * v for k, v in component_scores.items())
    pool["_inverse_fit"] = 1.0 - avg_fit
    pool["_inverse_behavioral"] = 1.0 - behav_mult

    dims = {}
    for dim_name, formula in RECRUITER_DIMENSION_WEIGHTS.items():
        val = sum(weight * pool.get(src, 0.0) for src, weight in formula.items())
        dims[dim_name] = round(max(0.0, min(1.0, val)), 4)
    return dims


def compute_confidence(feats: dict, fit_score: float) -> tuple[float, str]:
    
    completeness = feats.get("completeness_fraction", 0.5)
    skill_evidence = min(1.0, (feats.get("core_skill_trust", 0) + feats.get("cluster_skill_trust", 0)) / 10.0)
    career_evidence = min(1.0, feats.get("total_career_months", 0) / 24.0)

    confidence = 0.5 * completeness + 0.25 * skill_evidence + 0.25 * career_evidence
    confidence = max(0.0, min(1.0, confidence))

    if completeness < CONFIDENCE_MIN_COMPLETENESS_FOR_HIGH:
        label = "low" if confidence < CONFIDENCE_MEDIUM_THRESHOLD else "medium"
    elif confidence >= CONFIDENCE_HIGH_THRESHOLD:
        label = "high"
    elif confidence >= CONFIDENCE_MEDIUM_THRESHOLD:
        label = "medium"
    else:
        label = "low"

    return round(confidence, 4), label


def score_candidate(feats: dict, semantic_similarity: float) -> ScoreBreakdown:
    title_s, title_r, title_c = score_title_seniority(feats)
    narr_s, narr_r, narr_c = score_career_narrative(semantic_similarity)
    skill_s, skill_r, skill_c = score_skills_trust(feats)
    traj_s, traj_r, traj_c = score_career_trajectory(feats)
    exp_s, exp_r, exp_c = score_experience_years(feats)
    comp_s, comp_r, comp_c = score_company_quality(feats)
    loc_s, loc_r, loc_c = score_location(feats)
    edu_s, edu_r, edu_c = score_education(feats)
    cap_s, cap_r, cap_c = score_recruiter_capabilities(feats)
    behavior_score = feats.get("hiring_readiness", 0.5)
    confidence_behavior = feats.get("recruiter_confidence", 0.5)

    component_scores = {
        "title_seniority": title_s,
        "career_narrative": narr_s,
        "skills_trust": skill_s,
        "career_trajectory": traj_s,
        "experience_years": exp_s,
        "company_quality": comp_s,
        "location": loc_s,
        "education": edu_s,
        "recruiter_capabilities": cap_s,
    }

    fit_score = sum(WEIGHTS[k] * v for k, v in component_scores.items())

    behav_mult, behav_r, behav_c = compute_behavioral_multiplier(feats)
    final_score = fit_score * behav_mult

    if feats["is_honeypot"]:
        final_score = min(final_score, HONEYPOT_SCORE_FLOOR)

    reasons = title_r + narr_r + skill_r + traj_r + exp_r + comp_r + loc_r + edu_r + cap_r + behav_r
    concerns = title_c + narr_c + skill_c + traj_c + exp_c + comp_c + loc_c + edu_c + cap_c + behav_c
    # PHASE 3: score_skills_trust and score_recruiter_capabilities can both
    # surface the same evidence-backed capability sentence (e.g. a strong
    # retrieval_systems capability shows up as the skills-trust headline AND
    # crosses the recruiter_capabilities threshold) -- dedupe while
    # preserving order so the top-N reasons a recruiter reads aren't wasted
    # repeating the same clause twice.
    reasons = list(dict.fromkeys(reasons))
    concerns = list(dict.fromkeys(concerns))

    recruiter_dims = compute_recruiter_dimensions(component_scores, behav_mult)
    confidence_score, confidence_label = compute_confidence(feats, fit_score)

    return ScoreBreakdown(
        candidate_id=feats["candidate_id"],
        fit_score=fit_score,
        behavioral_multiplier=behav_mult,
        final_score=final_score,
        component_scores=component_scores,
        reasons=reasons,
        concerns=concerns,
        is_honeypot=feats["is_honeypot"],
        honeypot_reasons=feats.get("honeypot_reasons", []),
        recruiter_dimensions=recruiter_dims,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
        capability_evidence=feats.get("capability_evidence_summary", {}),
    )
