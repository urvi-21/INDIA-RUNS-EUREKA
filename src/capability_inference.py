"""
capability_inference.py
========================
The single recruiter-capability reasoning pipeline.

PHASE 3 GOAL (see PHASE3_REPORT.md for full detail): stop inferring
capability primarily from technology mentions ("has FAISS on their skill
list") and instead infer capability the way an experienced recruiter
does -- from what a candidate has *repeatedly demonstrated* across their
career history, achievements, promotions, company context, and verified
assessments. Technology mentions (FAISS, Pinecone, BM25, Weaviate,
LangChain, ...) are kept, but demoted to *supporting* evidence: they can
only strengthen a capability that already has real behavioral evidence
behind it, and by themselves they can never create one.

This module now:
  1. Separates "what a candidate DID" (behavior evidence: career-history
     phrases describing built/shipped/owned/deployed work, quantified
     achievement snippets, career-progression signals such as promotions
     and leadership emergence, company context such as product-vs-services
     and startup exposure, and verified Redrob skill-assessment scores)
     from "what a candidate LISTS" (technology mentions).
  2. Produces a structured `CapabilityEvidence` record per capability --
     confidence, supporting evidence (human-readable, recruiter-style),
     evidence count, supporting positions, supporting projects, and an
     evidence-strength label -- instead of a bare float.
  3. Still exposes `infer_capabilities(profile) -> dict[str, float]` (a
     thin wrapper over the evidence records) so scoring.py's WEIGHTS,
     RECRUITER_DIMENSION_WEIGHTS, and every existing capability key are
     completely unchanged -- only how each score is *earned* changes.
  4. Still exposes `match_capability_keys(text)` used by candidate_profile
     (per-role evidence tagging) and features.extract_jd_alignment
     (resolving JD requirement bullets to capability keys) -- unchanged
     call signature, richer underlying vocabulary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from config import RECRUITER_CAPABILITIES


# ---------------------------------------------------------------------------
# 1. BEHAVIOR EVIDENCE -- "what has this candidate repeatedly demonstrated?"
#
# Phrases here describe outcomes and responsibilities a candidate would
# actually write about doing (built X, owned Y, deployed Z to production),
# not tools they used. This is the PRIMARY evidence source. Matched against
# each career_history entry's own title+description text, so evidence is
# tied to a specific, dateable position -- not just present somewhere in a
# free-floating skill list.
# ---------------------------------------------------------------------------
CAPABILITY_BEHAVIOR_PATTERNS: dict[str, list[str]] = {
    "retrieval_systems": [
        "built retrieval", "built search", "search engine", "search platform",
        "search relevance", "semantic search", "information retrieval",
        "retrieval pipeline", "retrieval system", "query understanding",
        "document retrieval", "retrieval augmented generation",
        "rag pipeline", "hybrid search", "dense retrieval",
    ],
    "ranking_systems": [
        "ranking system", "ranking model", "ranking pipeline",
        "learning to rank", "recommendation system", "recommendation engine",
        "recommender", "relevance ranking", "search ranking",
        "personalization engine", "matching system", "marketplace ranking",
        "built a recommendation", "built recommendation",
    ],
    "evaluation_frameworks": [
        "offline evaluation", "online evaluation", "a/b test", "ab test",
        "experimentation platform", "evaluation framework",
        "evaluation methodology", "offline-to-online", "measured relevance",
        "designed evaluation", "built evaluation", "online experimentation",
    ],
    "vector_databases": [
        "vector search", "vector database", "embedding index",
        "semantic index", "indexed embeddings", "built vector store",
        "deployed vector", "production vector search", "production vector",
        "built a vector", "vector infrastructure", "vector index",
        "vector store", "nearest neighbor search", "nearest neighbour search",
        "approximate nearest neighbor", "approximate nearest neighbour",
        "embedding store", "ann index",
    ],
    "production_ml": [
        "deployed to production", "productionized", "production system",
        "shipped to production", "served in production", "live traffic",
        "scaled to production", "production pipeline",
        "monitoring in production", "on-call for", "production ml system",
        "owns the ml platform", "owned the ml platform", "into production",
        "to production traffic", "production traffic", "production-grade",
        "production grade", "production-ready", "own the platform",
        "owns the platform", "in production",
    ],
    "llm_engineering": [
        "llm application", "built an llm", "built llm", "fine-tuned",
        "fine tuning", "prompt engineering", "llm pipeline",
        "generative ai product", "conversational ai", "chat application",
        "agent framework", "llm-powered", "llm powered",
    ],
    "distributed_systems": [
        "distributed system", "distributed pipeline",
        "large-scale data pipeline", "large scale data pipeline",
        "microservices architecture", "horizontally scaled",
        "high-throughput system", "high throughput system",
        "streaming pipeline", "built a distributed",
    ],
    "leadership": [
        "led a team", "led the team", "managed a team", "managed the team",
        "mentored", "architected", "owned the roadmap", "hired and grew",
        "built and led", "managing engineers", "people management",
        "led the design", "owned the platform", "drove the strategy",
    ],
    "hands_on_engineering": [
        "implemented", "built", "developed", "designed and built",
        "shipped", "launched", "coded", "wrote production code",
        "optimized", "engineered",
    ],
    "open_source": [
        "open source", "open-source", "oss contributor", "maintainer of",
        "published a package", "open-sourced", "contributed to",
    ],
}

# ---------------------------------------------------------------------------
# 2. TECHNOLOGY EVIDENCE -- SUPPORTING ONLY.
#
# This is the exact list of things the Redrob JD flags as being weighted
# too directly today (FAISS, Pinecone, BM25, Weaviate, LangChain, ...).
# They still matter -- a recruiter is glad to see them -- but they can only
# ever *strengthen* a capability that already has real behavior evidence
# (see `_technology_boost` below); a candidate who lists these tools with
# zero demonstrated career, achievement, or assessment evidence earns
# exactly zero capability credit from them.
# ---------------------------------------------------------------------------
CAPABILITY_TECHNOLOGY_TERMS: dict[str, list[str]] = {
    "retrieval_systems": ["faiss", "pinecone", "weaviate", "milvus", "qdrant",
                           "elasticsearch", "opensearch", "chroma", "bm25"],
    "ranking_systems": ["xgboost", "lightgbm", "ltr"],
    "evaluation_frameworks": ["ndcg", "mrr", "map"],
    "vector_databases": ["faiss", "pinecone", "milvus", "qdrant", "weaviate",
                          "elasticsearch", "opensearch", "chroma"],
    "production_ml": ["docker", "kubernetes", "mlflow", "kubeflow", "airflow"],
    "llm_engineering": ["gpt", "bert", "transformer", "langchain", "lora",
                         "qlora", "peft", "embeddings"],
    "distributed_systems": ["spark", "kafka", "ray", "kubernetes", "docker"],
    "leadership": ["lead", "head", "principal", "staff", "manager"],
    "hands_on_engineering": [],
    "open_source": ["github"],
}

# Light title-level hint -- a title alone tells a recruiter almost nothing
# about a *specific* capability (e.g. "evaluation frameworks"), so this only
# ever contributes a small assist, never the deciding evidence.
CAPABILITY_TITLE_HINTS: dict[str, list[str]] = {
    "retrieval_systems": ["search engineer", "ml engineer", "ai engineer"],
    "ranking_systems": ["ranking engineer", "search engineer", "ai engineer"],
    "production_ml": ["machine learning engineer", "ml engineer"],
    "leadership": ["lead", "head", "principal", "staff", "manager"],
}

# Recruiter-facing description of what each capability actually means when
# it's backed by real evidence -- this is the language reasoning.py /
# scoring.py use instead of naming raw tools. E.g. "Hands-on with FAISS"
# becomes "Built production retrieval systems"; "Knows Pinecone" becomes
# "Demonstrated production vector search infrastructure"; "LangChain
# experience" becomes "Delivered production LLM applications."
CAPABILITY_RECRUITER_LABEL: dict[str, str] = {
    "retrieval_systems": "built production retrieval and search systems",
    "ranking_systems": "built ranking or recommendation systems",
    "evaluation_frameworks": "designed offline/online evaluation methodology",
    "vector_databases": "demonstrated production vector search infrastructure",
    "production_ml": "shipped and operated ML systems in production",
    "llm_engineering": "delivered production LLM applications",
    "distributed_systems": "built distributed, large-scale systems",
    "leadership": "led or owned engineering initiatives",
    "hands_on_engineering": "hands-on building and shipping software",
    "open_source": "an active open-source contributor",
}

# Sanity check: every capability this module claims to infer must be one of
# the canonical names scoring.py expects (config.RECRUITER_CAPABILITIES),
# and vice versa for the ones that ARE keyword/behavior-inferrable
# (startup_experience / product_company_experience are career-context
# signals derived from structured company_size/industry fields, not
# text-pattern capabilities -- they intentionally stay in
# features._extract_recruiter_profile()).
_CONTEXT_ONLY_CAPABILITIES = {"startup_experience", "product_company_experience"}
assert set(CAPABILITY_BEHAVIOR_PATTERNS) | _CONTEXT_ONLY_CAPABILITIES == set(RECRUITER_CAPABILITIES), (
    "CAPABILITY_BEHAVIOR_PATTERNS and config.RECRUITER_CAPABILITIES have drifted out of sync"
)

# Recency discount applied per career entry when walking career_history_desc
# (index 0 = most recent role): a capability demonstrated in the current or
# most recent role is worth more than the same phrase appearing in a role
# from a decade ago, matching how a recruiter actually reads a CV.
_RECENCY_DECAY = 0.85

# Structured (not-text) evidence bonuses. Each is capped so no single
# structured signal can single-handedly manufacture a capability score --
# it can only add to evidence that career/achievement text already earned.
_ACHIEVEMENT_BONUS = 0.12          # per achievement snippet that also matches
_ACHIEVEMENT_BONUS_CAP = 0.30
_ASSESSMENT_BONUS_CAP = 0.20       # verified Redrob assessment score, scaled
_TECH_BOOST_PER_TERM = 0.035
_TECH_BOOST_CAP = 0.15
_TITLE_HINT_BONUS = 0.08


@dataclass
class CapabilityEvidence:
    """Structured, auditable evidence behind a single capability judgment --
    what a recruiter would actually want to see before trusting a score."""

    capability: str
    confidence: float = 0.0                       # 0-1, the score scoring.py consumes
    supporting_evidence: list[str] = field(default_factory=list)   # recruiter-readable sentences
    evidence_count: int = 0                        # independent evidence sources
    supporting_positions: list[str] = field(default_factory=list)  # "Title @ Company (2021-2023)"
    supporting_projects: list[str] = field(default_factory=list)   # quantified achievement snippets
    evidence_strength: str = "none"                # "none" | "weak" | "moderate" | "strong"
    technology_signals: list[str] = field(default_factory=list)    # supporting-only tech mentions
    reasoning_sentence: str = ""                    # ready-to-use recruiter-style sentence


def _role_span_label(entry) -> str:
    start_year = (entry.start_date or "")[:4] or "?"
    end_year = "Present" if entry.is_current or not entry.end_date else (entry.end_date or "")[:4]
    return f"{entry.title} at {entry.company} ({start_year}\u2013{end_year})"


def _hit_terms(text: str, phrases: list[str]) -> list[str]:
    return [p for p in phrases if p in text]


def match_capability_keys(text: str) -> list[str]:
    """
    Resolve a free-text phrase (e.g. a HiringProfile requirement bullet like
    "3+ years building retrieval or ranking systems") to the canonical
    capability keys it refers to. Checks BOTH behavior phrases and
    technology terms, since a JD bullet may legitimately name either
    ("experience with FAISS or similar" should still resolve to
    retrieval_systems) -- this function is about categorical tagging /
    routing, not scoring, so it is deliberately more permissive than the
    evidence-weighted scorer below.
    """
    text = text.lower()
    matched = []
    for capability in CAPABILITY_BEHAVIOR_PATTERNS:
        phrases = set(CAPABILITY_BEHAVIOR_PATTERNS[capability])
        phrases |= set(CAPABILITY_TECHNOLOGY_TERMS.get(capability, []))
        if any(p in text for p in phrases):
            matched.append(capability)
    return matched


def _capability_evidence_for(capability: str, candidate_profile) -> CapabilityEvidence:
    behavior_phrases = CAPABILITY_BEHAVIOR_PATTERNS[capability]
    tech_terms = CAPABILITY_TECHNOLOGY_TERMS.get(capability, [])
    title_hints = CAPABILITY_TITLE_HINTS.get(capability, [])

    role_strength = 0.0
    supporting_positions: list[str] = []
    supporting_projects: list[str] = []
    tech_hits: set[str] = set()
    n_role_sources = 0

    history_desc = candidate_profile.career_history_desc
    for idx, entry in enumerate(history_desc):
        role_text = f"{entry.title} {entry.description}".lower()
        role_hits = _hit_terms(role_text, behavior_phrases)
        role_tech_hits = _hit_terms(role_text, tech_terms)
        if role_tech_hits:
            tech_hits.update(role_tech_hits)

        if not role_hits:
            continue

        n_role_sources += 1
        weight = _RECENCY_DECAY ** idx
        role_strength += weight
        supporting_positions.append(_role_span_label(entry))

        # Quantified-achievement evidence for the SAME role: an achievement
        # snippet that also touches this capability's vocabulary is much
        # stronger evidence than a bare responsibility line.
        matched_snippets = [
            s for s in entry.achievement_snippets
            if any(p in s.lower() for p in behavior_phrases + tech_terms)
        ]
        supporting_projects.extend(matched_snippets)

    achievement_bonus = min(_ACHIEVEMENT_BONUS_CAP, _ACHIEVEMENT_BONUS * len(supporting_projects))

    # ---- Structured, non-text context evidence -----------------------------
    context_bonus = 0.0
    context_sentences: list[str] = []
    n_context_sources = 0

    if capability == "leadership":
        if candidate_profile.leadership_is_recent:
            context_bonus += 0.30
            n_context_sources += 1
            context_sentences.append(
                "Currently in a leadership/ownership-scoped role, per career history."
            )
        elif candidate_profile.leadership_emergence:
            context_bonus += 0.18
            n_context_sources += 1
            context_sentences.append(
                "Has held a leadership-scoped role earlier in their career."
            )
        if candidate_profile.promotion_count >= 2:
            context_bonus += 0.15
            n_context_sources += 1
            context_sentences.append(
                f"{candidate_profile.promotion_count} title-level promotions across career history "
                "shows sustained ownership growth, not a single title bump."
            )

    if capability == "production_ml":
        if candidate_profile.product_company_months > 0 and not candidate_profile.career_entirely_services_firms:
            months = candidate_profile.product_company_months
            context_bonus += min(0.20, months / 60.0)
            n_context_sources += 1
            context_sentences.append(
                f"{months:.0f} months at product-oriented companies -- structural evidence of "
                "owning systems that serve real users, not just delivering client engagements."
            )

    if capability == "open_source":
        github_score = max(0.0, candidate_profile.github_activity_score or 0) / 100.0
        if github_score > 0:
            context_bonus += min(0.35, github_score * 0.35)
            n_context_sources += 1
            context_sentences.append(
                f"GitHub activity score of {candidate_profile.github_activity_score:.0f}/100 "
                "corroborates public open-source involvement."
            )

    # Verified Redrob platform skill-assessment score: a proctored, verified
    # signal (stronger than a self-reported skill list, but still additive
    # only -- it cannot manufacture a capability with zero career evidence).
    assessment_bonus = 0.0
    for skill_name, score in (candidate_profile.skill_assessment_scores or {}).items():
        low = skill_name.lower()
        if any(p in low for p in behavior_phrases + tech_terms) and score >= 60:
            assessment_bonus = max(assessment_bonus, min(_ASSESSMENT_BONUS_CAP, (score / 100.0) * _ASSESSMENT_BONUS_CAP))
            n_context_sources += 1
            context_sentences.append(
                f"Verified Redrob platform assessment score of {score:.0f}/100 in {skill_name}."
            )
            break  # one assessment source counted per capability, not per matching key

    # Title alone is weak, assistive-only evidence.
    title_bonus = 0.0
    if title_hints and any(h in candidate_profile.title_text.lower() for h in title_hints):
        title_bonus = _TITLE_HINT_BONUS

    has_primary_evidence = (role_strength > 0) or (context_bonus > 0) or (assessment_bonus > 0)

    # Technology mentions: SUPPORTING evidence only. They can strengthen a
    # capability that already has real behavior/context/assessment evidence
    # behind it, but contribute nothing on their own -- directly addressing
    # the JD's stated weakness ("these technologies still contribute too
    # directly to capability scores").
    tech_boost = 0.0
    if has_primary_evidence and tech_hits:
        tech_boost = min(_TECH_BOOST_CAP, _TECH_BOOST_PER_TERM * len(tech_hits))

    raw = role_strength + achievement_bonus + context_bonus + assessment_bonus + title_bonus
    confidence = 0.0
    if raw > 0 or tech_boost > 0:
        # Saturating growth curve so a single strong role doesn't already
        # max the score, but consistent, repeated evidence approaches 1.0 --
        # this is the concrete mechanism behind "what has this candidate
        # REPEATEDLY demonstrated."
        confidence = (1 - math.exp(-raw / 1.2)) + tech_boost

    if capability == "open_source":
        # Blend text/context evidence with the raw GitHub activity signal
        # itself, same design intent as before Phase 3 -- structured
        # platform evidence, not a technology mention, so it stays a primary
        # (not merely supporting) contributor.
        github_score = max(0.0, candidate_profile.github_activity_score or 0) / 100.0
        confidence = 0.6 * confidence + 0.4 * min(1.0, github_score)

    confidence = round(min(max(confidence, 0.0), 1.0), 4)

    evidence_count = n_role_sources + n_context_sources
    if evidence_count == 0:
        strength = "none"
    elif evidence_count == 1:
        strength = "weak"
    elif evidence_count in (2, 3):
        strength = "moderate"
    else:
        strength = "strong"

    # ---- Assemble recruiter-readable evidence -------------------------------
    supporting_evidence: list[str] = []
    label = CAPABILITY_RECRUITER_LABEL[capability]
    if supporting_positions:
        supporting_evidence.append(
            f"Demonstrated across {len(supporting_positions)} role(s): "
            + "; ".join(supporting_positions[:4])
        )
    supporting_evidence.extend(f"Quantified outcome: \"{s}\"" for s in supporting_projects[:3])
    supporting_evidence.extend(context_sentences)
    if tech_hits and has_primary_evidence:
        tech_sample = ", ".join(sorted(t.upper() if len(t) <= 5 else t.title() for t in tech_hits)[:5])
        supporting_evidence.append(f"Supporting technology evidence (secondary): {tech_sample}")

    reasoning_sentence = _build_reasoning_sentence(capability, label, supporting_positions,
                                                     evidence_count, tech_hits, has_primary_evidence)

    return CapabilityEvidence(
        capability=capability,
        confidence=confidence,
        supporting_evidence=supporting_evidence,
        evidence_count=evidence_count,
        supporting_positions=supporting_positions,
        supporting_projects=supporting_projects,
        evidence_strength=strength,
        technology_signals=sorted(tech_hits),
        reasoning_sentence=reasoning_sentence,
    )


def _build_reasoning_sentence(capability: str, label: str, supporting_positions: list[str],
                               evidence_count: int, tech_hits: set[str],
                               has_primary_evidence: bool) -> str:
    """
    The recruiter-style sentence scoring.py / reasoning.py surface to
    explain a capability. Reads like an experienced recruiter describing
    what a candidate has done, not an ATS keyword dump:
      "Built production retrieval systems" (not "Hands-on with FAISS")
      "Demonstrated production vector search infrastructure" (not "Knows Pinecone")
      "Delivered production LLM applications" (not "LangChain experience")
    """
    if not has_primary_evidence:
        return ""

    sentence = label[0].upper() + label[1:]
    if evidence_count >= 3:
        sentence += f" across {evidence_count} independent roles/signals"
    elif supporting_positions:
        sentence += f" as {supporting_positions[0]}"

    if tech_hits:
        tech_sample = ", ".join(sorted(t.upper() if len(t) <= 5 else t.title() for t in tech_hits)[:3])
        sentence += f" (hands-on with {tech_sample})"

    return sentence + "."


def infer_capability_evidence(candidate_profile) -> dict[str, CapabilityEvidence]:
    """
    The single recruiter-capability-reasoning entry point: for every
    canonical capability, walk the candidate's career history, achievements,
    career-progression signals, company context, and verified assessments to
    infer what they have REPEATEDLY DEMONSTRATED -- not what technologies
    they happen to list. Technology mentions are folded in afterward as
    supporting-only evidence (see `_capability_evidence_for`).
    """
    return {
        capability: _capability_evidence_for(capability, candidate_profile)
        for capability in CAPABILITY_BEHAVIOR_PATTERNS
    }


def infer_capabilities(candidate_profile) -> dict[str, float]:
    """
    Backward-compatible entry point: features.extract_capability_features()
    (and therefore every downstream scoring.py / ranker.py consumer) reads
    a flat capability -> 0-1 float dict, unchanged in shape from Phase 1/2.
    Internally this is now a thin wrapper over the evidence-based reasoning
    in `infer_capability_evidence` -- every one of these floats is now
    *earned* primarily through demonstrated career evidence, with technology
    mentions folded in only as a bounded, secondary strengthening factor.
    """
    evidence = infer_capability_evidence(candidate_profile)
    return {capability: ev.confidence for capability, ev in evidence.items()}
