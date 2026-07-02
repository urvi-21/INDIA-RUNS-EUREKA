"""
test_pipeline.py
=================
Lightweight unit tests (no pytest dependency required -- run with
`python tests/test_pipeline.py`) covering the behaviors we most need to
prove to judges: honeypot detection, keyword-stuffer demotion, and
validator-compliant output ordering. These are not exhaustive, but they
cover the specific failure modes the dataset was designed to test for.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import extract_all_features
from scoring import score_candidate


def make_minimal_candidate(**overrides) -> dict:
    base = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "headline": "Software Engineer",
            "summary": "Generic profile.",
            "location": "Bengaluru, Karnataka",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "Software Engineer",
        },
        "career_history": [
            {
                "company": "Acme Corp", "title": "Software Engineer",
                "start_date": "2020-01-01", "end_date": None,
                "duration_months": 72, "description": "Generic backend work.",
            }
        ],
        "skills": [],
        "education": [{"tier": "tier_2"}],
        "redrob_signals": {
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.5,
            "interview_completion_rate": 0.5,
            "offer_acceptance_rate": -1,
            "notice_period_days": 30,
            "willing_to_relocate": False,
            "github_activity_score": 0.5,
            "profile_completeness_score": 70.0,
            "saved_by_recruiters_30d": 0,
            "search_appearance_30d": 0,
        },
    }
    for key, val in overrides.items():
        if isinstance(val, dict) and key in base:
            base[key].update(val)
        else:
            base[key] = val
    return base


def test_keyword_stuffer_demoted():
    """A non-AI-title candidate with many AI-sounding skills should still
    get title_tier='other' and near-zero core_skill_trust contribution
    relative to a genuine ML engineer."""
    stuffer = make_minimal_candidate(
        profile={"current_title": "Marketing Manager"},
        skills=[
            {"name": "NLP", "proficiency": "expert", "endorsements": 1, "duration_months": 0},
            {"name": "LLM", "proficiency": "expert", "endorsements": 1, "duration_months": 0},
            {"name": "Embeddings", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        ],
    )
    genuine = make_minimal_candidate(
        candidate_id="CAND_0000002",
        profile={"current_title": "Machine Learning Engineer"},
        skills=[
            {"name": "NLP", "proficiency": "expert", "endorsements": 40, "duration_months": 48},
            {"name": "Embeddings", "proficiency": "expert", "endorsements": 30, "duration_months": 36},
        ],
    )
    f1 = extract_all_features(stuffer)
    f2 = extract_all_features(genuine)
    f1.pop("_text_for_embedding"); f2.pop("_text_for_embedding")

    assert f1["title_tier"] == "other", f"expected 'other', got {f1['title_tier']}"
    assert f2["title_tier"] == "core", f"expected 'core', got {f2['title_tier']}"

    b1 = score_candidate(f1, semantic_similarity=0.1)
    b2 = score_candidate(f2, semantic_similarity=0.6)
    assert b2.final_score > b1.final_score * 3, (
        f"genuine ML engineer ({b2.final_score:.3f}) should score well above "
        f"keyword-stuffer ({b1.final_score:.3f})"
    )
    print("PASS: test_keyword_stuffer_demoted")


def test_honeypot_career_overrun_detected():
    """Career history totaling far more months than stated years_of_experience
    should be flagged as a honeypot and score-floored."""
    honeypot = make_minimal_candidate(
        profile={"current_title": "Machine Learning Engineer", "years_of_experience": 3.0},
        career_history=[
            {"company": "Acme", "title": "ML Engineer", "start_date": "2010-01-01",
             "end_date": None, "duration_months": 180, "description": "ML work."}
        ],
    )
    feats = extract_all_features(honeypot)
    feats.pop("_text_for_embedding")
    assert feats["is_honeypot"], "expected honeypot flag to be True"
    bd = score_candidate(feats, semantic_similarity=0.8)
    assert bd.final_score <= 0.02 + 1e-9, f"expected near-zero floored score, got {bd.final_score}"
    print("PASS: test_honeypot_career_overrun_detected")


def test_honeypot_zero_duration_experts_detected():
    """3+ skills marked 'expert' with 0 months used should be flagged."""
    honeypot = make_minimal_candidate(
        skills=[
            {"name": "Python", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
            {"name": "NLP", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
            {"name": "LLM", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
        ]
    )
    feats = extract_all_features(honeypot)
    feats.pop("_text_for_embedding")
    assert feats["is_honeypot"], "expected honeypot flag to be True"
    print("PASS: test_honeypot_zero_duration_experts_detected")


def test_services_only_career_penalized():
    services_candidate = make_minimal_candidate(
        profile={"current_title": "Machine Learning Engineer"},
        career_history=[
            {"company": "Tata Consultancy Services", "title": "ML Engineer",
             "start_date": "2018-01-01", "end_date": None,
             "duration_months": 72, "description": "ML work for clients."}
        ],
    )
    feats = extract_all_features(services_candidate)
    feats.pop("_text_for_embedding")
    assert feats["career_entirely_services_firms"] is True
    bd = score_candidate(feats, semantic_similarity=0.5)
    assert bd.component_scores["company_quality"] < 0.2
    print("PASS: test_services_only_career_penalized")


def test_outside_india_no_relocate_penalized():
    cand = make_minimal_candidate(
        profile={"current_title": "Machine Learning Engineer", "location": "New York", "country": "USA"},
        redrob_signals={"willing_to_relocate": False},
    )
    feats = extract_all_features(cand)
    feats.pop("_text_for_embedding")
    assert feats["location_tier"] == "outside_india"
    bd = score_candidate(feats, semantic_similarity=0.5)
    assert bd.component_scores["location"] <= 0.2
    print("PASS: test_outside_india_no_relocate_penalized")


def test_skill_cluster_equivalence_credit():
    """A candidate listing 'Retrieval Augmented Generation' (not the literal
    string 'rag') should still earn core-skill-equivalent trust via the
    skill-cluster synonym index, and an adjacent-only skill like 'langgraph'
    should earn a smaller, non-zero credit."""
    cand = make_minimal_candidate(
        profile={"current_title": "Machine Learning Engineer"},
        skills=[
            {"name": "Retrieval Augmented Generation", "proficiency": "expert",
             "endorsements": 20, "duration_months": 24},
            {"name": "LangGraph", "proficiency": "intermediate",
             "endorsements": 5, "duration_months": 12},
        ],
    )
    feats = extract_all_features(cand)
    feats.pop("_text_for_embedding")
    assert feats["cluster_skill_trust"] > 0, "expected nonzero cluster_skill_trust"
    assert any("Retrieval Augmented Generation" in h for h in feats["cluster_skill_hits"])
    print("PASS: test_skill_cluster_equivalence_credit")


def test_career_trajectory_promotion_detected():
    """A candidate whose titles escalate in seniority over time should show
    promotion_count > 0 and a positive career_trajectory component score."""
    promoted = make_minimal_candidate(
        career_history=[
            {"company": "Acme", "title": "Software Engineer", "start_date": "2018-01-01",
             "end_date": "2021-01-01", "duration_months": 36, "description": "Built backend services."},
            {"company": "Acme", "title": "Senior Software Engineer", "start_date": "2021-01-01",
             "end_date": "2023-01-01", "duration_months": 24, "description": "Led backend projects."},
            {"company": "Acme", "title": "Engineering Lead", "start_date": "2023-01-01",
             "end_date": None, "duration_months": 24, "description": "Leading a team of engineers."},
        ],
    )
    feats = extract_all_features(promoted)
    feats.pop("_text_for_embedding")
    assert feats["promotion_count"] >= 2
    assert feats["leadership_is_recent"] is True
    bd = score_candidate(feats, semantic_similarity=0.4)
    assert bd.component_scores["career_trajectory"] > 0.5
    print("PASS: test_career_trajectory_promotion_detected")


def test_recruiter_dimensions_present_and_bounded():
    """Every named recruiter-intelligence dimension should be present and
    fall in [0, 1] for an ordinary candidate."""
    cand = make_minimal_candidate()
    feats = extract_all_features(cand)
    feats.pop("_text_for_embedding")
    bd = score_candidate(feats, semantic_similarity=0.3)
    expected_dims = {
        "technical_competence", "domain_fit", "career_quality", "ownership",
        "production_readiness", "leadership", "learning_velocity",
        "behavioral_reliability", "hiring_risk",
    }
    assert expected_dims.issubset(bd.recruiter_dimensions.keys())
    for k, v in bd.recruiter_dimensions.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of bounds"
    assert bd.confidence_label in ("low", "medium", "high")
    print("PASS: test_recruiter_dimensions_present_and_bounded")


def test_sparse_profile_capped_confidence():
    """A candidate with almost no profile detail should be capped at low/
    medium confidence regardless of how the few available fields score."""
    sparse = make_minimal_candidate(
        profile={"summary": "", "headline": ""},
        career_history=[],
        skills=[],
        education=[],
        redrob_signals={"profile_completeness_score": 10.0},
    )
    feats = extract_all_features(sparse)
    feats.pop("_text_for_embedding")
    bd = score_candidate(feats, semantic_similarity=0.0)
    assert bd.confidence_label in ("low", "medium")
    print("PASS: test_sparse_profile_capped_confidence")


def test_weights_sum_to_one():
    from config import WEIGHTS
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9
    print("PASS: test_weights_sum_to_one")


def _make_breakdown(candidate_id, final_score, recruiter_dimensions, is_honeypot=False):
    from scoring import ScoreBreakdown
    return ScoreBreakdown(
        candidate_id=candidate_id,
        fit_score=final_score,
        behavioral_multiplier=1.0,
        final_score=final_score,
        recruiter_dimensions=recruiter_dimensions,
        is_honeypot=is_honeypot,
    )


def test_cohort_twin_pair_finds_deciding_dimension():
    """Two candidates with near-identical final_score but a clear gap on
    exactly one recruiter dimension should be flagged as a twin pair, with
    that dimension surfaced as the deciding factor for both."""
    from cohort import compute_cohort_notes

    a = _make_breakdown("CAND_A", 0.801, {
        "technical_competence": 0.70, "domain_fit": 0.70, "career_quality": 0.70,
        "ownership": 0.60, "production_readiness": 0.90, "leadership": 0.50,
        "learning_velocity": 0.60, "behavioral_reliability": 0.80, "hiring_risk": 0.20,
    })
    b = _make_breakdown("CAND_B", 0.800, {
        "technical_competence": 0.70, "domain_fit": 0.70, "career_quality": 0.70,
        "ownership": 0.60, "production_readiness": 0.45, "leadership": 0.50,
        "learning_velocity": 0.60, "behavioral_reliability": 0.80, "hiring_risk": 0.20,
    })
    ranked = sorted([a, b], key=lambda bd: (-bd.final_score, bd.candidate_id))
    notes = compute_cohort_notes(ranked)
    assert "CAND_A" in notes and "CAND_B" in notes
    assert notes["CAND_A"].deciding_dimension == "production_readiness"
    assert notes["CAND_A"].direction == "higher"
    assert notes["CAND_B"].direction == "lower"
    assert notes["CAND_A"].twin_id == "CAND_B"
    print("PASS: test_cohort_twin_pair_finds_deciding_dimension")


def test_cohort_no_note_when_scores_far_apart():
    """Candidates whose scores aren't actually close shouldn't get a
    cohort note even if their dimensions differ a lot -- a real score gap
    already explains the ordering without a comparative clause."""
    from cohort import compute_cohort_notes

    a = _make_breakdown("CAND_C", 0.95, {"technical_competence": 0.9, "hiring_risk": 0.1})
    b = _make_breakdown("CAND_D", 0.40, {"technical_competence": 0.2, "hiring_risk": 0.1})
    ranked = sorted([a, b], key=lambda bd: (-bd.final_score, bd.candidate_id))
    notes = compute_cohort_notes(ranked)
    assert notes == {}
    print("PASS: test_cohort_no_note_when_scores_far_apart")


def test_cohort_no_note_when_dimensions_are_also_similar():
    """A genuine near-tie where every dimension is also close shouldn't
    force a deciding-factor claim that overstates a sub-threshold gap."""
    from cohort import compute_cohort_notes

    a = _make_breakdown("CAND_E", 0.700, {"technical_competence": 0.70, "hiring_risk": 0.20})
    b = _make_breakdown("CAND_F", 0.699, {"technical_competence": 0.72, "hiring_risk": 0.21})
    ranked = sorted([a, b], key=lambda bd: (-bd.final_score, bd.candidate_id))
    notes = compute_cohort_notes(ranked)
    assert notes == {}
    print("PASS: test_cohort_no_note_when_dimensions_are_also_similar")


def test_cohort_honeypots_excluded():
    """Honeypots should never be treated as a 'twin' or receive a note --
    they're already excluded from genuine contention."""
    from cohort import compute_cohort_notes

    a = _make_breakdown("CAND_G", 0.500, {"technical_competence": 0.5, "hiring_risk": 0.5})
    trap = _make_breakdown("CAND_H", 0.499, {"technical_competence": 0.1, "hiring_risk": 0.9}, is_honeypot=True)
    ranked = sorted([a, trap], key=lambda bd: (-bd.final_score, bd.candidate_id))
    notes = compute_cohort_notes(ranked)
    assert notes == {}
    print("PASS: test_cohort_honeypots_excluded")


def test_cohort_note_renders_in_reasoning():
    """End-to-end: a cohort note passed into build_reasoning should produce
    a grounded comparative clause naming the actual deciding dimension."""
    from reasoning import build_reasoning
    from cohort import CohortNote

    cand = make_minimal_candidate(
        profile={"current_title": "Machine Learning Engineer"},
        skills=[{"name": "faiss", "endorsements": 5, "months_used": 24}],
    )
    feats = extract_all_features(cand)
    feats.pop("_text_for_embedding")
    bd = score_candidate(feats, semantic_similarity=0.6)
    note = CohortNote(
        twin_id="CAND_0000099", deciding_dimension="production_readiness",
        deciding_label="production readiness", margin=0.12, direction="higher",
    )
    text = build_reasoning(bd, rank=9, cohort_note=note)
    assert "CAND_0000099" in text
    assert "production readiness" in text
    print("PASS: test_cohort_note_renders_in_reasoning")


if __name__ == "__main__":
    test_keyword_stuffer_demoted()
    test_honeypot_career_overrun_detected()
    test_honeypot_zero_duration_experts_detected()
    test_services_only_career_penalized()
    test_outside_india_no_relocate_penalized()
    test_skill_cluster_equivalence_credit()
    test_career_trajectory_promotion_detected()
    test_recruiter_dimensions_present_and_bounded()
    test_sparse_profile_capped_confidence()
    test_weights_sum_to_one()
    test_cohort_twin_pair_finds_deciding_dimension()
    test_cohort_no_note_when_scores_far_apart()
    test_cohort_no_note_when_dimensions_are_also_similar()
    test_cohort_honeypots_excluded()
    test_cohort_note_renders_in_reasoning()
    print("\nAll tests passed.")
