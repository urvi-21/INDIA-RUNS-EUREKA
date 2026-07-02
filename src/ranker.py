

from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from scoring import ScoreBreakdown


FEATURE_COLUMNS = [
    "title_tier_core", "title_tier_adjacent",
    "is_stale_ic",
    "core_skill_trust", "support_skill_trust", "off_domain_skill_trust",
    "cluster_skill_trust",
    "n_skills_total",
    "years_of_experience", "total_career_months",
    "earlier_has_ml_signal",
    "career_entirely_services_firms",
    "location_tier_preferred", "location_tier_welcome", "location_tier_india_other",
    "best_tier_score",
    "days_inactive", "open_to_work_flag", "recruiter_response_rate",
    "interview_completion_rate", "offer_acceptance_rate", "notice_period_days",
    "willing_to_relocate", "github_activity_score", "profile_completeness_score",
    # Career trajectory (JD weakness #4: career history as a sequence, not
    # independent entries).
    "promotion_velocity", "n_company_transitions", "avg_tenure_months",
    "leadership_is_recent", "domain_consistency",
    # Explicit interaction features: trees can learn interactions implicitly,
    # but giving the GBM a couple of hand-built ones directly (JD weakness
    # #7: "feature interactions") lets it find a good split immediately
    # rather than needing many boosting rounds to approximate the product.
    "core_skill_trust_x_title_core", "skill_trust_x_trajectory", "availability_score",
    "recruiter_confidence",
    "reliability_score",
    "market_demand_score",
    "platform_engagement",
    "hiring_readiness",
]

_EDU_TIER_SCORE = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0}


def _featurize_for_gbm(feats: dict) -> dict:
    title_core = int(feats["title_tier"] == "core")
    core_skill = feats["core_skill_trust"]
    cluster_skill = feats.get("cluster_skill_trust", 0.0)
    promo_velocity = feats.get("promotion_velocity", 0.0)
    availability_score = feats.get("availability_score", 0.5)
    recruiter_confidence = feats.get("recruiter_confidence", 0.5)
    reliability_score = feats.get("reliability_score", 0.5)
    market_demand_score = feats.get("market_demand_score", 0.5)
    platform_engagement = feats.get("platform_engagement", 0.5)
    hiring_readiness = feats.get("hiring_readiness", 0.5)

    return {
        "title_tier_core": title_core,
        "title_tier_adjacent": int(feats["title_tier"] == "adjacent"),
        "is_stale_ic": int(feats["is_stale_ic"]),
        "core_skill_trust": core_skill,
        "support_skill_trust": feats["support_skill_trust"],
        "off_domain_skill_trust": feats["off_domain_skill_trust"],
        "cluster_skill_trust": cluster_skill,
        "n_skills_total": feats["n_skills_total"],
        "years_of_experience": feats["years_of_experience"],
        "total_career_months": feats["total_career_months"],
        "earlier_has_ml_signal": int(feats["earlier_has_ml_signal"]),
        "career_entirely_services_firms": int(feats["career_entirely_services_firms"]),
        "location_tier_preferred": int(feats["location_tier"] == "preferred"),
        "location_tier_welcome": int(feats["location_tier"] == "welcome"),
        "location_tier_india_other": int(feats["location_tier"] == "india_other"),
        "best_tier_score": _EDU_TIER_SCORE.get(feats["best_tier"], 0),
        "days_inactive": min(feats["days_inactive"], 2000),
        "open_to_work_flag": int(feats["open_to_work_flag"]),
        "recruiter_response_rate": feats["recruiter_response_rate"],
        "interview_completion_rate": feats["interview_completion_rate"],
        "offer_acceptance_rate": max(feats["offer_acceptance_rate"], -1),
        "notice_period_days": feats["notice_period_days"],
        "willing_to_relocate": int(feats["willing_to_relocate"]),
        "github_activity_score": feats["github_activity_score"],
        "profile_completeness_score": feats["profile_completeness_score"],
        "promotion_velocity": promo_velocity,
        "n_company_transitions": feats.get("n_company_transitions", 0),
        "avg_tenure_months": feats.get("avg_tenure_months", 0.0),
        "leadership_is_recent": int(feats.get("leadership_is_recent", False)),
        "domain_consistency": feats.get("domain_consistency", 0.0),
        "core_skill_trust_x_title_core": core_skill * title_core,
        "skill_trust_x_trajectory": (core_skill + cluster_skill) * promo_velocity,
        "availability_score": availability_score,
        "recruiter_confidence": recruiter_confidence,
        "reliability_score": reliability_score,
        "market_demand_score": market_demand_score,
        "platform_engagement": platform_engagement,
        "hiring_readiness": hiring_readiness,
    }


def _relevance_grade(scores: np.ndarray, n_grades: int = 5) -> np.ndarray:
    """Bucket scores into 0..n_grades-1 by quantile, so honeypots / score-0
    candidates land in grade 0 and the genuine top slice lands in the top
    grade. Ties in quantile edges are handled by rank-based binning."""
    ranks = pd.Series(scores).rank(method="first", ascending=True).values
    n = len(scores)
    grades = np.floor((ranks - 1) / n * n_grades).astype(int)
    return np.clip(grades, 0, n_grades - 1)


def train_and_rerank(all_features: list[dict], breakdowns: list[ScoreBreakdown],
                      n_grades: int = 5) -> list[ScoreBreakdown]:
    id_to_feats = {f["candidate_id"]: f for f in all_features}
    id_to_breakdown = {b.candidate_id: b for b in breakdowns}

    rows = []
    rule_scores = []
    cids = []
    for cid, feats in id_to_feats.items():
        rows.append(_featurize_for_gbm(feats))
        rule_scores.append(id_to_breakdown[cid].final_score)
        cids.append(cid)

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    rule_scores = np.array(rule_scores)
    y = _relevance_grade(rule_scores, n_grades=n_grades)

   
    rng = np.random.default_rng(42)
    shuffle_idx = rng.permutation(len(X))
    X = X.iloc[shuffle_idx].reset_index(drop=True)
    y = y[shuffle_idx]
    cids = [cids[i] for i in shuffle_idx]
    rule_scores = rule_scores[shuffle_idx]

    chunk_size = 8000
    n = len(X)
    group = [chunk_size] * (n // chunk_size)
    remainder = n % chunk_size
    if remainder:
        group.append(remainder)
    assert sum(group) == n

    train_data = lgb.Dataset(X, label=y, group=group)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 50],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "verbose": -1,
        "force_row_wise": True,
        "feature_fraction":0.85,
        "bagging_fraction":0.90,
        "bagging_freq":1,
    }
    model = lgb.train(params, train_data, num_boost_round=250)

    gbm_scores = model.predict(X)
    gbm_scores = np.nan_to_num(
    gbm_scores,
    nan=0.0,
    posinf=1.0,
    neginf=0.0,
    )
    
    gbm_norm = (gbm_scores - gbm_scores.min()) / (gbm_scores.max() - gbm_scores.min() + 1e-9)
    rule_norm = (rule_scores - rule_scores.min()) / (rule_scores.max() - rule_scores.min() + 1e-9)
    blended = 0.55 * rule_norm + 0.45 * gbm_norm

    for cid, b_score in zip(cids, blended):
        bd = id_to_breakdown[cid]
        # Preserve honeypot floor regardless of GBM opinion.
        if bd.is_honeypot:
            continue
        bd.final_score = float(b_score)

    feature_importance = dict(zip(FEATURE_COLUMNS, model.feature_importance(importance_type="gain")))
    top_feats = sorted(feature_importance.items(), key=lambda x: -x[1])[:8]
    print("    [ranker] Top GBM feature importances (gain):")
    for name, imp in top_feats:
        print(f"        {name}: {imp:.1f}")

    return list(id_to_breakdown.values())
