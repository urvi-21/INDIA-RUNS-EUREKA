
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
from features import extract_all_features
from semantic import SemanticSimilarityModel
from scoring import score_candidate
from reasoning import build_reasoning, build_dimension_summary
from cohort import compute_cohort_notes
from config import OUTPUT_TOP_N
from hiring_profile import HiringProfileBuilder

def load_candidates(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jd_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def run_pipeline(candidates_path: str, jd_path: str, output_path: str,
                  top_n: int = OUTPUT_TOP_N, use_lgbm_rerank: bool = True,
                  limit: int | None = None, write_explanations: bool = True) -> None:
    t0 = time.time()

    print("[1/7] Building HiringProfile from the job description (single source of truth) ...")
    jd_text = load_jd_text(jd_path)
    hiring_profile = HiringProfileBuilder().build(jd_text)

    print("[2/7] Loading candidates and extracting features ...")
    all_features = []
    text_blobs = []
    n = 0
    for cand in load_candidates(candidates_path):
        feats = extract_all_features(cand, hiring_profile=hiring_profile)
        all_features.append(feats)
        text_blobs.append(feats.pop("_text_for_embedding"))
        n += 1
        if limit and n >= limit:
            break
        if n % 20000 == 0:
            print(f"    ... {n} candidates processed ({time.time()-t0:.1f}s elapsed)")
    print(f"    Loaded {n} candidates in {time.time()-t0:.1f}s")

    print("[3/7] Fitting semantic similarity model (TF-IDF + SVD, concept-expanded) ...")
    t1 = time.time()
    sem_model = SemanticSimilarityModel(n_components=256)

    # hiring_profile.raw_text is the JD's free-text prose -- the one place
    # raw JD text is still needed as-is, since TF-IDF/SVD requires a text
    # corpus rather than the structured HiringProfile. HiringProfile itself
    # (not a second independent jd_text variable) is the single carrier of
    # that text from here on, so there is exactly one JD representation in
    # flight, not two.
    fit_corpus = [hiring_profile.raw_text] + text_blobs
    sem_model.fit(fit_corpus)
    similarities = sem_model.similarity_to_jd(text_blobs, hiring_profile.raw_text)
    similarities = np.clip(similarities, 0.0, 1.0)
    print(f"    Semantic model fit + scored in {time.time()-t1:.1f}s")

    print("[4/7] Scoring all candidates (rule-based fit + career trajectory + behavioral multiplier) ...")
    t2 = time.time()
    breakdowns = []
    for feats, sim in zip(all_features, similarities):
        bd = score_candidate(feats, float(sim))
        breakdowns.append(bd)
    print(f"    Scored {len(breakdowns)} candidates in {time.time()-t2:.1f}s")
    conf_counts = {"high": 0, "medium": 0, "low": 0}
    for bd in breakdowns:
        conf_counts[bd.confidence_label] = conf_counts.get(bd.confidence_label, 0) + 1
    print(f"    Confidence distribution: {conf_counts}")

    if use_lgbm_rerank:
        print("[5/7] Re-ranking with LightGBM LambdaMART ranker ...")
        t3 = time.time()
        try:
            from ranker import train_and_rerank
            breakdowns = train_and_rerank(all_features, breakdowns)
            print(f"    Re-ranked in {time.time()-t3:.1f}s")
        except Exception as e:
            print(f"    [WARN] LightGBM re-rank skipped due to: {e}")
            print(f"    Falling back to rule-based fit_score * behavioral_multiplier ranking.")
    else:
        print("[5/7] Skipping LightGBM re-rank (disabled).")

    print("[6/7] Selecting top candidates, building output ...")
    
    breakdowns_sorted = sorted(
        breakdowns, key=lambda b: (-b.final_score, b.candidate_id)
    )

    non_honeypot = [b for b in breakdowns_sorted if not b.is_honeypot]
    top = non_honeypot[:top_n]
    if len(top) < top_n:
        # Backfill from honeypots only if there are somehow not enough
        # legitimate candidates (should not happen at this dataset size).
        backfill = [b for b in breakdowns_sorted if b.is_honeypot][: top_n - len(top)]
        top += backfill
        top = sorted(top, key=lambda b: (-b.final_score, b.candidate_id))

    print("[7/7] Computing cohort differentiation (twin-pair deciding factors) ...")
    cohort_notes = compute_cohort_notes(top)
    print(f"    {len(cohort_notes)}/{len(top)} candidates have a near-tied score "
          f"neighbor with a clear deciding dimension")
    
    top = sorted(
        top,
        key=lambda b: (
            -b.final_score,
            -b.confidence_score,
            b.candidate_id,
        ),
    )
    write_csv(top, output_path, cohort_notes)
    print(f"    Wrote {len(top)} rows to {output_path}")

    if write_explanations:
        explain_path = str(Path(output_path).with_name(Path(output_path).stem + "_explanations.json"))
        write_explanations_json(top, explain_path, cohort_notes)
        print(f"    Wrote recruiter-intelligence explanations to {explain_path} (internal use, not for submission)")

    elapsed = time.time() - t0

    print("=" * 70)
    print(f"Pipeline completed in {elapsed:.1f} seconds")
    print(f"Candidates processed : {n}")
    print(f"Candidates submitted : {len(top)}")
    print("=" * 70)

def write_csv(top_breakdowns, output_path: str, cohort_notes: dict | None = None) -> None:
    cohort_notes = cohort_notes or {}
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, bd in enumerate(top_breakdowns, start=1):
            reasoning = build_reasoning(bd, rank=i, cohort_note=cohort_notes.get(bd.candidate_id))
            # Score normalized to a clean 0-1 float, rounded for readability
            # while preserving the strict ordering already established.
            writer.writerow([bd.candidate_id, i, round(bd.final_score, 6), reasoning])


def write_explanations_json(top_breakdowns, explain_path: str, cohort_notes: dict | None = None) -> None:

    cohort_notes = cohort_notes or {}
    payload = []
    for i, bd in enumerate(top_breakdowns, start=1):
        note = cohort_notes.get(bd.candidate_id)
        payload.append({
            "rank": i,
            "candidate_id": bd.candidate_id,
            "final_score": round(bd.final_score, 6),
            "fit_score": round(bd.fit_score, 6),
            "behavioral_multiplier": round(bd.behavioral_multiplier, 6),
            "component_scores": {k: round(v, 4) for k, v in bd.component_scores.items()},
            "recruiter_dimensions": bd.recruiter_dimensions,
            "confidence_score": bd.confidence_score,
            "confidence_label": bd.confidence_label,
            "reasons": bd.reasons,
            "concerns": bd.concerns,
            "dimension_summary": build_dimension_summary(bd),
            # PHASE 3: evidence-based capability reasoning -- per capability,
            # confidence + WHY (supporting evidence, positions, projects,
            # evidence count/strength) instead of a bare number.
            "capability_evidence": bd.capability_evidence,
            "cohort_note": None if note is None else {
                "twin_id": note.twin_id,
                "deciding_dimension": note.deciding_dimension,
                "margin": note.margin,
                "direction": note.direction,
            },
        })
    with open(explain_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Redrob candidate ranking pipeline")
    parser.add_argument("--candidates", default="D:\\DATA SCIENCE\\repo\\data\\candidates.jsonl")
    parser.add_argument("--jd", default="D:\\DATA SCIENCE\\repo\\job_description.txt")
    parser.add_argument("--output", default="D:\\DATA SCIENCE\\repo\\artifacts\\team_eureka.csv")
    parser.add_argument("--top-n", type=int, default=OUTPUT_TOP_N)
    parser.add_argument("--no-lgbm", action="store_true", help="Disable LightGBM re-ranking stage")
    parser.add_argument("--no-explanations", action="store_true", help="Skip writing the internal explanations JSON")
    parser.add_argument("--limit", type=int, default=None, help="Debug: only process first N candidates")
    args = parser.parse_args()

    run_pipeline(
        candidates_path=args.candidates,
        jd_path=args.jd,
        output_path=args.output,
        top_n=args.top_n,
        use_lgbm_rerank=not args.no_lgbm,
        limit=args.limit,
        write_explanations=not args.no_explanations,
    )


if __name__ == "__main__":
    main()
