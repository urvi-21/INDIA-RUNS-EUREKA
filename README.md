# 🚀 Recruiter Intelligence Engine
### AI-Powered Candidate Ranking Beyond Keyword Matching

> **Hackathon Submission – India Runs Eureka**
>
> An explainable recruiter intelligence system that ranks candidates based on demonstrated capabilities, career evidence, hiring readiness, and recruiter reasoning—not keyword matching.

---

# Problem

Traditional Applicant Tracking Systems rely heavily on keyword matching.

This creates several problems:

- Excellent candidates are missed because they use different terminology.
- Keyword stuffing is rewarded.
- Career progression and achievements are ignored.
- Recruiters still spend hours manually reviewing resumes.

Our goal was to build a system that reasons about candidates like an experienced recruiter rather than a search engine.

---

# Our Approach

Instead of matching technologies, our system infers **capabilities** from multiple evidence sources.

The pipeline evaluates:

- Career trajectory
- Production experience
- Project ownership
- Hiring readiness
- Behavioral signals
- Technical capability
- Semantic alignment with the job
- Explainable recruiter reasoning

---

# System Architecture

```text
Job Description
        │
        ▼
Hiring Profile Builder
        │
        ▼
Feature Extraction
        │
        ▼
Capability Inference
        │
        ▼
Semantic Matching
        │
        ▼
Recruiter Intelligence Scoring
        │
        ▼
LightGBM LambdaMART Re-ranking
        │
        ▼
Reasoning Generation
        │
        ▼
Top-N Ranked Candidates
```

---

# Pipeline

## 1. Hiring Profile Builder

Converts an arbitrary Job Description into a structured hiring profile.

Extracts:

- responsibilities
- seniority
- capability requirements
- experience expectations
- hiring priorities

---

## 2. Candidate Feature Extraction

Builds recruiter-oriented features from each candidate.

Examples include:

- experience
- title progression
- project quality
- assessments
- behavioral indicators
- availability
- company context

---

## 3. Capability Inference

Rather than matching keywords, capabilities are inferred from evidence such as:

- career history
- quantified achievements
- leadership progression
- production ownership
- supporting projects

Technology names only strengthen existing evidence—they never create capabilities by themselves.

---

## 4. Semantic Understanding

A concept-expanded TF-IDF + SVD representation measures semantic alignment between the hiring profile and candidate profile.

This allows recognition of related concepts rather than exact wording.

---

## 5. Recruiter Intelligence Scoring

Each candidate receives multidimensional scores across recruiter-centric dimensions including:

- capability fit
- hiring readiness
- production experience
- seniority
- career consistency
- recruiter confidence

---

## 6. Learning-to-Rank

A LightGBM LambdaMART model re-ranks candidates using recruiter-oriented features.

This optimizes the final ordering instead of relying solely on handcrafted weights.

---

## 7. Explainability

Every ranked candidate includes:

- recruiter rationale
- strengths
- concerns
- capability evidence
- recruiter dimensions
- confidence
- cohort differentiation

---

# Streamlit Dashboard

The repository includes a recruiter dashboard featuring:

- Job Description upload
- PDF / DOCX / TXT support
- Candidate ranking
- Candidate Intelligence Dossier
- Capability evidence
- Recruiter dimensions
- Pipeline execution log
- Downloadable submission CSV

---

# Repository Structure

```
repo/
│
├── src/
│   ├── pipeline.py
│   ├── hiring_profile.py
│   ├── capability_inference.py
│   ├── semantic.py
│   ├── scoring.py
│   ├── reasoning.py
│   ├── ranker.py
│   └── ...
│
├── sandbox/
│   └── streamlit_app.py
│
├── tests/
│
├── artifacts/
│
└── README.md
```

---

# Running the Pipeline

```bash
python src/pipeline.py
```

Output:

```
artifacts/team_submission.csv
artifacts/team_submission_explanations.json
```

---

# Running the Dashboard

```bash
streamlit run sandbox/streamlit_app.py
```

---

# Submission

The pipeline produces:

```
artifacts/team_submission.csv
```

which conforms to the required submission schema.

---

# Design Principles

- No LLM API calls
- CPU-only inference
- Explainable ranking
- Reproducible results
- Capability-first reasoning
- Recruiter-centric scoring
- Universal Job Description compatibility

---

# Key Innovation

Most ranking systems answer:

> "Does this resume mention the required technologies?"

Our system answers:

> "Has this candidate demonstrated the capabilities the recruiter actually needs?"

This shifts candidate evaluation from keyword matching to evidence-based recruiter intelligence.

---

# Future Improvements

- Learned capability inference models
- Knowledge graph based reasoning
- Interview recommendation engine
- Recruiter feedback learning
- Multi-role hiring profiles
- Continuous ranking adaptation

---

# Authors

Developed for the **India Runs Eureka Hackathon**.
