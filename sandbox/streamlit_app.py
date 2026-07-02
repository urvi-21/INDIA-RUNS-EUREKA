"""
Redrob Ranker — Candidate Intelligence Dossier
================================================
A Streamlit front end for the redrob_ranker pipeline. Defaults to the
official bundled candidate database (data/candidates.jsonl) with zero
upload required, accepts a job description as an uploaded file (PDF /
DOCX / TXT / MD) or pasted text, runs the real pipeline unchanged, and
presents the result as a recruiter-facing "intelligence dossier":
leaderboard, confidence/­risk breakdown, per-candidate evidence radar,
and the fully-grounded reasoning text the pipeline produced.

Run with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — the pipeline modules use bare `from features import ...`
# style imports, so src/ needs to be on sys.path (matches how the repo's
# own CLI is invoked from inside src/).
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

LOCAL_DB = ROOT / "data" / "candidates.jsonl"
DEMO_DB = ROOT / "data" / "demo_candidates.jsonl"

if LOCAL_DB.exists():
    BUNDLED_CANDIDATES = LOCAL_DB
else:
    BUNDLED_CANDIDATES = DEMO_DB
DEFAULT_JD_PATH = ROOT / "job_description.txt"

DIM_LABELS = {
    "technical_competence": "Technical Competence",
    "domain_fit": "Domain Fit",
    "career_quality": "Career Quality",
    "ownership": "Ownership",
    "production_readiness": "Production Readiness",
    "leadership": "Leadership",
    "learning_velocity": "Learning Velocity",
    "behavioral_reliability": "Behavioral Reliability",
    "hiring_risk": "Hiring Risk",
}
COMPONENT_LABELS = {
    "title_seniority": "Title / Seniority",
    "career_narrative": "Career Narrative Fit",
    "skills_trust": "Skills Trust",
    "recruiter_capabilities": "Recruiter Capabilities",
    "career_trajectory": "Career Trajectory",
    "experience_years": "Experience Years",
    "company_quality": "Company Quality",
    "location": "Location",
    "education": "Education",
}

CONF_COLOR = {"high": "#2FD9A8", "medium": "#E8A33D", "low": "#E8574A"}
CONF_DOT = {"high": "🟢", "medium": "🟡", "low": "🔴"}

# ---------------------------------------------------------------------------
# Page config + theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Redrob Ranker — Candidate Dossier",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#0A0D12; --panel:#12161F; --card:#161B26; --card2:#1B2130;
  --border:#242B3A; --text:#E9EDF4; --muted:#8891A3;
  --amber:#E8A33D; --teal:#2FD9A8; --red:#E8574A; --blue:#5B8DEF;
}

html, body, [class*="css"]  { font-family:'Inter', sans-serif; }
.stApp {
  background:
    radial-gradient(circle at 18% -10%, #182034 0%, rgba(24,32,52,0) 45%),
    radial-gradient(circle at 100% 0%, #14202a 0%, rgba(20,32,42,0) 40%),
    var(--bg);
  color: var(--text);
}
section[data-testid="stSidebar"] {
  background: #0C0F16; border-right: 1px solid var(--border);
}
#MainMenu, footer, header[data-testid="stHeader"] { background: transparent; }

h1,h2,h3,h4 { font-family:'Sora', sans-serif; letter-spacing:-0.01em; }

.eyebrow{
  font-family:'IBM Plex Mono', monospace; font-size:0.72rem; letter-spacing:0.18em;
  text-transform:uppercase; color: var(--amber); margin-bottom:6px;
}
.hero-title{ font-size:2.3rem; font-weight:800; color:var(--text); margin:0 0 6px 0; line-height:1.1;}
.hero-sub{ color:var(--muted); font-size:0.98rem; max-width:820px; line-height:1.55;}

.status-strip{
  display:flex; gap:22px; flex-wrap:wrap; margin-top:16px; padding-top:14px;
  border-top:1px dashed var(--border); font-family:'IBM Plex Mono',monospace;
  font-size:0.78rem; color:var(--muted);
}
.status-strip b{ color: var(--teal); }

/* KPI cards */
.kpi-row{ display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin: 18px 0 8px 0;}
@media (max-width: 1100px){ .kpi-row{ grid-template-columns:repeat(2,1fr);} }
.kpi{
  background:linear-gradient(180deg,var(--card) 0%, var(--panel) 100%);
  border:1px solid var(--border); border-left:3px solid var(--amber);
  border-radius:10px; padding:14px 16px;
}
.kpi .label{ font-family:'IBM Plex Mono',monospace; font-size:0.68rem; letter-spacing:0.1em;
  text-transform:uppercase; color:var(--muted); margin-bottom:6px;}
.kpi .value{ font-family:'Sora',sans-serif; font-size:1.5rem; font-weight:700; color:var(--text);}
.kpi .sub{ font-size:0.72rem; color:var(--muted); margin-top:2px;}

/* section headers */
.sec-head{ display:flex; align-items:baseline; gap:10px; margin: 26px 0 10px 0;}
.sec-head .num{ font-family:'IBM Plex Mono',monospace; color:var(--amber); font-size:0.85rem;}
.sec-head h3{ margin:0; font-size:1.05rem; color:var(--text);}
.sec-head .line{ flex:1; height:1px; background:var(--border);}

/* dossier detail card */
.dossier-card{
  background:var(--card); border:1px solid var(--border); border-radius:14px;
  padding:20px 22px; margin-top:6px;
}
.id-row{ display:flex; align-items:center; gap:12px; flex-wrap:wrap;}
.rank-badge{
  font-family:'IBM Plex Mono',monospace; font-weight:600; font-size:0.85rem;
  color:var(--bg); background:var(--amber); border-radius:50px; padding:5px 12px;
}
.cid-chip{
  font-family:'IBM Plex Mono',monospace; font-size:0.95rem; color:var(--text);
  background:var(--card2); border:1px solid var(--border); border-radius:6px; padding:5px 10px;
}
.redact{
  font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#5b6577;
  background: repeating-linear-gradient(45deg,#000 0,#000 3px,#0e0e0e 3px,#0e0e0e 6px);
  border-radius:4px; padding:5px 10px; letter-spacing:0.08em;
}
.stamp{
  font-family:'IBM Plex Mono',monospace; font-weight:700; font-size:0.72rem;
  letter-spacing:0.12em; text-transform:uppercase; padding:5px 12px; border-radius:5px;
  transform: rotate(-2deg); display:inline-block;
}
.reasoning-block{
  margin-top:16px; padding:14px 16px; background:var(--card2);
  border-left:3px solid var(--blue); border-radius:8px; font-size:0.93rem;
  line-height:1.6; color:#D8DEEA; font-style:italic;
}
.tag-list{ margin-top:10px; }
.tag-good{ color:#B7F5DF; }
.tag-bad{ color:#F7C9C2; }
.cohort-box{
  margin-top:14px; padding:10px 14px; border:1px dashed var(--amber);
  border-radius:8px; font-size:0.82rem; color:#F0D9AE; background:rgba(232,163,61,0.06);
}

.badge-caption{ font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:var(--muted); }

/* bundled DB status card */
.db-card{
  border:1px solid var(--border); border-left:3px solid var(--teal);
  background:var(--card2); border-radius:8px; padding:10px 12px; margin-bottom:10px;
}
.db-card .db-title{ color:var(--teal); font-weight:600; font-size:0.85rem; }
.db-card .db-meta{ font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:var(--muted); margin-top:4px; }
.db-missing{
  border:1px solid var(--red); border-left:3px solid var(--red);
  background:rgba(232,87,74,0.08); border-radius:8px; padding:10px 12px;
  font-size:0.8rem; color:#F7C9C2; margin-bottom:10px;
}

div[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:10px; overflow:hidden;}
.stButton>button, .stDownloadButton>button{
  background:var(--card2); color:var(--text); border:1px solid var(--border);
  border-radius:8px; font-family:'IBM Plex Mono',monospace; font-size:0.82rem;
}
.stButton>button:hover, .stDownloadButton>button:hover{ border-color:var(--amber); color:var(--amber);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small generic helpers
# ---------------------------------------------------------------------------
def count_jsonl_lines(path: Path) -> int:
    """Count candidate records without loading the file into memory."""
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def count_lines_bytes(data: bytes) -> int:
    if not data:
        return 0
    n = data.count(b"\n")
    return n if data.endswith(b"\n") else n + 1


@st.cache_data(show_spinner=False)
def cached_line_count(path_str: str, size: int, mtime: float) -> int:
    # size/mtime are only there to bust the cache when the file on disk changes.
    return count_jsonl_lines(Path(path_str))


def bundled_db_info() -> dict:
    if BUNDLED_CANDIDATES.exists() and BUNDLED_CANDIDATES.stat().st_size > 0:
        stat = BUNDLED_CANDIDATES.stat()
        count = cached_line_count(str(BUNDLED_CANDIDATES), stat.st_size, stat.st_mtime)
        return {"exists": True, "count": count, "path": str(BUNDLED_CANDIDATES), "size_mb": stat.st_size / 1e6}
    return {"exists": False}


def parse_jd_file(uploaded_file) -> str:
    """Extract plain text from an uploaded PDF / DOCX / TXT / MD job description."""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    elif name.endswith(".docx"):
        import docx

        document = docx.Document(io.BytesIO(data))
        text = "\n".join(p.text for p in document.paragraphs)
    elif name.endswith(".txt") or name.endswith(".md"):
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported job description file type: {uploaded_file.name}")

    return text.strip()


def explanations_to_df(explanations: list[dict], csv_reason_by_id: dict | None = None) -> pd.DataFrame:
    csv_reason_by_id = csv_reason_by_id or {}
    rows = []
    for e in explanations:
        rd = e.get("recruiter_dimensions", {}) or {}
        reasons = e.get("reasons") or []
        rows.append(
            {
                "rank": e.get("rank"),
                "candidate_id": e.get("candidate_id"),
                "final_score": e.get("final_score", 0.0),
                "fit_score": e.get("fit_score", 0.0),
                "confidence": e.get("confidence_label", "n/a"),
                "confidence_score": e.get("confidence_score", 0.0),
                "technical_competence": rd.get("technical_competence"),
                "domain_fit": rd.get("domain_fit"),
                "production_readiness": rd.get("production_readiness"),
                "hiring_risk": rd.get("hiring_risk"),
                "top_reason": reasons[0] if reasons else "—",
                "concerns": len(e.get("concerns") or []),
                "cohort_flag": bool(e.get("cohort_note")),
                "reasoning": csv_reason_by_id.get(e.get("candidate_id"), ""),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline execution (single code path for both bundled DB and uploads)
# ---------------------------------------------------------------------------
def _run_pipeline_to_paths(candidates_path: str, jd_path: str, top_n: int, use_lgbm: bool, limit: int | None):
    from pipeline import run_pipeline  # local import: needs SRC on sys.path

    out_path = ROOT / f".tmp_out_{uuid.uuid4().hex[:8]}.csv"
    log_buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(log_buf):
        run_pipeline(
            candidates_path=candidates_path,
            jd_path=jd_path,
            output_path=str(out_path),
            top_n=top_n,
            use_lgbm_rerank=use_lgbm,
            limit=limit,
            write_explanations=True,
        )
    elapsed = time.time() - t0
    explain_path = out_path.with_name(out_path.stem + "_explanations.json")
    csv_df = pd.read_csv(out_path)
    explanations = json.loads(explain_path.read_text(encoding="utf-8"))
    out_path.unlink(missing_ok=True)
    explain_path.unlink(missing_ok=True)
    return csv_df, explanations, log_buf.getvalue(), elapsed


@st.cache_data(show_spinner=False)
def run_pipeline_cached(
    cache_key: str,
    candidates_path: str | None,
    jd_text: str,
    top_n: int,
    use_lgbm: bool,
    limit: int | None,
    _upload_bytes: bytes | None,
):
    """Runs the real pipeline unchanged, against either a path already on disk
    (bundled DB) or freshly-uploaded bytes (written to a temp .jsonl first).
    `_upload_bytes` is prefixed with `_` so Streamlit's cache does not hash a
    potentially large payload — `cache_key` alone determines cache identity.
    """
    tmp_cand = None
    jd_path = ROOT / f".tmp_jd_{uuid.uuid4().hex[:8]}.txt"
    try:
        if _upload_bytes is not None:
            tmp_cand = ROOT / f".tmp_up_{uuid.uuid4().hex[:8]}.jsonl"
            tmp_cand.write_bytes(_upload_bytes)
            cpath = str(tmp_cand)
        else:
            cpath = candidates_path

        jd_path.write_text(jd_text, encoding="utf-8")
        return _run_pipeline_to_paths(cpath, str(jd_path), top_n, use_lgbm, limit)
    finally:
        jd_path.unlink(missing_ok=True)
        if tmp_cand is not None:
            tmp_cand.unlink(missing_ok=True)


def build_dossier(csv_df: pd.DataFrame, explanations: list[dict], source_label: str, n_processed: int, top_n: int, elapsed: float, log: str):
    csv_reason_by_id = dict(zip(csv_df["candidate_id"], csv_df["reasoning"]))
    df = explanations_to_df(explanations, csv_reason_by_id)
    exp_by_id = {e["candidate_id"]: e for e in explanations}
    meta = {
        "source": source_label,
        "n_processed": n_processed,
        "n_shortlisted": len(df),
        "elapsed": elapsed,
        "top_n": top_n,
        "log": log,
    }
    return df, exp_by_id, meta


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def plotly_theme(fig, height=320):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E9EDF4", size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def fit_score_histogram(df: pd.DataFrame):
    fig = go.Figure(
        go.Histogram(
            x=df["fit_score"],
            marker=dict(color="#E8A33D", line=dict(color="#0A0D12", width=1)),
            nbinsx=min(20, max(5, df["fit_score"].nunique())),
        )
    )
    fig.update_layout(title="Fit score distribution — shortlist", bargap=0.08)
    fig.update_xaxes(title="fit_score", gridcolor="#242B3A")
    fig.update_yaxes(title="candidates", gridcolor="#242B3A")
    return plotly_theme(fig)


def confidence_donut(df: pd.DataFrame):
    counts = df["confidence"].value_counts()
    order = [c for c in ["high", "medium", "low"] if c in counts.index]
    fig = go.Figure(
        go.Pie(
            labels=order,
            values=[counts[c] for c in order],
            hole=0.62,
            marker=dict(colors=[CONF_COLOR.get(c, "#8891A3") for c in order]),
            textinfo="label+percent",
            textfont=dict(family="IBM Plex Mono, monospace", size=12),
        )
    )
    fig.update_layout(title="Confidence breakdown", showlegend=False)
    return plotly_theme(fig, height=320)


def dimension_bar(df: pd.DataFrame):
    dims = [d for d in DIM_LABELS if d in df.columns]
    means = [df[d].mean() for d in dims]
    labels = [DIM_LABELS[d] for d in dims]
    colors = ["#E8574A" if d == "hiring_risk" else "#2FD9A8" for d in dims]
    fig = go.Figure(go.Bar(x=means, y=labels, orientation="h", marker=dict(color=colors)))
    fig.update_layout(title="Average recruiter dimensions — shortlist")
    fig.update_xaxes(range=[0, 1], gridcolor="#242B3A")
    fig.update_yaxes(gridcolor="#242B3A", autorange="reversed")
    return plotly_theme(fig, height=340)


def radar_chart(dims: dict):
    cats = [DIM_LABELS.get(k, k) for k in dims.keys()]
    vals = list(dims.values())
    cats_closed = cats + [cats[0]]
    vals_closed = vals + [vals[0]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=vals_closed, theta=cats_closed, fill="toself",
            line=dict(color="#E8A33D", width=2), fillcolor="rgba(232,163,61,0.22)",
            name="dimensions",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#242B3A", tickfont=dict(size=8)),
            angularaxis=dict(gridcolor="#242B3A", tickfont=dict(size=10, family="IBM Plex Mono, monospace")),
        ),
        showlegend=False,
    )
    return plotly_theme(fig, height=380)


def component_bar(components: dict):
    keys = [k for k in COMPONENT_LABELS if k in components]
    labels = [COMPONENT_LABELS[k] for k in keys]
    vals = [components[k] for k in keys]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker=dict(color="#5B8DEF")))
    fig.update_xaxes(range=[0, 1], gridcolor="#242B3A")
    fig.update_yaxes(gridcolor="#242B3A", autorange="reversed")
    fig.update_layout(title="Component scores")
    return plotly_theme(fig, height=340)


# ---------------------------------------------------------------------------
# Sidebar — data source controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div class='eyebrow'>REDROB // OPS CONSOLE</div>"
        "<div style='font-family:Sora,sans-serif;font-weight:700;font-size:1.15rem;'>Candidate Database</div>",
        unsafe_allow_html=True,
    )

    db_info = bundled_db_info()

    if db_info["exists"]:
        source_mode = st.radio(
            "Candidate database",
            ["Use bundled database (default)", "Upload another dataset (.jsonl)"],
            label_visibility="collapsed",
        )
    else:
        st.markdown(
            "<div class='db-missing'>⚠ Official Redrob Candidate Database not found at "
            f"<code>data/candidates.jsonl</code> — switched to upload mode.</div>",
            unsafe_allow_html=True,
        )
        source_mode = "Upload another dataset (.jsonl)"

    upload_bytes: bytes | None = None
    candidates_path: str | None = None
    total_candidates = 0
    cand_file = None

    if source_mode == "Use bundled database (default)":
        st.markdown(
            f"""<div class="db-card">
                  <div class="db-title">✓ Official Redrob Candidate Database</div>
                  <div class="db-meta">{db_info['count']:,} candidates · {db_info['size_mb']:.1f} MB</div>
                  <div class="db-meta">{db_info['path']}</div>
                </div>""",
            unsafe_allow_html=True,
        )
        candidates_path = db_info["path"]
        total_candidates = db_info["count"]
    else:
        cand_file = st.file_uploader("Upload candidates.jsonl", type=["jsonl"])
        if cand_file is not None:
            upload_bytes = cand_file.getvalue()
            total_candidates = count_lines_bytes(upload_bytes)
            st.caption(f"{total_candidates:,} candidates detected in upload.")

    st.divider()
    st.markdown(
        "<div style='font-family:Sora,sans-serif;font-weight:700;font-size:1.05rem;'>Job Description</div>",
        unsafe_allow_html=True,
    )

    jd_mode = st.radio("Job description source", ["Upload File", "Paste Text"], label_visibility="collapsed")

    jd_text = ""
    if jd_mode == "Upload File":
        jd_file = st.file_uploader("Upload JD (PDF, DOCX, TXT, or MD)", type=["pdf", "docx", "txt", "md"])
        if jd_file is not None:
            try:
                jd_text = parse_jd_file(jd_file)
            except Exception as exc:
                st.error(f"Could not parse job description: {exc}")
    else:
        default_paste = DEFAULT_JD_PATH.read_text(encoding="utf-8") if DEFAULT_JD_PATH.exists() else ""
        jd_text = st.text_area("Paste job description text", value=default_paste, height=200)

    if jd_text:
        with st.expander("Preview extracted JD text"):
            preview = jd_text if len(jd_text) <= 5000 else jd_text[:5000] + " …"
            st.text(preview)

    st.divider()
    st.markdown(
        "<div style='font-family:Sora,sans-serif;font-weight:700;font-size:1.05rem;'>Run Settings</div>",
        unsafe_allow_html=True,
    )
    top_n = st.slider("Shortlist size (top_n)", 10, 200, 100)
    use_lgbm = st.checkbox("Use LightGBM re-rank", value=True)
    quick_demo = st.checkbox("Quick demo — cap at first N candidates", value=True)
    limit = st.number_input("Cap (N)", min_value=50, max_value=200000, value=3000, step=50, disabled=not quick_demo)

    can_run = jd_text.strip() != "" and (candidates_path is not None or upload_bytes is not None)
    if st.button("Run pipeline", width="stretch", disabled=not can_run):
        limit_val = int(limit) if quick_demo else None
        source_label = (
            f"Official Redrob Candidate Database ({total_candidates:,} candidates)"
            if candidates_path is not None
            else f"Uploaded dataset — {cand_file.name} ({total_candidates:,} candidates)"
        )
        cache_key = (
            f"{candidates_path or (cand_file.name if cand_file else '')}"
            f"-{total_candidates}-{top_n}-{use_lgbm}-{limit_val}-{hash(jd_text)}"
        )
        with st.spinner("Running feature extraction → semantic fit → scoring → re-rank..."):
            csv_df, explanations, log, elapsed = run_pipeline_cached(
                cache_key, candidates_path, jd_text, top_n, use_lgbm, limit_val, upload_bytes
            )
        n_processed = min(total_candidates, limit_val) if limit_val else total_candidates
        df, exp_by_id, meta = build_dossier(csv_df, explanations, source_label, n_processed, top_n, elapsed, log)
        st.session_state["dossier"] = {"df": df, "exp": exp_by_id, "meta": meta}
    elif not can_run:
        st.caption("Provide a candidate database and a job description to enable the run.")

    if "dossier" in st.session_state:
        st.divider()
        log_text = st.session_state["dossier"]["meta"].get("log")
        if log_text:
            with st.expander("Mission log"):
                st.code(log_text, language=None)
        st.caption("Session dossier loaded. Adjust settings and re-run to replace it.")


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="eyebrow">REDROB AI &nbsp;·&nbsp; CANDIDATE INTELLIGENCE DOSSIER</div>
    <div class="hero-title">Redrob Ranker — Shortlist Dossier</div>
    <div class="hero-sub">
      Every row below is a fully-grounded, non-hallucinated read on a candidate: a
      recruiter-facing score decomposed into nine named dimensions, a confidence
      rating derived from evidence depth (not the score itself), and reasoning text
      traceable to the candidate's own record — no field asserted that isn't backed
      by data.
    </div>
    <div class="status-strip">
      <span>ENGINE: <b>TF-IDF + SVD</b> semantic fit · <b>LightGBM</b> LambdaMART re-rank</span>
      <span>COMPUTE: <b>CPU-only</b>, offline, no network at rank time</span>
      <span>BENCHMARK: <b>~81s</b> / 100,000 candidates (README, 4-core CPU)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if "dossier" not in st.session_state:
    st.markdown(
        """
        <div class="dossier-card" style="margin-top:22px;">
          <div class="eyebrow">STANDBY</div>
          <h3 style="margin-top:0;">No dossier loaded yet</h3>
          <p style="color:var(--muted); font-size:0.92rem; line-height:1.6;">
            The <b>Ops Console</b> in the sidebar defaults to the official bundled
            candidate database when present. Add a job description (upload or paste),
            then click <b>Run pipeline</b> to score the pool and open the dossier.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

df = st.session_state["dossier"]["df"]
exp_by_id = st.session_state["dossier"]["exp"]
meta = st.session_state["dossier"]["meta"]

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
n_high = int((df["confidence"] == "high").sum())
n_flagged = int((df["concerns"] > 0).sum())
avg_fit = df["fit_score"].mean() if len(df) else 0.0
elapsed_str = f"{meta['elapsed']:.1f}s" if meta.get("elapsed") else "—"
processed_str = f"{meta['n_processed']:,}" if meta.get("n_processed") else "—"

st.markdown(
    f"""
    <div class="kpi-row">
      <div class="kpi"><div class="label">Candidates scanned</div><div class="value">{processed_str}</div><div class="sub">source pool size</div></div>
      <div class="kpi"><div class="label">Shortlisted</div><div class="value">{meta['n_shortlisted']}</div><div class="sub">top_n = {meta.get('top_n','—')}</div></div>
      <div class="kpi"><div class="label">Avg fit score</div><div class="value">{avg_fit:.3f}</div><div class="sub">rule-based fit, pre-behavioral</div></div>
      <div class="kpi"><div class="label">High confidence</div><div class="value">{n_high}/{len(df)}</div><div class="sub">{n_flagged} with flagged concerns</div></div>
      <div class="kpi"><div class="label">Runtime</div><div class="value">{elapsed_str}</div><div class="sub">{meta['source']}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="sec-head"><span class="num">01</span><h3>Shortlist signal overview</h3><div class="line"></div></div>',
    unsafe_allow_html=True,
)
c1, c2, c3 = st.columns([1.1, 1, 1.2])
with c1:
    st.plotly_chart(fit_score_histogram(df), width="stretch", config={"displayModeBar": False})
with c2:
    st.plotly_chart(confidence_donut(df), width="stretch", config={"displayModeBar": False})
with c3:
    st.plotly_chart(dimension_bar(df), width="stretch", config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="sec-head"><span class="num">02</span><h3>Leaderboard</h3><div class="line"></div></div>',
    unsafe_allow_html=True,
)

fc1, fc2, fc3 = st.columns([1.4, 1, 1])
with fc1:
    search = st.text_input("Search candidate ID", placeholder="CAND_0000030")
with fc2:
    conf_filter = st.multiselect("Confidence", ["high", "medium", "low"], default=["high", "medium", "low"])
with fc3:
    min_fit = st.slider("Min fit score", 0.0, 1.0, 0.0, 0.01)

view = df[df["confidence"].isin(conf_filter) & (df["fit_score"] >= min_fit)]
if search:
    view = view[view["candidate_id"].str.contains(search, case=False, na=False)]
view = view.sort_values("rank")

st.dataframe(
    view[["rank", "candidate_id", "final_score", "fit_score", "confidence", "concerns", "top_reason"]],
    width="stretch",
    hide_index=True,
    height=360,
    column_config={
        "rank": st.column_config.NumberColumn("Rank", width="small"),
        "candidate_id": st.column_config.TextColumn("Candidate ID"),
        "final_score": st.column_config.ProgressColumn("Final score", min_value=0, max_value=max(1.0, float(df["final_score"].max() or 1.0)), format="%.3f"),
        "fit_score": st.column_config.ProgressColumn("Fit score", min_value=0, max_value=1, format="%.3f"),
        "confidence": st.column_config.TextColumn("Confidence"),
        "concerns": st.column_config.NumberColumn("Concerns", width="small"),
        "top_reason": st.column_config.TextColumn("Leading evidence", width="large"),
    },
)

csv_download = view.drop(columns=["reasoning"]).to_csv(index=False).encode("utf-8")
st.download_button("Download filtered shortlist (CSV)", csv_download, file_name="shortlist_view.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Candidate dossier detail
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="sec-head"><span class="num">03</span><h3>Candidate dossier</h3><div class="line"></div></div>',
    unsafe_allow_html=True,
)

if view.empty:
    st.info("No candidates match the current filters.")
else:
    options = [f"#{r.rank:>3}  {r.candidate_id}  {CONF_DOT.get(r.confidence,'⚪')} {r.confidence}" for r in view.itertuples()]
    id_lookup = dict(zip(options, view["candidate_id"]))
    choice = st.selectbox("Open dossier for", options)
    cid = id_lookup[choice]
    e = exp_by_id.get(cid, {})

    conf = e.get("confidence_label", "n/a")
    conf_color = CONF_COLOR.get(conf, "#8891A3")

    st.markdown(
        f"""
        <div class="dossier-card">
          <div class="id-row">
            <span class="rank-badge">RANK №{e.get('rank','—'):03}</span>
            <span class="cid-chip">{cid}</span>
            <span class="redact">IDENTITY REDACTED</span>
            <span class="stamp" style="color:{conf_color}; border:1.5px solid {conf_color};">{conf} confidence</span>
          </div>
        """,
        unsafe_allow_html=True,
    )

    dc1, dc2 = st.columns([1, 1])
    with dc1:
        dims = e.get("recruiter_dimensions") or {}
        if dims:
            st.plotly_chart(radar_chart(dims), width="stretch", config={"displayModeBar": False})
    with dc2:
        comps = e.get("component_scores") or {}
        if comps:
            st.plotly_chart(component_bar(comps), width="stretch", config={"displayModeBar": False})

    reasoning_text = view.loc[view["candidate_id"] == cid, "reasoning"].values
    if len(reasoning_text) and reasoning_text[0]:
        st.markdown(f'<div class="reasoning-block">"{reasoning_text[0]}"</div>', unsafe_allow_html=True)

    note = e.get("cohort_note")
    if note:
        st.markdown(
            f"""<div class="cohort-box">🧬 <b>Cohort differentiation:</b> near-tied against
            <code>{note.get('twin_id')}</code> — decided by
            <b>{DIM_LABELS.get(note.get('deciding_dimension'), note.get('deciding_dimension'))}</b>
            (margin {note.get('margin'):.3f}, {note.get('direction')})</div>""",
            unsafe_allow_html=True,
        )

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("**Supporting evidence**")
        reasons = e.get("reasons") or []
        if reasons:
            st.markdown(
                "<div class='tag-list'>" + "".join(f"<div class='tag-good'>✓ {r}</div>" for r in reasons) + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("No structured reasons recorded.")
    with rc2:
        st.markdown("**Flagged concerns**")
        concerns = e.get("concerns") or []
        if concerns:
            st.markdown(
                "<div class='tag-list'>" + "".join(f"<div class='tag-bad'>⚠ {c}</div>" for c in concerns) + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("None on record.")

    cap_evidence = e.get("capability_evidence") or {}
    if cap_evidence:
        st.markdown("**Capability evidence**")
        for cap_name, cap in cap_evidence.items():
            conf_val = cap.get("confidence", 0)
            with st.expander(f"{cap_name.replace('_',' ').title()} — confidence {conf_val:.2f} ({cap.get('evidence_strength','n/a')})"):
                for ev in cap.get("supporting_evidence", []):
                    st.markdown(f"- {ev}")
                if cap.get("supporting_projects"):
                    st.caption("Projects: " + "; ".join(cap["supporting_projects"]))

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="margin-top:34px; padding-top:14px; border-top:1px solid var(--border);
                font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:var(--muted);">
      Redrob Ranker · CPU-only, offline, validator-compliant top-N shortlist · dossier view for internal review only
    </div>
    """,
    unsafe_allow_html=True,
)