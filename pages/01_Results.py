# pages/01_Results.py — TeamReadi Results (ranked tiles + PDF report)

import os, io, re, json, html, textwrap
import datetime as dt
from typing import List, Dict, Any

import requests
import streamlit as st
import pandas as pd
import fitz                    # PyMuPDF
from docx import Document

# --- ensure project root is on sys.path so "backend" is importable from /pages ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Backend
from backend.pipeline import run_teamreadi_pipeline
from backend.calendar_backend import llm_explain_employee

# ---------- UI shell ----------
st.set_page_config(page_title="TeamReadi — Results", layout="wide")
st.title("TeamReadi — Ranked Results")

# ---------- Tile Styling ----------
st.markdown(
    """
<style>
html, body, .stApp, [data-testid="stAppViewContainer"], .main {
  background-color: #ffffff !important;
}

/* Navy cards for each employee */
.tr-card {
  background: #10233D;
  border-radius: 18px;
  padding: 14px 14px 16px 14px;
  color: #ffffff;
  box-shadow: 0 10px 24px rgba(16, 35, 61, 0.22);
  margin-bottom: 18px;
  min-height: 230px;
}

/* Top name bar (employee ID) */
.tr-card-header {
  color: #FFB020;
  font-weight: 800;
  letter-spacing: 0.8px;
  font-size: 1.0rem;
  text-transform: uppercase;
  margin-bottom: 6px;
}

/* Row with score + icon */
.tr-score-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
}

.tr-score-main {
  display: flex;
  flex-direction: column;
}

.tr-score-value {
  font-size: 1.9rem;
  font-weight: 800;
  color: #FF8A1E;
  line-height: 1.0;
}

.tr-score-label {
  font-size: 0.85rem;
  font-weight: 600;
}

.tr-hardhat-icon {
  font-size: 2.1rem;
}

/* Metrics block */
.tr-metrics {
  font-size: 0.9rem;
  margin-top: 4px;
  margin-bottom: 6px;
}

.tr-metrics div {
  margin-bottom: 2px;
}

/* Highlights */
.tr-highlights-title {
  color: #FFD34D;
  font-weight: 700;
  font-size: 0.9rem;
  margin-top: 4px;
  margin-bottom: 2px;
}

.tr-highlights {
  list-style: none;
  padding-left: 0;
  margin: 0;
  font-size: 0.8rem;
}

.tr-highlights li {
  margin-bottom: 2px;
}

.tr-highlights span.icon {
  display: inline-block;
  width: 16px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Helpers: files & text ----------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = []
    with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)

def extract_text_from_docx(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as f:
        doc = Document(f)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_any(upload) -> str:
    name = getattr(upload, "name", "file.txt").lower()
    data = upload.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    if name.endswith(".docx"):
        return extract_text_from_docx(data)
    return data.decode(errors="ignore")

def read_text_from_url(url: str) -> str:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.text
    except Exception:
        return ""

def filename_stem(fname: str) -> str:
    return re.sub(r"\.[^.]+$", "", fname or "").strip()

def canonical_emp_id_from_name(name: str) -> str:
    """
    Normalize anything like:
      'Employee 1', 'employee_01', 'EMPLOYEE-001', 'John Doe - employee 7'
    to a canonical 'Employee_007'.

    If we don't find an 'employee + number' pattern, we fall back to the stem.
    """
    stem = filename_stem(name)
    m = re.search(r"employee[\s_\-]*([0-9]+)", stem, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        return f"Employee_{n:03d}"
    return stem

# ---------- PDF report ----------
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

def build_pdf(results: List[Dict[str, Any]], params: Dict[str, Any]) -> bytes:
    """
    Build a multi-page PDF:
    - Page 1: summary table of all candidates
    - Following pages: one page per candidate with explanation + highlights.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    w, h = LETTER
    margin = 72
    y = h - margin

    def line(txt, dy=14, size=11, bold=False):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(margin, y, txt)
        y -= dy

    # Summary page
    line("TeamReadi — Ranked Results", dy=20, size=16, bold=True)
    line(f"Window: {params.get('start_date')} to {params.get('end_date')}")
    line(
        f"Workdays: {', '.join(params.get('workdays', []))}   "
        f"Max hrs/day: {params.get('max_hours')}   "
        f"α (Skill weight): {params.get('alpha')}"
    )
    y -= 8

    # Summary table data
    table_data = [["Rank", "Candidate", "ReadiScore", "Skill Match", "Avail. hrs"]]
    for i, r in enumerate(results, start=1):
        m = r["metrics"]
        rs = f"{m['readiscore']:.0f}%"
        sm = f"{m['skill_match_pct']:.0f}%"
        hrs = f"{int(round(m['availability_hours']))}"
        table_data.append([i, r["emp_id"], rs, sm, hrs])

    t = Table(table_data, colWidths=[40, 160, 90, 90, 90])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.05, 0.22, 0.37)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (2, 1), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.whitesmoke, colors.lightgrey]),
            ]
        )
    )
    tw, th = t.wrapOn(c, w - 2 * margin, y - margin)
    t.drawOn(c, margin, max(margin, y - th))
    c.showPage()

    # Per-candidate pages
    for i, r in enumerate(results, start=1):
        m = r["metrics"]
        emp_id = r["emp_id"]

        # Explanation using backend LLM helper
        try:
            explanation = llm_explain_employee(emp_id, m, project_name="TeamReadi Project")
        except Exception:
            explanation = "Explanation unavailable due to an API error."

        # Highlights text
        highlights = r.get("highlights", [])
        highlight_lines = []
        for hlt in highlights[:8]:
            prefix = "✓ " if hlt.get("met") else "× "
            highlight_lines.append(prefix + (hlt.get("skill") or ""))

        y = h - margin
        line(f"{i}. {emp_id}", dy=20, size=15, bold=True)
        line(f"ReadiScore: {m['readiscore']:.1f}%", size=12)
        line(
            f"Skill Match: {m['skill_match_pct']:.1f}%   "
            f"Available Hours: {m['availability_hours']:.1f}",
            size=11,
        )
        line("", dy=6)

        line("Highlights:", bold=True)
        for hl in highlight_lines:
            wrapped = textwrap.wrap(hl, width=80)
            for wline in wrapped:
                line("  " + wline)
        line("", dy=4)

        line("Why this score:", bold=True)
        wrapped_exp = textwrap.wrap(explanation, width=90)
        for wline in wrapped_exp:
            line(wline)
        c.showPage()

    c.save()
    return buf.getvalue()

# ---------- Orchestrator (runs automatically) ----------
with st.spinner("Analyzing resumes, project requirements, and calendar…"):
    ss = st.session_state

    if "resumes" not in ss or not ss["resumes"]:
        st.error("No resumes found. Please return to the start page and upload resumes.")
        st.stop()

    # Rebuild in-memory file-like objects from session_state
    resumes_raw = [
        type("Mem", (), {"name": x["name"], "read": lambda self=None, d=x["data"]: d})
        for x in ss.get("resumes", [])
    ]
    req_files = [
        type("Mem", (), {"name": x["name"], "read": lambda self=None, d=x["data"]: d})
        for x in ss.get("req_files", [])
    ]

    req_url      = ss.get("req_url", "")
    cal_method   = ss.get("cal_method", "Calendar link")
    cal_link     = ss.get("cal_link", "")
    start_date   = dt.date.fromisoformat(ss.get("start_date", str(dt.date.today())))
    end_date     = dt.date.fromisoformat(ss.get("end_date", str(dt.date.today() + dt.timedelta(days=30))))
    workdays_l   = ss.get("workdays", ["Mon", "Tue", "Wed", "Thu", "Fri"])
    max_hours    = float(ss.get("max_hours", 8.0))
    alpha        = float(ss.get("alpha", 0.7))  # used only for PDF context

    # Map workday labels -> weekday ints
    wd_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    working_days = {wd_map[d] for d in workdays_l if d in wd_map}
    if not working_days:
        working_days = {0, 1, 2, 3, 4}

    # Build project text from uploaded files + optional URL
    job_parts = []
    for f in req_files:
        job_parts.append(extract_text_from_any(f))
    if req_url:
        job_parts.append(read_text_from_url(req_url))
    project_text = "\n\n".join([p for p in job_parts if p])

    if not project_text.strip():
        st.error("No project requirements text found (file or URL). Please go back and upload or link a project.")
        st.stop()

    # Build resume dictionary: {Employee_ID: resume_text}
    resumes_dict: Dict[str, str] = {}
    for up in resumes_raw:
        resume_text = extract_text_from_any(up)
        emp_id = canonical_emp_id_from_name(up.name)   # robust Employee_### id
        if emp_id:
            resumes_dict[emp_id] = resume_text

    if not resumes_dict:
        st.error("Could not derive any employee IDs from resume filenames.")
        st.stop()

    # Calendar handling
    if cal_method == "Calendar link":
        if not cal_link.strip():
            st.error("You selected 'Calendar link' but did not provide a URL.")
            st.stop()
        calendar_url = cal_link.strip()
    else:
        # For now we keep demo mode disabled until we wire a proper random-hours branch
        st.error("Randomize hours (demo mode) is not implemented yet. Please use 'Calendar link'.")
        st.stop()

    # Run the unified TeamReadi pipeline
    ranked_rows = run_teamreadi_pipeline(
        calendar_url=calendar_url,
        start_date=start_date,
        end_date=end_date,
        working_days=working_days,
        hours_per_day=max_hours,
        project_text=project_text,
        resumes=resumes_dict,
    )

# ---------- Transform results for tiles & PDF ----------
results: List[Dict[str, Any]] = []
for row in ranked_rows:
    emp_id = row["employee_id"]
    m = row["metrics"]
    results.append(
        {
            "emp_id": emp_id,
            "metrics": m,
            "highlights": row["highlights"],
        }
    )

# Ensure sorted
results = sorted(results, key=lambda r: r["metrics"]["readiscore"], reverse=True)

if not results:
    st.warning("No candidates were scored.")
    st.stop()

# ---------- Render ranked tiles ----------
st.markdown("### Ranked Candidates")

tiles_per_row = 4  # about 4 tiles per row

for idx, r in enumerate(results):
    if idx % tiles_per_row == 0:
        cols = st.columns(tiles_per_row, gap="medium")
    col = cols[idx % tiles_per_row]

    m = r["metrics"]
    emp_id = r["emp_id"]
    score = f"{m['readiscore']:.0f}%"
    skill_pct = f"{m['skill_match_pct']:.0f}%"
    hrs = f"{m['availability_hours']:.0f}"
    highlights = r["highlights"][:6]  # show up to 6 highlights

    # Build highlights HTML
    hl_items = []
    for hlt in highlights:
        met = hlt.get("met")
        icon = "✅" if met else "❌"
        color = "#57D163" if met else "#FF6B6B"
        text = html.escape(hlt.get("skill", "") or "")
        hl_items.append(
            f'<li><span class="icon" style="color:{color};">{icon}</span>{text}</li>'
        )
    hl_html = "".join(hl_items) or "<li><span class='icon'></span>No highlights available.</li>"

    card_html = f"""
    <div class="tr-card">
      <div class="tr-card-header">{html.escape(emp_id)}</div>
      <div class="tr-score-row">
        <div class="tr-score-main">
          <span class="tr-score-value">{score}</span>
          <span class="tr-score-label">ReadiScore™</span>
        </div>
        <div class="tr-hardhat-icon">👷‍♂️</div>
      </div>
      <div class="tr-metrics">
        <div><strong>Skill Match:</strong> {skill_pct}</div>
        <div><strong>Total Time Available:</strong> {hrs} hours</div>
      </div>
      <div class="tr-highlights-title">Highlights:</div>
      <ul class="tr-highlights">
        {hl_html}
      </ul>
    </div>
    """
    with col:
        st.markdown(card_html, unsafe_allow_html=True)

# ---------- Buttons: PDF + Return ----------
params = {
    "start_date": str(st.session_state.get("start_date")),
    "end_date": str(st.session_state.get("end_date")),
    "workdays": st.session_state.get("workdays", ["Mon", "Tue", "Wed", "Thu", "Fri"]),
    "max_hours": st.session_state.get("max_hours", 8),
    "alpha": float(st.session_state.get("alpha", 0.7)),
}

pdf_bytes = build_pdf(results, params)

st.markdown("---")
c1, c2 = st.columns([1, 1])
with c1:
    st.download_button(
        "Download Full Report",
        data=pdf_bytes,
        file_name="teamreadi_results.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

def _reset_and_return():
    for k in (
        "resumes", "req_files", "req_url", "cal_method", "cal_link",
        "random_target", "cal_upload", "start_date", "end_date",
        "workdays", "max_hours", "alpha",
    ):
        if k in st.session_state:
            del st.session_state[k]
    st.switch_page("app.py")

with c2:
    st.button("Return to Start", on_click=_reset_and_return, use_container_width=True)
