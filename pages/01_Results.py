# pages/01_Results.py — ReadiReport (ranked tiles + PDF, using new pipeline helpers)

import os, io, re, json, requests
import datetime as dt
from typing import List, Dict, Any, Optional, Tuple, Set

import streamlit as st
import fitz                    # PyMuPDF
from docx import Document
from icalendar import Calendar
from dateutil.tz import UTC
from bs4 import BeautifulSoup

from backend.roles_backend import infer_resume_role
from backend.pipeline import (
    build_project_profile,
    build_candidate_profile,
    compute_skill_match,
)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def format_employee_label(raw_id: str) -> str:
    """
    Make nicer labels like 'Employee 007' instead of 'Employee_007 Resume.pdf'.
    """
    if not raw_id:
        return ""
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", raw_id).strip()
    stem = re.sub(r"resume", "", stem, flags=re.IGNORECASE).strip()
    m = re.match(r"(Employee)[ _-]*(\d+)", stem, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).title()} {m.group(2)}"
    return stem


def filename_stem(path_or_name: str) -> str:
    base = os.path.basename(str(path_or_name))
    return re.sub(r"\.[A-Za-z0-9]+$", "", base)


# ---------- Session / params ----------

REQUIRED_KEYS = [
    "resumes",
    "req_files",
    "req_url",
    "cal_method",
    "cal_link",
    "start_date",
    "end_date",
    "workdays",
    "max_hours",
    "alpha",
    "random_target",
]

RESET_KEYS = list(REQUIRED_KEYS)


def require_session_keys():
    missing = [k for k in REQUIRED_KEYS if k not in st.session_state]
    if missing:
        st.error(
            "Session data is missing. Please return to the landing page and "
            "re-submit the form."
        )
        st.stop()


require_session_keys()

resumes_raw = st.session_state["resumes"]
req_files = st.session_state["req_files"]
req_url = st.session_state["req_url"]
cal_method = st.session_state["cal_method"]
cal_link = st.session_state["cal_link"]
start_date = dt.date.fromisoformat(st.session_state["start_date"])
end_date = dt.date.fromisoformat(st.session_state["end_date"])
workdays_l = st.session_state["workdays"]
max_hours = st.session_state["max_hours"]
alpha = st.session_state["alpha"] or 0.7

# ---------- UI shell ----------

st.set_page_config(page_title="TeamReadi — Results", layout="wide")
st.title("ReadiReport")
st.caption("PM / Admin Roles")

# ---------- Text extraction helpers ----------

def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    text = []
    with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)


def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as f:
        doc = Document(f)
        return "\n".join(p.text for p in doc.paragraphs)


def extract_text_from_upload(obj) -> str:
    """
    Handle both Streamlit UploadedFile and saved dicts like {"name": ..., "data": ...}.
    """
    if hasattr(obj, "name"):
        name = obj.name.lower()
        data = obj.getvalue()
    elif isinstance(obj, dict):
        name = str(obj.get("name", "")).lower()
        data = obj.get("data") or obj.get("bytes") or b""
    else:
        return ""

    if not isinstance(data, (bytes, bytearray)):
        try:
            data = bytes(data)
        except Exception:
            return ""

    if name.endswith(".pdf"):
        return extract_text_from_pdf_bytes(data)
    if name.endswith(".docx"):
        return extract_text_from_docx_bytes(data)
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def read_text_from_url(url: str) -> str:
    if not url:
        return ""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception:
        return ""
    content_type = r.headers.get("Content-Type", "")
    if "pdf" in content_type.lower():
        return extract_text_from_pdf_bytes(r.content)
    # treat as HTML
    soup = BeautifulSoup(r.text, "html.parser")
    return soup.get_text(separator="\n")


# ---------- Calendar & availability helpers ----------

def load_ics_from_bytes(b: bytes) -> Calendar:
    return Calendar.from_ical(b)


def busy_blocks_from_ics_for_employee(
    ics_bytes: bytes,
    window_start: dt.datetime,
    window_end: dt.datetime,
    working_days: Set[int],
    emp_tag: str,
) -> List[Tuple[dt.datetime, dt.datetime]]:
    cal = load_ics_from_bytes(ics_bytes)
    blocks: List[Tuple[dt.datetime, dt.datetime]] = []
    for comp in cal.walk():
        if comp.name != "VEVENT":
            continue

        summary = str(comp.get("SUMMARY", "")).upper()
        if emp_tag.upper() not in summary:
            continue

        dtstart = comp.get("DTSTART").dt
        dtend = comp.get("DTEND").dt

        if dtstart.tzinfo is None:
            dtstart = dtstart.replace(tzinfo=UTC)
        if dtend.tzinfo is None:
            dtend = dtend.replace(tzinfo=UTC)

        if dtend <= window_start or dtstart >= window_end:
            continue

        s = max(dtstart, window_start)
        e = min(dtend, window_end)

        if s.weekday() not in working_days and e.weekday() not in working_days:
            continue

        blocks.append((s, e))

    return blocks


def total_work_hours(
    window_start: dt.datetime,
    window_end: dt.datetime,
    working_days: Set[int],
    max_daily_hours: float,
) -> int:
    cur = window_start
    total = 0.0
    while cur < window_end:
        if cur.weekday() in working_days:
            total += max_daily_hours
        cur += dt.timedelta(days=1)
    return int(total)


def remaining_hours_for_employee(
    ics_bytes: bytes,
    emp_tag: str,
    window_start: dt.datetime,
    window_end: dt.datetime,
    working_days: Set[int],
    max_daily_hours: float,
) -> int:
    baseline = total_work_hours(window_start, window_end, working_days, max_daily_hours)
    if not ics_bytes:
        return baseline

    busy_secs = sum(
        (e - s).total_seconds()
        for s, e in busy_blocks_from_ics_for_employee(
            ics_bytes, window_start, window_end, working_days, emp_tag
        )
    )
    busy_hours = busy_secs / 3600.0
    return max(0, int(round(baseline - busy_hours)))


# ---------- Highlights builder ----------

def build_highlights_from_profiles(
    project_profile: Dict[str, Any],
    candidate_profile: Dict[str, Any],
    max_items: int = 5,
) -> List[Dict[str, Any]]:
    proj_must = [
        str(x).strip()
        for x in project_profile.get("must_have_skills", [])
        if str(x).strip()
    ][:max_items]

    matched_raw = candidate_profile.get("matched_must_have_skills")
    missing_raw = candidate_profile.get("missing_must_have_skills")
    if matched_raw is not None or missing_raw is not None:
        matched_set = {str(s).strip().lower() for s in (matched_raw or []) if str(s).strip()}
        missing_set = {str(s).strip().lower() for s in (missing_raw or []) if str(s).strip()}
        highlights: List[Dict[str, Any]] = []
        for label in proj_must:
            key = label.lower()
            if key in matched_set:
                met = True
            elif key in missing_set:
                met = False
            else:
                met = False
            highlights.append({"skill": label, "met": met})
        return highlights

    cand_skills = [
        str(s).strip().lower()
        for s in candidate_profile.get("candidate_skills", [])
        if str(s).strip()
    ]
    cand_set = set(cand_skills)
    highlights: List[Dict[str, Any]] = []
    for label in proj_must:
        met = label.lower() in cand_set
        highlights.append({"skill": label, "met": met})
    return highlights


# ---------- PDF report ----------

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas


def build_pdf(results: List[Dict[str, Any]], params: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    w, h = LETTER

    def header():
        c.setFont("Helvetica-Bold", 18)
        c.drawString(72, h - 72, "ReadiReport")
        c.setFont("Helvetica", 10)
        c.drawString(72, h - 90, f"Window: {params.get('start_date')} to {params.get('end_date')}")
        c.drawString(
            72,
            h - 104,
            f"Workdays: {', '.join(params.get('workdays', []))}   "
            f"Max hrs/day: {params.get('max_hours')}",
        )

    def wrap_text(text: str, width_chars: int = 92) -> List[str]:
        words = (text or "").split()
        lines, line = [], []
        for w_ in words:
            if sum(len(w) for w in line) + len(line) + len(w_) > width_chars:
                lines.append(" ".join(line))
                line = [w_]
            else:
                line.append(w_)
        if line:
            lines.append(" ".join(line))
        return lines

    # Page 1: project summary + role mix
    header()
    y = h - 130

    proj_summary = params.get("project_summary", "")
    if proj_summary:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, y, "Project summary:")
        y -= 18
        c.setFont("Helvetica", 10)
        for line in wrap_text(proj_summary, 95):
            if y < 80:
                c.showPage()
                header()
                y = h - 130
                c.setFont("Helvetica", 10)
            c.drawString(72, y, line)
            y -= 14
        y -= 10

    role_counts = params.get("role_counts", {})
    if role_counts:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, y, "Role mix by bucket:")
        y -= 18
        c.setFont("Helvetica", 10)
        for bucket, count in role_counts.items():
            if y < 80:
                c.showPage()
                header()
                y = h - 130
                c.setFont("Helvetica", 10)
            c.drawString(80, y, f"- {bucket}: {count} candidate(s)")
            y -= 14

    c.showPage()

    window_baseline = params.get("window_baseline", 1) or 1

    for idx, r in enumerate(results, start=1):
        header()
        y = h - 130

        c.setFont("Helvetica-Bold", 12)
        display_name = format_employee_label(r["emp_id"])
        c.drawString(72, y, f"Candidate: {display_name}")
        y -= 18

        c.setFont("Helvetica", 10)
        c.drawString(72, y, f"ReadiScore: {int(r['readiscore']*100)}%   (Rank #{idx})")
        y -= 14
        c.drawString(
            72,
            y,
            f"Ideal Fit: {r.get('role_title','')}   Bucket: {r.get('role_bucket','')}",
        )
        y -= 14
        c.drawString(
            72,
            y,
            f"Skill match: {int(r['skillfit']*100)}%   Availability: {r['hours']} hrs "
            f"({int((r['hours']/window_baseline)*100)}% of window capacity)",
        )
        y -= 22

        profile = r.get("profile") or {}

        strengths = profile.get("strengths") or [h["skill"] for h in r.get("highlights", []) if h.get("met")]
        gaps = profile.get("gaps") or [h["skill"] for h in r.get("highlights", []) if not h.get("met")]

        c.setFont("Helvetica-Bold", 11)
        c.drawString(72, y, "Best-aligned project strengths:")
        y -= 16
        c.setFont("Helvetica", 10)
        if strengths:
            for s_ in strengths:
                if y < 80:
                    c.showPage()
                    header()
                    y = h - 130
                    c.setFont("Helvetica", 10)
                c.drawString(80, y, f"• {s_}")
                y -= 14
        else:
            c.drawString(80, y, "• No key project requirements clearly met yet.")
            y -= 14

        y -= 8
        if y < 80:
            c.showPage()
            header()
            y = h - 130
        c.setFont("Helvetica-Bold", 11)
        c.drawString(72, y, "Gaps / risks vs must-have skills:")
        y -= 16
        c.setFont("Helvetica", 10)
        if gaps:
            for g_ in gaps:
                if y < 80:
                    c.showPage()
                    header()
                    y = h - 130
                    c.setFont("Helvetica", 10)
                c.drawString(80, y, f"• {g_}")
                y -= 14
        else:
            c.drawString(80, y, "• No major gaps identified against extracted must-have skills.")
            y -= 14

        y -= 8
        if y < 80:
            c.showPage()
            header()
            y = h - 130
        c.setFont("Helvetica-Bold", 11)
        c.drawString(72, y, "Availability impact:")
        y -= 16
        c.setFont("Helvetica", 10)
        avail_ratio = r["hours"] / window_baseline
        if avail_ratio >= 0.75:
            msg = "• High availability; candidate can likely take on a full lead role."
        elif avail_ratio >= 0.45:
            msg = "• Moderate availability; may need to balance this assignment with existing workload."
        else:
            msg = "• Limited availability; may only be suitable for partial support on this project."
        c.drawString(80, y, msg)
        y -= 22

        fit = (profile.get("candidate_summary") or r.get("project_fit_summary") or "").strip()
        if fit:
            if y < 100:
                c.showPage()
                header()
                y = h - 130
            c.setFont("Helvetica-Bold", 11)
            c.drawString(72, y, "Overall recommendation:")
            y -= 16
            c.setFont("Helvetica", 10)
            for line in wrap_text(fit, 95):
                if y < 80:
                    c.showPage()
                    header()
                    y = h - 130
                    c.setFont("Helvetica", 10)
                c.drawString(80, y, line)
                y -= 14

        c.showPage()

    c.save()
    return buf.getvalue()


# ---------- Main pipeline for results page ----------

def fetch_ics_bytes(url: str) -> bytes:
    if not url:
        return b""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.content
    except Exception:
        return b""


def run_results_pipeline() -> Tuple[List[Dict[str, Any]], Dict[str, Any], int]:
    start_dt = dt.datetime.combine(start_date, dt.time(8, 0)).replace(tzinfo=UTC)
    end_dt = dt.datetime.combine(end_date, dt.time(17, 0)).replace(tzinfo=UTC)

    wd_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    working_days: Set[int] = {wd_map[d] for d in workdays_l if d in wd_map}

    # Project text from uploaded files or URL
    if req_files:
        parts = [extract_text_from_upload(f) for f in req_files]
        job_text = "\n\n".join(p for p in parts if p)
    elif req_url:
        job_text = read_text_from_url(req_url)
    else:
        job_text = ""

    project_profile = build_project_profile(job_text)

    calendars = []
    if cal_method == "Calendar link" and cal_link:
        b = fetch_ics_bytes(cal_link)
        if b:
            calendars.append(
                {
                    "id": filename_stem(cal_link) or "calendar_link",
                    "fname": cal_link,
                    "stem": filename_stem(cal_link),
                    "_bytes": b,
                }
            )

    window_baseline = total_work_hours(start_dt, end_dt, working_days, max_hours)

    def availability_for_employee(emp_id: str) -> int:
        if not calendars:
            return window_baseline
        cal_bytes = calendars[0]["_bytes"]
        m = re.search(r"Employee_\d+", emp_id)
        emp_tag = m.group(0) if m else emp_id
        return remaining_hours_for_employee(
            cal_bytes,
            emp_tag,
            start_dt,
            end_dt,
            working_days,
            max_hours,
        )

    candidates: List[Dict[str, Any]] = []
    for up in resumes_raw:
        stem = filename_stem(getattr(up, "name", getattr(up, "filename", "employee")))
        text = extract_text_from_upload(up)

        cand_profile = build_candidate_profile(text, project_profile)
        skill_match_pct = cand_profile.get("skill_match_percent")
        if skill_match_pct is None:
            skill_match_pct = compute_skill_match(
                project_profile.get("must_have_skills", []),
                cand_profile.get("candidate_skills", []),
            )
        skillfit = float(skill_match_pct) / 100.0

        role_info = infer_resume_role(job_text, text)

        candidates.append(
            {
                "id": stem,
                "fname": getattr(up, "name", stem),
                "stem": stem,
                "profile": cand_profile,
                "role_bucket": role_info.get("bucket", "Out-of-scope"),
                "role_title": role_info.get("role_title", "Unspecified role"),
                "project_fit_summary": role_info.get("project_fit_summary", ""),
                "unsuitable_reason": role_info.get("unsuitable_reason", ""),
                "skillfit": skillfit,
            }
        )

    results: List[Dict[str, Any]] = []
    for c in candidates:
        emp_id = c["id"]
        avail = availability_for_employee(emp_id)
        avail_frac = avail / max(window_baseline, 1)

        readiscore = alpha * c["skillfit"] + (1.0 - alpha) * avail_frac

        highlights = build_highlights_from_profiles(
            project_profile,
            c["profile"],
            max_items=5,
        )

        results.append(
            {
                "emp_id": c["id"],
                "skillfit": round(c["skillfit"], 4),
                "hours": int(avail),
                "readiscore": round(readiscore, 4),
                "role_bucket": c["role_bucket"],
                "role_title": c["role_title"],
                "project_fit_summary": c["project_fit_summary"],
                "unsuitable_reason": c["unsuitable_reason"],
                "highlights": highlights,
                "profile": c["profile"],
            }
        )

    return results, project_profile, window_baseline


# ---------- Render results (bucketed tiles) ----------

results, project_profile, window_baseline = run_results_pipeline()
results = sorted(results, key=lambda r: r["readiscore"], reverse=True)

BUCKET_ORDER = ["PM/Admin", "Support/Coordination", "Field/Operator", "Out-of-scope"]
bucket_labels = {
    "PM/Admin": "PM / Admin Roles",
    "Support/Coordination": "Support / Coordination Roles",
    "Field/Operator": "Field / Operator Roles",
    "Out-of-scope": "Out-of-scope / Non-target Roles",
}

grouped: Dict[str, List[Dict[str, Any]]] = {b: [] for b in BUCKET_ORDER}
for r in results:
    b = r.get("role_bucket", "Out-of-scope")
    if b not in grouped:
        grouped["Out-of-scope"].append(r)
    else:
        grouped[b].append(r)

for b in BUCKET_ORDER:
    group = grouped[b]
    if not group:
        continue

    st.subheader(bucket_labels[b])
    cols = st.columns(4)

    for i, r in enumerate(group):
        col = cols[i % 4]
        with col:
            display_name = format_employee_label(r["emp_id"])
            hl = r.get("highlights", [])
            lines = []
            for h in hl:
                icon = "✓" if h.get("met") else "✗"
                lines.append(f"{icon} {h.get('skill','')}")
            highlights_html = "<br>".join(lines) if lines else "No key requirements clearly met yet."

            st.markdown(
                f"""
<div style="
  background-color:#041E3A;
  border-radius:18px;
  padding:18px 20px;
  margin-bottom:18px;
  box-shadow:0 8px 16px rgba(0,0,0,0.25);
  color:white;
  min-height:170px;
">
  <div style="font-size:1.4rem;font-weight:800;color:#FF8A1E;">
    {display_name}
  </div>
  <div style="font-size:1.4rem;font-weight:800;margin:2px 0 4px;">
    ReadiScore: {int(r["readiscore"]*100)}%
  </div>
  <div style="font-size:0.9rem;opacity:0.95;">
    Skill match: {int(r["skillfit"]*100)}% &nbsp;|&nbsp;
    Available: {r["hours"]} hrs
  </div>
  <div style="font-size:0.9rem;margin-top:2px;opacity:0.95;">
    Ideal Fit: {r.get("role_title","")}
  </div>
  <div style="font-size:0.85rem;margin-top:8px;font-weight:700;">
    Highlights
  </div>
  <div style="font-size:0.8rem;margin-top:2px;opacity:0.95;">
    {highlights_html}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

# ---------- PDF + bottom buttons ----------

role_counts: Dict[str, int] = {}
for r in results:
    bucket = r.get("role_bucket", "Out-of-scope")
    role_counts[bucket] = role_counts.get(bucket, 0) + 1

params = {
    "start_date": str(st.session_state.get("start_date")),
    "end_date": str(st.session_state.get("end_date")),
    "workdays": st.session_state.get("workdays", ["Mon", "Tue", "Wed", "Thu", "Fri"]),
    "max_hours": st.session_state.get("max_hours", 8),
    "window_baseline": window_baseline,
    "project_summary": project_profile.get("project_summary", ""),
    "role_counts": role_counts,
}

col_dl, col_reset = st.columns([1, 1])

with col_dl:
    st.download_button(
        "Download Full PDF Report",
        data=build_pdf(results, params),
        file_name="teamreadi_results.pdf",
        mime="application/pdf",
    )

with col_reset:
    if st.button("Return to Start"):
        for k in RESET_KEYS:
            st.session_state.pop(k, None)
        st.markdown(
            "<meta http-equiv='refresh' content='0; URL=https://teamreadi.streamlit.app/'>",
            unsafe_allow_html=True,
        )
        st.stop()
