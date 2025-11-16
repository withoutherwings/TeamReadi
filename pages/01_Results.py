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

def format_employee_label(raw_id: str) -> str:
    """
    Turn things like:
      'Employee_007 Resume', 'Employee_007_Resume.pdf', 'ice cream sundae.docx'
    into a clean display label:
      'Employee 007', 'Employee 007', 'ice cream sundae'

    This is only for DISPLAY. The underlying employee_id key is unchanged so
    your calendar matching still works.
    """
    if not raw_id:
        return ""

    label = str(raw_id)

    # If something that looks like a path, just keep the filename
    label = os.path.basename(label)

    # Strip common extensions
    for ext in (".pdf", ".docx", ".doc", ".txt"):
        if label.lower().endswith(ext):
            label = label[: -len(ext)]

    # Strip a trailing "resume" word if present (case-insensitive)
    if label.lower().endswith(" resume"):
        label = label[: -len(" resume")]

    # Replace underscores with spaces and normalize extra spaces
    label = label.replace("_", " ")
    label = " ".join(label.split())

    return label


# ---------- Page shell ----------
st.set_page_config(page_title="ReadiReport", layout="wide")
st.title("ReadiReport")

# Keys to clear when “Return to Start” is clicked
RESET_KEYS = (
    "resumes",
    "req_files",
    "req_url",
    "cal_method",
    "cal_link",
    "cal_upload",
    "start_date",
    "end_date",
    "workdays",
    "max_hours",
    "alpha",
    "random_target",
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
    """
    `upload` is a small in-memory object with .name and .read() -> bytes.
    """
    name = getattr(upload, "name", "file.txt").lower()
    data = upload.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    if name.endswith(".docx"):
        return extract_text_from_docx(data)
    return data.decode(errors="ignore")


def read_text_from_url(url: str) -> str:
    """
    Fetch an RFP/job webpage and extract readable text,
    stripping scripts, styles, nav, etc.
    """
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "meta", "noscript", "header", "footer", "nav", "form"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Trim very long pages to keep prompts sane
        return text[:20000]
    except Exception:
        return ""


def filename_stem(fname: str) -> str:
    return re.sub(r"\.[^.]+$", "", fname or "").strip()


# ---------- Calendar math ----------

def daterange_days(start: dt.datetime, end: dt.datetime):
    d = start.date()
    while d <= end.date():
        yield d
        d += dt.timedelta(days=1)


def total_work_hours(
    start: dt.datetime,
    end: dt.datetime,
    working_days: Set[int],
    max_hours_per_day: int,
) -> int:
    total = 0
    for d in daterange_days(start, end):
        if d.weekday() in working_days:
            total += max_hours_per_day
    return total


def fetch_ics_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.content


def busy_blocks_from_ics_for_employee(
    ics_bytes: bytes,
    window_start: dt.datetime,
    window_end: dt.datetime,
    working_days: Set[int],
    emp_tag: Optional[str] = None,
) -> List[Tuple[dt.datetime, dt.datetime]]:
    """
    Return merged busy blocks for a single employee, based on a shared calendar.

    If emp_tag is provided (e.g. 'Employee_001'), only events whose SUMMARY contains
    that tag are counted as busy for this employee.
    """
    cal = Calendar.from_ical(ics_bytes)
    blocks: List[Tuple[dt.datetime, dt.datetime]] = []

    for comp in cal.walk("VEVENT"):
        dtstart = comp.get("dtstart").dt
        dtend = comp.get("dtend").dt

        # Filter by employee tag in the SUMMARY line, if requested
        if emp_tag:
            summary = str(comp.get("summary", ""))
            if emp_tag not in summary:
                continue

        if isinstance(dtstart, dt.date) and not isinstance(dtstart, dt.datetime):
            dtstart = dt.datetime.combine(dtstart, dt.time.min).replace(tzinfo=UTC)
        if isinstance(dtend, dt.date) and not isinstance(dtend, dt.datetime):
            dtend = dt.datetime.combine(dtend, dt.time.min).replace(tzinfo=UTC)

        s = max(window_start, dtstart)
        e = min(window_end, dtend)
        if e > s and (s.weekday() in working_days or e.weekday() in working_days):
            blocks.append((s, e))

    # merge overlaps
    blocks.sort(key=lambda x: x[0])
    merged: List[List[dt.datetime]] = []
    for s, e in blocks:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    return [(s, e) for s, e in merged]


def remaining_hours_for_employee(
    ics_bytes: bytes,
    emp_tag: Optional[str],
    window_start: dt.datetime,
    window_end: dt.datetime,
    working_days: Set[int],
    max_hours_per_day: int,
) -> int:
    """
    Compute remaining hours for ONE employee, by filtering calendar events
    using emp_tag (e.g. 'Employee_001').
    """
    baseline = total_work_hours(window_start, window_end, working_days, max_hours_per_day)
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


# ---------- Highlights from project + candidate profiles ----------

def build_highlights_from_profiles(
    project_profile: Dict[str, Any],
    candidate_profile: Dict[str, Any],
    max_items: int = 5,
) -> List[Dict[str, Any]]:
    """
    Build top-N requirement highlights for tiles & PDF based on:
      project_profile["must_have_skills"]
      candidate_profile["candidate_skills"]
    """
    proj_must = [
        str(x).strip()
        for x in project_profile.get("must_have_skills", [])
        if str(x).strip()
    ][:max_items]

    cand_skills = [
        str(s).strip().lower()
        for s in candidate_profile.get("candidate_skills", [])
        if str(s).strip()
    ]
    cand_set = set(cand_skills)

    highlights: List[Dict[str, Any]] = []
    for label in proj_must:
        lbl_lower = label.lower()
        met = lbl_lower in cand_set
        highlights.append({"skill": label, "met": met})

    return highlights


# ---------- PDF report ----------

from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.pdfgen import canvas


def build_pdf(results: List[Dict[str, Any]], params: Dict[str, Any]) -> bytes:
    """
    Build a multi-page PDF:
      - Page 1: project summary + role mix
      - Subsequent pages: one page per candidate with narrative
    """
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

    # ----- Page 1: project summary + role mix -----
    header()
    y = h - 130

    proj_summary = params.get("project_summary", "")
    if proj_summary:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, y, "Project Summary")
        y -= 18
        c.setFont("Helvetica", 10)
        for line in wrap_text(proj_summary, 95):
            if y < 80:
                c.showPage()
                header()
                y = h - 130
                c.setFont("Helvetica", 10)
            c.drawString(80, y, line)
            y -= 14

    role_counts: Dict[str, int] = params.get("role_counts", {})
    if role_counts:
        if y < 110:
            c.showPage()
            header()
            y = h - 130
        y -= 4
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, y, "Recommended role mix for this project")
        y -= 18
        c.setFont("Helvetica", 10)
        for bucket, count in role_counts.items():
            if count <= 0:
                continue
            line = f"• {bucket} — {count} candidate(s) matched"
            if y < 80:
                c.showPage()
                header()
                y = h - 130
                c.setFont("Helvetica", 10)
            c.drawString(80, y, line)
            y -= 14

    c.showPage()

    # ----- Per-candidate pages -----
    window_baseline = params.get("window_baseline", 1) or 1

    for idx, r in enumerate(results, start=1):
        header()
        y = h - 130

        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, y, f"{r['emp_id']}")
        y -= 18

        c.setFont("Helvetica", 10)
        c.drawString(
            72,
            y,
            f"ReadiScore: {int(r['readiscore']*100)}%   (Rank #{idx})",
        )
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

        # Best-aligned strengths
        strengths = [h["skill"] for h in r.get("highlights", []) if h.get("met")]
        gaps = [h["skill"] for h in r.get("highlights", []) if not h.get("met")]

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

        # Gaps / risks
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

        # Availability impact
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

        # Overall recommendation (from role inference narrative)
        fit = (r.get("project_fit_summary") or "").strip()
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
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes


# ---------- Orchestrator (runs automatically) ----------

with st.spinner("Analyzing inputs with AI and calendars…"):
    ss = st.session_state

    # Landing inputs
    resumes_raw = [
        type("Mem", (), {"name": x["name"], "read": (lambda self=None, d=x["data"]: d)})
        for x in ss.get("resumes", [])
    ]
    req_files = [
        type("Mem", (), {"name": x["name"], "read": (lambda self=None, d=x["data"]: d)})
        for x in ss.get("req_files", [])
    ]
    req_url = ss.get("req_url", "")
    cal_method = ss.get("cal_method", "Calendar link")
    cal_link = ss.get("cal_link", "")
    start_date = dt.date.fromisoformat(ss.get("start_date", str(dt.date.today())))
    end_date = dt.date.fromisoformat(ss.get("end_date", str(dt.date.today())))
    workdays_l = ss.get("workdays", ["Mon", "Tue", "Wed", "Thu", "Fri"])
    max_hours = int(ss.get("max_hours", 8))
    alpha = float(ss.get("alpha", 0.7))

    # Window / masks
    start_dt = dt.datetime.combine(start_date, dt.time(8, 0)).replace(tzinfo=UTC)
    end_dt = dt.datetime.combine(end_date, dt.time(17, 0)).replace(tzinfo=UTC)
    wd_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    working_days: Set[int] = {wd_map[d] for d in workdays_l if d in wd_map}

    # Build job text: prefer uploaded files; otherwise use URL
    if req_files:
        job_parts = [extract_text_from_any(f) for f in req_files]
        job_text = "\n\n".join(p for p in job_parts if p)
    elif req_url:
        job_text = read_text_from_url(req_url)
    else:
        job_text = ""

    if not job_text.strip():
        st.error("Please upload a project requirements file or paste a valid job / RFP URL.")
        st.stop()

    # -------- Project profile via new pipeline helpers --------
    project_profile = build_project_profile(job_text)
    # project_profile keys: project_summary, must_have_skills, nice_to_have_skills

    # -------- Calendar(s) --------
    calendars = []
    if cal_method == "Calendar link" and cal_link:
        try:
            b = fetch_ics_bytes(cal_link)
            calendars.append(
                {
                    "id": filename_stem(cal_link) or "calendar_link",
                    "fname": cal_link,
                    "stem": filename_stem(cal_link),
                    "_bytes": b,
                }
            )
        except Exception:
            # If calendar fails, fall back to full-availability window
            calendars = []

    # If no calendars, availability = baseline (full work window)
    window_baseline = total_work_hours(start_dt, end_dt, working_days, max_hours)

    def availability_for_employee(emp_id: str) -> int:
        """
        Compute remaining hours for this employee ID using the shared calendar,
        if one is provided. Assumes calendar events are tagged with something
        like 'Employee_001' in the SUMMARY line.
        """
        if not calendars:
            return window_baseline

        cal_bytes = calendars[0]["_bytes"]  # single shared calendar

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

    # -------- Build candidates using new candidate profile helper --------
    candidates: List[Dict[str, Any]] = []
    for up in resumes_raw:
        stem = filename_stem(up.name)
        text = extract_text_from_any(up)

        # LLM-based candidate profile
        cand_profile = build_candidate_profile(text, project_profile)

        # Skill match (0–100) then convert to 0–1 for ReadiScore
        skill_match_pct = compute_skill_match(
            project_profile.get("must_have_skills", []),
            cand_profile.get("candidate_skills", []),
        )
        skillfit = skill_match_pct / 100.0

        # Role inference (existing backend)
        role_info = infer_resume_role(job_text, text)

        candidates.append(
            {
                "id": stem,
                "fname": up.name,
                "stem": stem,
                "profile": cand_profile,
                "role_bucket": role_info.get("bucket", "Out-of-scope"),
                "role_title": role_info.get("role_title", "Unspecified role"),
                "project_fit_summary": role_info.get("project_fit_summary", ""),
                "unsuitable_reason": role_info.get("unsuitable_reason", ""),
                "skillfit": skillfit,
            }
        )

    # -------- Compute availability, highlights, ReadiScore --------
    results: List[Dict[str, Any]] = []
    for c in candidates:
        avail = availability_for_employee(c["id"])
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
            }
        )

# ---------- Render results (bucketed tiles) ----------

# Sort by ReadiScore descending
results = sorted(results, key=lambda r: r["readiscore"], reverse=True)

BUCKET_ORDER = ["PM/Admin", "Support/Coordination", "Field/Operator", "Out-of-scope"]
bucket_labels = {
    "PM/Admin": "PM / Admin Roles",
    "Support/Coordination": "Support & Coordination Roles",
    "Field/Operator": "Field / Operator Roles",
    "Out-of-scope": "Out-of-scope / Not a fit",
}

# Group
grouped: Dict[str, List[Dict[str, Any]]] = {b: [] for b in BUCKET_ORDER}
for r in results:
    b = r.get("role_bucket", "Out-of-scope")
    if b not in grouped:
        grouped["Out-of-scope"].append(r)
    else:
        grouped[b].append(r)

# Tiles by bucket
for b in BUCKET_ORDER:
    group = grouped[b]
    if not group:
        continue

    st.subheader(bucket_labels[b])

    cols = st.columns(4)
    for i, r in enumerate(group):
        col = cols[i % 4]
        with col:
            hl = r.get("highlights", [])
            lines = []
            for h in hl:
                icon = "✓" if h.get("met") else "✗"
                lines.append(f"{icon} {h.get('skill','')}")

            highlights_html = "<br/>".join(lines) if lines else "No key requirements clearly met yet."

            col.markdown(
                f"""
<div style="
  background:#10233D;
  color:#FFFFFF;
  border-radius:14px;
  padding:14px 16px;
  margin-bottom:14px;
  box-shadow:0 10px 28px rgba(0,0,0,.35);
  min-height:170px;
">
  <div style="font-size:1.4rem;font-weight:800;color:#FF8A1E;">
    {r["emp_id"]}
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

# Role mix for PDF
role_counts: Dict[str, int] = {}
for r in results:
    bucket = r.get("role_bucket", "Out-of-scope")
    role_counts[bucket] = role_counts.get(bucket, 0) + 1

params = {
    "start_date": str(st.session_state.get("start_date")),
    "end_date": str(st.session_state.get("end_date")),
    "workdays": st.session_state.get("workdays", ["Mon", "Tue", "Wed", "Thu", "Fri"]),
    "max_hours": st.session_state.get("max_hours", 8),
    "window_baseline": total_work_hours(
        dt.datetime.combine(
            dt.date.fromisoformat(st.session_state.get("start_date")), dt.time(8, 0)
        ).replace(tzinfo=UTC),
        dt.datetime.combine(
            dt.date.fromisoformat(st.session_state.get("end_date")), dt.time(17, 0)
        ).replace(tzinfo=UTC),
        {0, 1, 2, 3, 4},  # approximate for the caption; detailed mask already in results
        st.session_state.get("max_hours", 8),
    ),
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
