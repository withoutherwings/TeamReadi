# pages/01_Results.py — ReadiReport Ranked Results + Roles + PDF narrative

import os, io, re, json, requests
import datetime as dt
from typing import List, Dict, Any

import streamlit as st
import fitz                    # PyMuPDF
from docx import Document
from icalendar import Calendar
from dateutil.tz import UTC

from backend.roles_backend import infer_resume_role
from backend.skills_backend import (
    extract_project_requirements,
    score_resume_against_requirements,
)

from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

# ---------- UI shell ----------
st.set_page_config(page_title="ReadiReport — Results", layout="wide")
st.title("ReadiReport — Ranked Results")

# ---- Fast "Return to Start" (runs before any heavy work) ----

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

if st.button("Return to Start"):
    # Clear sensitive session values
    for k in RESET_KEYS:
        st.session_state.pop(k, None)

    # Redirect to homepage
    st.markdown(
        "<meta http-equiv='refresh' content='0; URL=https://teamreadi.streamlit.app/'>",
        unsafe_allow_html=True,
    )
    st.stop()


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


# ---------- Calendar math ----------

def daterange_days(start: dt.datetime, end: dt.datetime):
    d = start.date()
    while d <= end.date():
        yield d
        d += dt.timedelta(days=1)


def total_work_hours(
    start: dt.datetime,
    end: dt.datetime,
    working_days: set[int],
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
    working_days: set[int],
    emp_tag: str | None = None,
) -> list[tuple[dt.datetime, dt.datetime]]:
    """
    Return merged busy blocks for a single employee, based on a shared calendar.
    If emp_tag is provided (e.g. 'Employee_001'), only events whose SUMMARY
    contains that tag are counted as busy for this employee.
    """
    cal = Calendar.from_ical(ics_bytes)
    blocks: list[tuple[dt.datetime, dt.datetime]] = []

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
    merged: list[list[dt.datetime]] = []
    for s, e in blocks:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    return [(s, e) for s, e in merged]


def remaining_hours_for_employee(
    ics_bytes: bytes,
    emp_tag: str | None,
    window_start: dt.datetime,
    window_end: dt.datetime,
    working_days: set[int],
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


# ---------- LLM client + job/profile extraction ----------

def get_openai():
    from openai import OpenAI
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key) if api_key else None


def _job_fallback(job_text: str) -> Dict[str, Any]:
    """
    Fallback if OpenAI is unavailable: keep a *short* generic summary so
    we don't paste pages of instructions into the PDF.
    """
    words = sorted(set(re.findall(r"[A-Za-z]{3,}", (job_text or "").lower())))[:20]
    return {
        "title": "",
        "summary": "Summary unavailable (offline mode). See source RFP / project docs.",
        "must_have_skills": words,
        "nice_to_have_skills": [],
        "certifications_required": [],
        "years_experience_min": 0,
        "location": "",
        "other_hard_requirements": [],
    }


def llm_extract_job(client, job_text: str) -> Dict[str, Any]:
    """
    Use the LLM to summarize the project / RFP into a structured spec.
    """
    if not client or not (job_text or "").strip():
        return _job_fallback(job_text)

    prompt = f"""
You are helping a construction firm understand an RFP or job posting.

Read the following project description and return JSON with these keys:
- title: short project or role title
- summary: 3–6 sentence summary of what is being requested
- must_have_skills: list of the 5–12 most critical skills, licenses, or experience
- nice_to_have_skills: list of bonus / preferred skills
- certifications_required: list of required certifications or licenses (OSHA, PE, PMP, etc.)
- years_experience_min: integer years of minimum required experience (0 if not specified)
- location: short location string if it is clearly mentioned
- other_hard_requirements: list of any other hard constraints (shift work, travel %, clearance, etc.)

Project text:
\"\"\"{job_text}\"\"\"

Respond ONLY with a single JSON object and no extra commentary.
"""

    try:
        resp = client.responses.create(
            model=st.secrets.get("MODEL_NAME", "gpt-4.1-mini"),
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=900,
        )
        raw = resp.output[0].content[0].text
        data = json.loads(raw)

        return {
            "title": data.get("title", ""),
            "summary": data.get("summary", "") or _job_fallback(job_text)["summary"],
            "must_have_skills": data.get("must_have_skills", []),
            "nice_to_have_skills": data.get("nice_to_have_skills", []),
            "certifications_required": data.get("certifications_required", []),
            "years_experience_min": int(data.get("years_experience_min", 0) or 0),
            "location": data.get("location", ""),
            "other_hard_requirements": data.get("other_hard_requirements", []),
        }

    except Exception:
        return _job_fallback(job_text)


def llm_extract_profile(client, resume_text: str) -> Dict[str, Any]:
    """
    Extract a structured candidate profile from a resume.
    """
    if not client or not (resume_text or "").strip():
        words = list(set(re.findall(r"[A-Za-z]{3,}", (resume_text or "").lower())))[:50]
        emails = re.findall(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            resume_text or "",
        )
        return {
            "name": "",
            "emails": emails,
            "summary": (resume_text or "")[:800],
            "skills": words,
            "certifications": [],
            "roles": [],
            "years_experience": 0,
        }

    prompt = f"""
You are analyzing a construction industry resume.

Read the resume text and return a JSON object with:
- name: candidate name (string, may be empty if not obvious)
- emails: list of email addresses found
- summary: 2–4 sentence summary of their background
- skills: list of 15–40 key skills / technologies / equipment / domains mentioned
- certifications: list of licenses or certifications (PE, PMP, OSHA 30, etc.)
- roles: list of typical role titles they have held (Project Manager, Superintendent, Estimator, Operator, etc.)
- years_experience: integer estimate of total years of relevant experience

Resume:
\"\"\"{resume_text}\"\"\"

Respond ONLY with a single JSON object and no extra commentary.
"""

    try:
        resp = client.responses.create(
            model=st.secrets.get("MODEL_NAME", "gpt-4.1-mini"),
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=900,
        )
        raw = resp.output[0].content[0].text
        data = json.loads(raw)

        return {
            "name": data.get("name", ""),
            "emails": data.get("emails", []),
            "summary": data.get("summary", (resume_text or "")[:800]),
            "skills": data.get("skills", []),
            "certifications": data.get("certifications", []),
            "roles": data.get("roles", []),
            "years_experience": int(data.get("years_experience", 0) or 0),
        }

    except Exception:
        # Fallback to simple regex extraction
        words = list(set(re.findall(r"[A-Za-z]{3,}", (resume_text or "").lower())))[:50]
        emails = re.findall(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            resume_text or "",
        )
        return {
            "name": "",
            "emails": emails,
            "summary": (resume_text or "")[:800],
            "skills": words,
            "certifications": [],
            "roles": [],
            "years_experience": 0,
        }


# ---------- PDF report ----------

def build_pdf(results: List[Dict[str, Any]], params: Dict[str, Any]) -> bytes:
    """
    Build a multi-page PDF:
      - Page 1: summary table of ranked candidates
      - Subsequent pages: one section per candidate with narrative
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    w, h = LETTER

    # Header helper for page 1
    def header():
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, h - 72, "ReadiReport — Ranked Results")
        c.setFont("Helvetica", 10)
        c.drawString(
            72,
            h - 90,
            f"Window: {params.get('start_date')} to {params.get('end_date')}",
        )
        c.drawString(
            72,
            h - 104,
            f"Workdays: {', '.join(params.get('workdays', []))}   "
            f"Max hrs/day: {params.get('max_hours')}   "
            f"α: {params.get('alpha')}",
        )

    # Simple text wrapper
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

    # ----- Page 1: summary table -----
    header()
    y = h - 130

    data = [["Rank", "Candidate", "Role", "ReadiScore", "SkillFit", "Avail. hrs"]]
    for i, r in enumerate(results, start=1):
        data.append(
            [
                i,
                r["emp_id"],
                r.get("role_title", ""),
                f"{int(r['readiscore'] * 100)}%",
                f"{int(r['skillfit'] * 100)}%",
                f"{r['hours']}",
            ]
        )

    t = Table(data, colWidths=[40, 140, 160, 80, 80, 70])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.05, 0.22, 0.37)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("ALIGN", (3, 1), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ]
        )
    )
    wtbl, htbl = t.wrapOn(c, w - 144, h - 200)
    t.drawOn(c, 72, y - htbl)
    c.showPage()

    # ----- Per-candidate pages -----
    proj_summary = params.get("project_summary", "")

    for r in results:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, h - 72, f"{r['emp_id']}")
        c.setFont("Helvetica", 11)
        c.drawString(72, h - 88, f"Optimal role: {r.get('role_title','')}")
        c.setFont("Helvetica", 10)
        c.drawString(
            72,
            h - 104,
            f"Bucket: {r.get('role_bucket','')}   "
            f"SkillFit: {int(r['skillfit']*100)}%   "
            f"ReadiScore: {int(r['readiscore']*100)}%   "
            f"Avail: {r['hours']} hrs",
        )

        y = h - 130

        # What the project asked for (shared summary)
        if proj_summary:
            c.setFont("Helvetica-Bold", 11)
            c.drawString(72, y, "What the project asked for:")
            y -= 16
            c.setFont("Helvetica", 10)
            for line in wrap_text(proj_summary, 96):
                if y < 80:
                    c.showPage()
                    y = h - 72
                    c.setFont("Helvetica", 10)
                c.drawString(80, y, line)
                y -= 14

        # How this person fits
        fit = r.get("project_fit_summary", "")
        if fit:
            y -= 10
            if y < 80:
                c.showPage()
                y = h - 72
            c.setFont("Helvetica-Bold", 11)
            c.drawString(72, y, "How this person could fit this project:")
            y -= 16
            c.setFont("Helvetica", 10)
            for line in wrap_text(fit, 96):
                if y < 80:
                    c.showPage()
                    y = h - 72
                    c.setFont("Helvetica", 10)
                c.drawString(80, y, line)
                y -= 14

        # Unsuitable reason (if any)
        uns = r.get("unsuitable_reason", "")
        if uns:
            y -= 10
            if y < 80:
                c.showPage()
                y = h - 72
            c.setFont("Helvetica-Bold", 11)
            c.drawString(72, y, "Why this person may not be suitable:")
            y -= 16
            c.setFont("Helvetica", 10)
            for line in wrap_text(uns, 96):
                if y < 80:
                    c.showPage()
                    y = h - 72
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
    cal_upload = ss.get("cal_upload")
    start_date = dt.date.fromisoformat(ss.get("start_date", str(dt.date.today())))
    end_date = dt.date.fromisoformat(ss.get("end_date", str(dt.date.today())))
    workdays_l = ss.get("workdays", ["Mon", "Tue", "Wed", "Thu", "Fri"])
    max_hours = int(ss.get("max_hours", 8))
    alpha = float(ss.get("alpha", 0.7))

    # Window / masks
    start_dt = dt.datetime.combine(start_date, dt.time(8, 0)).replace(tzinfo=UTC)
    end_dt = dt.datetime.combine(end_date, dt.time(17, 0)).replace(tzinfo=UTC)
    wd_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    working_days = {wd_map[d] for d in workdays_l if d in wd_map}

    # Build job text (files + url)
    job_parts = []
    for f in req_files:
        job_parts.append(extract_text_from_any(f))
    if req_url:
        job_parts.append(read_text_from_url(req_url))
    job_text = "\n\n".join([p for p in job_parts if p])

    # LLM client
    client = get_openai()

    # LLM-derived project struct (for summary/narrative)
    job_struct = llm_extract_job(client, job_text)

    # Structured requirements for scoring/tiles
    requirements = extract_project_requirements(job_text)

    # Build candidates
    candidates: List[Dict[str, Any]] = []
    for up in resumes_raw:
        stem = filename_stem(up.name)
        text = extract_text_from_any(up)
        prof = llm_extract_profile(client, text)

        # Per-skill scoring from skills_backend
        scored = score_resume_against_requirements(requirements, text)
        per_skill = scored.get("per_skill", [])
        skill_pct = float(scored.get("skill_match_pct", 0.0))

        # Role inference (bucket + narrative)
        role_info = infer_resume_role(job_text, text)

        candidates.append(
            {
                "id": stem,
                "fname": up.name,
                "stem": stem,
                "name": prof.get("name", ""),
                "emails": prof.get("emails", []),
                "profile": prof,
                "skill_pct": skill_pct,
                "top_needs": per_skill[:5],  # top 5 requirements for tiles
                "role_bucket": role_info.get("bucket", "Out-of-scope"),
                "role_title": role_info.get("role_title", "Unspecified role"),
                "project_fit_summary": role_info.get("project_fit_summary", ""),
                "unsuitable_reason": role_info.get("unsuitable_reason", ""),
            }
        )

    # Calendars
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
            pass
    elif cal_method == "Manual Entry / Upload" and cal_upload:
        b = cal_upload["data"]
        calendars.append(
            {
                "id": filename_stem(cal_upload["name"]),
                "fname": cal_upload["name"],
                "stem": filename_stem(cal_upload["name"]),
                "_bytes": b,
            }
        )

    # If no calendars, availability = baseline (full work window)
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

    results: List[Dict[str, Any]] = []
    for c in candidates:
        skillfit = max(0.0, min(1.0, c["skill_pct"] / 100.0))

        avail = availability_for_employee(c["id"])
        avail_frac = avail / max(window_baseline, 1)

        readiscore = alpha * skillfit + (1.0 - alpha) * avail_frac

        results.append(
            {
                "emp_id": c["id"],
                "skillfit": round(skillfit, 4),
                "hours": int(avail),
                "readiscore": round(readiscore, 4),
                "role_bucket": c["role_bucket"],
                "role_title": c["role_title"],
                "project_fit_summary": c["project_fit_summary"],
                "unsuitable_reason": c["unsuitable_reason"],
                "top_needs": c["top_needs"],
            }
        )

# ---------- Render results (bucketed tiles only) ----------

results = sorted(results, key=lambda r: r["readiscore"], reverse=True)

BUCKET_ORDER = ["PM/Admin", "Support/Coordination", "Field/Operator", "Out-of-scope"]
bucket_labels = {
    "PM/Admin": "PM / Admin Roles",
    "Support/Coordination": "Support & Coordination Roles",
    "Field/Operator": "Field / Operator Roles",
    "Out-of-scope": "Out-of-scope / Not a fit",
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
            needs = r.get("top_needs", [])[:5]
            lines: List[str] = []
            for need in needs:
                label = str(need.get("label", "")).strip()
                status = str(need.get("match_status", "no_match"))
                if not label:
                    continue
                if status == "strong_match":
                    icon = "✅"
                elif status == "partial_match":
                    icon = "⚠️"
                else:
                    icon = "✖️"
                lines.append(f"{icon} {label}")
            needs_html = "<br/>".join(lines) if lines else "No key requirements clearly met. See PDF for details."

            st.markdown(
                f"""
<div style="
  background:#10233D;
  color:#FFFFFF;
  border-radius:14px;
  padding:12px 14px;
  margin-bottom:12px;
  box-shadow:0 10px 28px rgba(0,0,0,.35);
  min-height:160px;
">
  <div style="font-size:0.85rem;font-weight:700;color:#FF8A1E;">
    {r["emp_id"]} — {r.get("role_title","")}
  </div>
  <div style="font-size:1.5rem;font-weight:800;margin:4px 0 2px;">
    {int(r["readiscore"]*100)}%
  </div>
  <div style="font-size:0.8rem;opacity:0.92;">
    Skill match: {int(r["skillfit"]*100)}%<br/>
    Available: {r["hours"]} hrs
  </div>
  <div style="font-size:0.75rem;margin-top:6px;opacity:0.9;line-height:1.25;">
    {needs_html}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

# ---------- PDF download ----------

params = {
    "start_date": str(st.session_state.get("start_date")),
    "end_date": str(st.session_state.get("end_date")),
    "workdays": st.session_state.get("workdays", ["Mon", "Tue", "Wed", "Thu", "Fri"]),
    "max_hours": st.session_state.get("max_hours", 8),
    "alpha": float(st.session_state.get("alpha", 0.7)),
    "project_summary": job_struct.get("summary", ""),
}

st.download_button(
    "Download Full PDF Report",
    data=build_pdf(results, params),
    file_name="teamreadi_results.pdf",
    mime="application/pdf",
)
