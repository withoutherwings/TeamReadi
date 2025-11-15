# pages/01_Results.py — ReadiReport (tiles + PDF)

import os, io, re, json, requests
import datetime as dt
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import fitz                    # PyMuPDF
from docx import Document
from icalendar import Calendar
from dateutil.tz import UTC

from backend.roles_backend import infer_resume_role  # role inference

# ---------- UI shell ----------
st.set_page_config(page_title="ReadiReport", layout="wide")
st.title("ReadiReport")

# ---- Session / fast "Return to Start" handling ----

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

# If a prior click on "Return to Start" set this flag, clear and redirect
if st.session_state.get("go_home", False):
    for k in RESET_KEYS:
        st.session_state.pop(k, None)
    st.session_state["go_home"] = False
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


def clean_emp_label(stem: str) -> str:
    """
    Turn 'Employee_004 Resume — Construction Manager'
    into 'Employee_004'.
    """
    if not stem:
        return stem
    s = re.sub(r"\s*resume.*$", "", stem, flags=re.IGNORECASE).strip()
    return s or stem

# ---------- Calendar math ----------

def daterange_days(start: dt.datetime, end: dt.datetime):
    d = start.date()
    while d <= end.date():
        yield d
        d += dt.timedelta(days=1)


def total_work_hours(start: dt.datetime,
                     end: dt.datetime,
                     working_days: set[int],
                     max_hours_per_day: int) -> int:
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

    If emp_tag is provided (e.g. 'Employee_001'), only events whose SUMMARY contains
    that tag are counted as busy for this employee.
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


# ---------- LLM + embeddings ----------

def get_openai():
    from openai import OpenAI
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key) if api_key else None


JOB_SCHEMA = {
    "name": "JobSpec",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "must_have_skills": {"type": "array", "items": {"type": "string"}},
            "nice_to_have_skills": {"type": "array", "items": {"type": "string"}},
            "certifications_required": {"type": "array", "items": {"type": "string"}},
            "years_experience_min": {"type": "integer"},
            "location": {"type": "string"},
            "other_hard_requirements": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["must_have_skills"],
    },
    "strict": True,
}

PROFILE_SCHEMA = {
    "name": "CandidateProfile",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "emails": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
            "skills": {"type": "array", "items": {"type": "string"}},
            "certifications": {"type": "array", "items": {"type": "string"}},
            "roles": {"type": "array", "items": {"type": "string"}},
            "years_experience": {"type": "integer"},
        },
        "required": ["skills"],
    },
    "strict": True,
}


def llm_extract_job(client, job_text: str) -> Dict[str, Any]:
    """
    Use the LLM to summarize the project / RFP into a structured spec.
    Falls back to a simple heuristic if the API is unavailable or JSON
    parsing fails.
    """
    if not client or not (job_text or "").strip():
        words = sorted(set(re.findall(r"[A-Za-z]{3,}", (job_text or "").lower())))[:20]
        return {
            "title": "",
            "summary": job_text[:1000],
            "must_have_skills": words,
            "nice_to_have_skills": [],
            "certifications_required": [],
            "years_experience_min": 0,
            "location": "",
            "other_hard_requirements": [],
        }

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
            "summary": data.get("summary", job_text[:1000]),
            "must_have_skills": data.get("must_have_skills", []),
            "nice_to_have_skills": data.get("nice_to_have_skills", []),
            "certifications_required": data.get("certifications_required", []),
            "years_experience_min": int(data.get("years_experience_min", 0) or 0),
            "location": data.get("location", ""),
            "other_hard_requirements": data.get("other_hard_requirements", []),
        }

    except Exception:
        words = sorted(set(re.findall(r"[A-Za-z]{3,}", (job_text or "").lower())))[:20]
        return {
            "title": "",
            "summary": job_text[:1000],
            "must_have_skills": words,
            "nice_to_have_skills": [],
            "certifications_required": [],
            "years_experience_min": 0,
            "location": "",
            "other_hard_requirements": [],
        }


def llm_extract_profile(client, resume_text: str) -> Dict[str, Any]:
    """
    Extract a structured candidate profile from a resume using the LLM.
    Falls back to simple regex heuristics if the API is unavailable.
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


def embed_text(client, text: str) -> List[float]:
    if not client:
        return []
    e = client.embeddings.create(
        model=st.secrets.get("EMBED_MODEL", "text-embedding-3-large"),
        input=(text or "")[:8000],
    )
    return e.data[0].embedding


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return (dot / (na * nb)) if na and nb else 0.0


# ---------- PDF report ----------

from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.pdfgen import canvas


def build_pdf(results: List[Dict[str, Any]], params: Dict[str, Any]) -> bytes:
    """
    Build a multi-page PDF:
      - Page 1: project summary + recommended roles
      - Subsequent pages: one page per candidate with bullet-style assessment
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    w, h = LETTER

    def header():
        c.setFont("Helvetica-Bold", 18)
        c.drawString(72, h - 72, "ReadiReport")
        c.setFont("Helvetica", 9)
        c.drawString(
            72,
            h - 88,
            f"Window: {params.get('start_date')} to {params.get('end_date')}",
        )
        c.drawString(
            72,
            h - 100,
            f"Workdays: {', '.join(params.get('workdays', []))}   "
            f"Max hrs/day: {params.get('max_hours')}   "
            f"α: {params.get('alpha')}",
        )

    def wrap_text(text: str, width_chars: int = 92) -> List[str]:
        words = (text or "").split()
        lines: List[str] = []
        line: List[str] = []
        for w_ in words:
            if sum(len(w) for w in line) + len(line) + len(w_) > width_chars:
                lines.append(" ".join(line))
                line = [w_]
            else:
                line.append(w_)
        if line:
            lines.append(" ".join(line))
        return lines

    # ----- Page 1: project summary / roles -----
    header()
    y = h - 130

    proj_summary = params.get("project_summary", "")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y, "Project Summary:")
    y -= 18
    c.setFont("Helvetica", 10)
    if proj_summary:
        for line in wrap_text(proj_summary, 95):
            if y < 72:
                c.showPage()
                header()
                y = h - 120
                c.setFont("Helvetica", 10)
            c.drawString(72, y, line)
            y -= 13
    else:
        c.drawString(72, y, "No project description text was provided.")
        y -= 13

    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(72, y, "Recommended roles for this project")
    y -= 16
    c.setFont("Helvetica", 10)

    bucket_counts: Dict[str, int] = {}
    for r in results:
        b = r.get("role_bucket", "Unspecified")
        bucket_counts[b] = bucket_counts.get(b, 0) + 1

    if bucket_counts:
        for bucket, count in bucket_counts.items():
            line = f"- {bucket} ({count} candidate{'s' if count != 1 else ''} matched)"
            if y < 72:
                c.showPage()
                header()
                y = h - 120
                c.setFont("Helvetica", 10)
            c.drawString(80, y, line)
            y -= 13
    else:
        c.drawString(72, y, "- No candidate roles inferred.")
        y -= 13

    c.showPage()

    # ----- Candidate pages -----
    baseline = float(params.get("window_baseline", 0) or 0)

    for rank, r in enumerate(results, start=1):
        header()
        y = h - 130

        label = r.get("label", r["emp_id"])

        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, y, label)
        y -= 18

        c.setFont("Helvetica-Bold", 11)
        c.drawString(
            72,
            y,
            f"ReadiScore: {int(r['readiscore'] * 100)}%   (Rank #{rank})",
        )
        y -= 16

        c.setFont("Helvetica", 10)
        c.drawString(
            72,
            y,
            f"Optimal role: {r.get('role_title','Unspecified role')}",
        )
        y -= 14

        bucket = r.get("role_bucket", "")
        skill_pct = int(r["skillfit"] * 100)
        avail = r["hours"]
        info_line = (
            f"Bucket: {bucket}    Skill match: {skill_pct}%    Availability: {avail} hrs"
        )
        c.drawString(72, y, info_line)
        y -= 20

        # Strengths
        strengths = [h["text"] for h in r.get("highlights", []) if h.get("met")]
        gaps = [h["text"] for h in r.get("highlights", []) if not h.get("met")]

        c.setFont("Helvetica-Bold", 11)
        c.drawString(72, y, "Best-aligned project strengths:")
        y -= 16
        c.setFont("Helvetica", 10)
        if strengths:
            for s in strengths:
                for line in wrap_text(f"• {s}", 92):
                    if y < 72:
                        c.showPage()
                        header()
                        y = h - 120
                        c.setFont("Helvetica", 10)
                    c.drawString(72, y, line)
                    y -= 13
        else:
            c.drawString(
                72,
                y,
                "• No clear strengths against the extracted project requirements.",
            )
            y -= 13

        # Gaps / risks
        y -= 6
        c.setFont("Helvetica-Bold", 11)
        c.drawString(72, y, "Gaps / risks:")
        y -= 16
        c.setFont("Helvetica", 10)
        has_gap = False
        for g in gaps:
            has_gap = True
            for line in wrap_text(f"• {g}", 92):
                if y < 72:
                    c.showPage()
                    header()
                    y = h - 120
                    c.setFont("Helvetica", 10)
                c.drawString(72, y, line)
                y -= 13

        uns = (r.get("unsuitable_reason") or "").strip()
        if uns:
            has_gap = True
            for line in wrap_text(f"• {uns}", 92):
                if y < 72:
                    c.showPage()
                    header()
                    y = h - 120
                    c.setFont("Helvetica", 10)
                c.drawString(72, y, line)
                y -= 13

        if not has_gap:
            c.drawString(72, y, "• No material risks identified from the narrative.")
            y -= 13

        # Availability impact
        y -= 6
        c.setFont("Helvetica-Bold", 11)
        c.drawString(72, y, "Availability impact:")
        y -= 16
        c.setFont("Helvetica", 10)
        if baseline > 0:
            pct_avail = avail / baseline
            if pct_avail >= 0.8:
                desc = "ample availability for the selected window."
            elif pct_avail >= 0.5:
                desc = "moderate availability; may require some workload balancing."
            else:
                desc = "limited availability; candidate may already be heavily committed."
            line = (
                f"• Available {avail:.0f} of ~{baseline:.0f} possible hours "
                f"({pct_avail*100:.0f}% of capacity); {desc}"
            )
        else:
            line = f"• Availability hours: {avail} (baseline not specified)."

        for line_part in wrap_text(line, 92):
            if y < 72:
                c.showPage()
                header()
                y = h - 120
                c.setFont("Helvetica", 10)
            c.drawString(72, y, line_part)
            y -= 13

        # Overall recommendation
        y -= 6
        c.setFont("Helvetica-Bold", 11)
        c.drawString(72, y, "Overall recommendation:")
        y -= 16
        c.setFont("Helvetica", 10)
        rec = (r.get("project_fit_summary") or "").strip()
        if not rec:
            rec = (
                "No detailed narrative recommendation was generated for this candidate."
            )
        for line in wrap_text(f"• {rec}", 92):
            if y < 72:
                c.showPage()
                header()
                y = h - 120
                c.setFont("Helvetica", 10)
            c.drawString(72, y, line)
            y -= 13

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

    # Extract job struct + embedding
    job_struct = llm_extract_job(client, job_text)
    job_emb = embed_text(client, job_text)

    must_have = job_struct.get("must_have_skills", [])

    def build_highlights(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build top-5 project requirement highlights (✓ / ✗) for tiles + PDF.
        """
        skills = [str(s).lower() for s in profile.get("skills", [])]
        highlights: List[Dict[str, Any]] = []
        for req in must_have:
            label = str(req).strip()
            if not label:
                continue
            label_l = label.lower()
            met = any(
                (label_l in s) or (s in label_l)
                for s in skills
            )
            highlights.append({"text": label, "met": met})
            if len(highlights) >= 5:
                break
        return highlights

    # Build candidates (profile + embedding + role inference)
    candidates: List[Dict[str, Any]] = []
    for up in resumes_raw:
        stem = filename_stem(up.name)
        display_label = clean_emp_label(stem)

        text = extract_text_from_any(up)
        prof = llm_extract_profile(client, text)
        prof_emb = embed_text(client, text)

        role_info = infer_resume_role(job_text, text)
        highlights = build_highlights(prof)

        candidates.append(
            {
                "id": stem,
                "label": display_label,
                "fname": up.name,
                "stem": stem,
                "name": prof.get("name", ""),
                "emails": prof.get("emails", []),
                "profile": prof,
                "emb": prof_emb,
                "role_bucket": role_info.get("bucket", "Out-of-scope"),
                "role_title": role_info.get("role_title", "Unspecified role"),
                "project_fit_summary": role_info.get("project_fit_summary", ""),
                "unsuitable_reason": role_info.get("unsuitable_reason", ""),
                "highlights": highlights,
            }
        )

    # Gather calendars (urls/uploads) as bytes list
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

    def score_candidate(profile: Dict[str, Any], job: Dict[str, Any], emb_sim: float) -> float:
        must = set(s.lower() for s in job.get("must_have_skills", []))
        skills = set(s.lower() for s in profile.get("skills", []))

        if must and not (must & skills):
            base = 0.0
        else:
            mh_overlap = len(must & skills) / max(len(must), 1) if must else 0.6

            cert_req = set(s.lower() for s in job.get("certifications_required", []))
            certs = set(s.lower() for s in profile.get("certifications", []))
            cert_match = 1.0 if (not cert_req or cert_req.issubset(certs)) else 0.0

            years_ok = 1.0
            if job.get("years_experience_min", 0) > 0:
                years_ok = (
                    1.0
                    if profile.get("years_experience", 0)
                    >= job["years_experience_min"]
                    else 0.6
                )

            base = (0.5 * mh_overlap + 0.2 * cert_match + 0.3 * emb_sim) * years_ok

        return max(0.0, min(1.0, base))

    def availability_for_employee(emp_id: str) -> int:
        """
        Compute remaining hours for this employee ID using the shared calendar,
        if one is provided. Assumes calendar events are tagged with something
        like 'Employee_001' in the SUMMARY line.
        """
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
        emb_sim = cosine(c["emb"], job_emb)
        skillfit = score_candidate(c["profile"], job_struct, emb_sim)
        avail = availability_for_employee(c["id"])
        avail_frac = avail / max(window_baseline, 1)
        readiscore = alpha * skillfit + (1.0 - alpha) * avail_frac

        results.append(
            {
                "emp_id": c["id"],
                "label": c["label"],
                "skillfit": round(skillfit, 4),
                "hours": int(avail),
                "readiscore": round(readiscore, 4),
                "role_bucket": c["role_bucket"],
                "role_title": c["role_title"],
                "project_fit_summary": c["project_fit_summary"],
                "unsuitable_reason": c["unsuitable_reason"],
                "highlights": c["highlights"],
            }
        )

# ---------- Render results (bucketed tiles) ----------

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
            highlights = r.get("highlights", [])[:5]
            if highlights:
                bullets = "".join(
                    f"<div>{'&#10003;' if h.get('met') else '&#10007;'} {h.get('text','')}</div>"
                    for h in highlights
                )
            else:
                bullets = "<div>No project requirements extracted.</div>"

            ideal_fit = r.get("role_bucket", "Unspecified")

            col.markdown(
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
  <div style="font-size:0.9rem;font-weight:700;color:#FF8A1E;">
    {r.get("label", r["emp_id"])}
  </div>
  <div style="font-size:1.5rem;font-weight:800;margin:4px 0 2px;">
    ReadiScore: {int(r["readiscore"]*100)}%
  </div>
  <div style="font-size:0.8rem;opacity:0.92;margin-bottom:4px;">
    Skill match: {int(r["skillfit"]*100)}%<br/>
    Available: {r["hours"]} hrs<br/>
    Ideal Fit: {ideal_fit}
  </div>
  <div style="font-size:0.78rem;font-weight:600;margin-top:4px;">
    Highlights
  </div>
  <div style="font-size:0.75rem;margin-top:2px;opacity:0.9;">
    {bullets}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

# ---------- PDF + Return (bottom row) ----------

params = {
    "start_date": str(st.session_state.get("start_date")),
    "end_date": str(st.session_state.get("end_date")),
    "workdays": st.session_state.get("workdays", ["Mon", "Tue", "Wed", "Thu", "Fri"]),
    "max_hours": st.session_state.get("max_hours", 8),
    "alpha": float(st.session_state.get("alpha", 0.7)),
    "project_summary": job_struct.get("summary", ""),
    "window_baseline": window_baseline,
}

col_pdf, col_back = st.columns([1, 1])

with col_pdf:
    st.download_button(
        "Download Full PDF Report",
        data=build_pdf(results, params),
        file_name="teamreadi_results.pdf",
        mime="application/pdf",
    )

with col_back:
    if st.button("Return to Start"):
        st.session_state["go_home"] = True
        st.experimental_rerun()
