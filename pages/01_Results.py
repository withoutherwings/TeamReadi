# pages/01_Results.py — ReadiReport (ranked tiles + PDF, using new pipeline helpers)

import os, io, re, json, requests
import datetime as dt
from typing import List, Dict, Any, Optional, Tuple, Set

import streamlit as st
import fitz                    # PyMuPDF
import base64
import pathlib

# Path to the repo root
ROOT = pathlib.Path(__file__).resolve().parents[1]

def load_base64(rel_path: str) -> str:
    file_path = ROOT / rel_path
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

WORKER_ICON = load_base64("assets/worker_icon.png")

from docx import Document
from icalendar import Calendar
from dateutil.tz import UTC
from bs4 import BeautifulSoup
from streamlit_extras.switch_page_button import switch_page
from backend.roles_backend import infer_resume_role
from backend.pipeline import (
    build_project_profile,
    build_candidate_profile,
    compute_skill_match,
)

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

# Make sure page config is set once, at the top
st.set_page_config(page_title="TeamReadi — Results", layout="wide")
st.markdown(
    """
<style>
/* Reduce horizontal space between columns */
div[data-testid="column"] {
    padding-left: 0.2rem !important;
    padding-right: 0.2rem !important;
}

/* Force all cards in a row to equal height */
.teamreadi-card {
    display: flex;
    flex-direction: column;
    height: 100% !important;
}
</style>
""",
    unsafe_allow_html=True,
)



# ---------------------------------------------------------------------------
# Label / ID helpers
# ---------------------------------------------------------------------------

def format_employee_label(raw_id: str) -> str:
    """
    Make nicer labels like 'Employee 007' instead of 'Employee_007 Resume.pdf'.
    """
    if not raw_id:
        return ""
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", raw_id).strip()
    stem = re.sub(r"resume", "", stem, flags=re.IGNORECASE).strip()
    m = re.match(r"(Employee)[ _-]*0*(\d+)", stem, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).title()} {m.group(2).zfill(3)}"
    return stem or "Employee"


def filename_stem(path_or_name: str) -> str:
    base = os.path.basename(str(path_or_name))
    return re.sub(r"\.[A-Za-z0-9]+$", "", base)


def infer_employee_display_name(stem: str, resume_text: str) -> str:
    """
    Try to infer a clean display name like 'Employee 001' from the resume text.
    Fallback to the filename-based label if we can't find it.
    """
    lines_checked = 0
    for line in resume_text.splitlines():
        line = line.strip()
        if not line:
            continue
        lines_checked += 1
        if lines_checked > 30:
            break
        m = re.search(r"(Employee)[ _-]*0*(\d+)", line, flags=re.IGNORECASE)
        if m:
            return f"{m.group(1).title()} {m.group(2).zfill(3)}"

    m2 = re.search(r"(Employee)[ _-]*0*(\d+)", stem, flags=re.IGNORECASE)
    if m2:
        return f"{m2.group(1).title()} {m2.group(2).zfill(3)}"

    return format_employee_label(stem)


def build_employee_calendar_tags(display_name: str, stem: str) -> List[str]:
    """
    Build a *minimal* set of tags for matching this candidate to calendar events.

    Goal: avoid generic words like 'employee' or random resume words
    so different employees don't all match the same events.
    """
    tags: Set[str] = set()
    source = f"{display_name} {stem}"

    # 1) Strong pattern: Employee + number (your typical case)
    m = re.search(r"(Employee)[ _-]*0*(\d+)", source, flags=re.IGNORECASE)
    if m:
        base = m.group(1).lower()       # 'employee'
        num = m.group(2)                # '2' or '010'
        num3 = num.zfill(3)             # '002'
        tags.add(f"{base} {num}")
        tags.add(f"{base} {num3}")
        tags.add(f"{base}_{num3}")
        tags.add(num3)                  # last-resort tag
        return sorted(tags)

    # 2) Fallback: use the full display name as a single tag
    dn = (display_name or "").strip().lower()
    if dn:
        tags.add(dn)

    return sorted(tags)



# ---------------------------------------------------------------------------
# Session / params
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

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
    soup = BeautifulSoup(r.text, "html.parser")
    return soup.get_text(separator="\n")


# ---------------------------------------------------------------------------
# Calendar & availability helpers (CALENDAR = 18 style)
# ---------------------------------------------------------------------------

def busy_blocks_from_ics_for_employee(
    ics_bytes: bytes,
    window_start: dt.datetime,
    window_end: dt.datetime,
    working_days: Set[int],
    tags: Optional[List[str]] = None,
) -> List[Tuple[dt.datetime, dt.datetime]]:
    """
    Return merged busy blocks for a single employee, based on a shared calendar.

    - If `tags` is provided, we only count events whose SUMMARY contains
      at least one of those tags (case-insensitive).
    """
    cal = Calendar.from_ical(ics_bytes)
    blocks: List[Tuple[dt.datetime, dt.datetime]] = []

    tag_list = [t.lower() for t in (tags or []) if t]
    use_tags = len(tag_list) > 0

    for comp in cal.walk("VEVENT"):
        # Safely get dtstart / dtend
        dtstart_prop = comp.get("dtstart")
        dtend_prop = comp.get("dtend")
        if not dtstart_prop or not dtend_prop:
            continue

        dtstart = dtstart_prop.dt
        dtend = dtend_prop.dt

        # Filter by tags in SUMMARY
        if use_tags:
            summary = str(comp.get("summary", "")).lower()
            if not any(tag in summary for tag in tag_list):
                continue

        # Normalise "date" values into datetimes
        if isinstance(dtstart, dt.date) and not isinstance(dtstart, dt.datetime):
            dtstart = dt.datetime.combine(dtstart, dt.time.min).replace(tzinfo=UTC)
        if isinstance(dtend, dt.date) and not isinstance(dtend, dt.datetime):
            dtend = dt.datetime.combine(dtend, dt.time.min).replace(tzinfo=UTC)

        # Clip to the scoring window
        s = max(window_start, dtstart)
        e = min(window_end, dtend)
        if e > s and (s.weekday() in working_days or e.weekday() in working_days):
            blocks.append((s, e))

    # Merge overlapping blocks
    blocks.sort(key=lambda x: x[0])
    merged: List[Tuple[dt.datetime, dt.datetime]] = []
    for s, e in blocks:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            prev_s, prev_e = merged[-1]
            merged[-1] = (prev_s, max(prev_e, e))

    return merged


def total_work_hours(
    window_start: dt.datetime,
    window_end: dt.datetime,
    working_days: Set[int],
    max_daily_hours: float,
) -> int:
    """
    Total *capacity* in hours over the scoring window, assuming max_daily_hours
    on each working day.
    """
    cur = window_start
    total = 0.0
    while cur < window_end:
        if cur.weekday() in working_days:
            total += max_daily_hours
        cur += dt.timedelta(days=1)
    return int(total)


def remaining_hours_for_employee(
    ics_bytes: bytes,
    tags: Optional[List[str]],
    window_start: dt.datetime,
    window_end: dt.datetime,
    working_days: Set[int],
    max_hours_per_day: int,
) -> int:
    """
    Baseline capacity (from total_work_hours) minus busy time from calendar.

    If no calendar bytes are provided, we assume the employee is fully available.
    """
    baseline = total_work_hours(
        window_start, window_end, working_days, max_hours_per_day
    )
    if not ics_bytes:
        return baseline

    busy_secs = sum(
        (e - s).total_seconds()
        for s, e in busy_blocks_from_ics_for_employee(
            ics_bytes, window_start, window_end, working_days, tags
        )
    )
    busy_hours = busy_secs / 3600.0
    return max(0, int(round(baseline - busy_hours)))


# ---------------------------------------------------------------------------
# Highlights builder (trust LLM's matched/missing lists)
# ---------------------------------------------------------------------------

def build_highlights_from_profiles(
    project_profile: Dict[str, Any],
    candidate_profile: Dict[str, Any],
    max_items: int = 5,
) -> List[Dict[str, Any]]:
    """
    Build a small list of highlights for the tiles.

    Uses the candidate_profile buckets:
      - matched_must_have_skills  -> status="yes"  (✅)
      - partial_must_have_skills  -> status="maybe" (⚠️)
      - missing_must_have_skills  -> status="no"   (❌)

    Falls back to simple overlap if those lists are missing.
    """
    matched = [
        str(s).strip()
        for s in candidate_profile.get("matched_must_have_skills") or []
        if str(s).strip()
    ]
    partial = [
        str(s).strip()
        for s in candidate_profile.get("partial_must_have_skills") or []
        if str(s).strip()
    ]
    missing = [
        str(s).strip()
        for s in candidate_profile.get("missing_must_have_skills") or []
        if str(s).strip()
    ]

    highlights: List[Dict[str, Any]] = []

    # Prefer the explicit buckets if we have any signal
    if matched or partial or missing:
        for label in matched:
            if len(highlights) >= max_items:
                break
            highlights.append({"skill": label, "status": "yes", "met": True})
        for label in partial:
            if len(highlights) >= max_items:
                break
            highlights.append({"skill": label, "status": "maybe", "met": False})
        for label in missing:
            if len(highlights) >= max_items:
                break
            highlights.append({"skill": label, "status": "no", "met": False})
        if highlights:
            return highlights

    # Fallback: simple overlap between project must-haves and candidate_skills
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

    for label in proj_must:
        met = label.lower() in cand_set
        highlights.append(
            {
                "skill": label,
                "status": "yes" if met else "no",
                "met": met,
            }
        )

    return highlights


# ---------------------------------------------------------------------------
# PDF report (clean layout + wrapped project title)
# ---------------------------------------------------------------------------

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

def build_pdf(results: List[Dict[str, Any]], params: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    w, h = LETTER

    project_name = params.get("project_name") or ""

    def header():
        # Title with optional project byline
        title = "ReadiReport"
        if project_name:
            title = f"ReadiReport: {project_name}"

        c.setFont("Helvetica-Bold", 18)
        max_width = 470  # printable width before it runs off the page
        y_title = h - 72

        # Split long titles across multiple lines
        words = title.split()
        line: List[str] = []
        for w_ in words:
            test_line = " ".join(line + [w_])
            if c.stringWidth(test_line, "Helvetica-Bold", 18) <= max_width:
                line.append(w_)
            else:
                c.drawString(72, y_title, " ".join(line))
                y_title -= 22
                line = [w_]

        # Draw last line
        if line:
            c.drawString(72, y_title, " ".join(line))

        # Continue header below wrapped title
        y0 = y_title - 18

        c.setFont("Helvetica", 10)
        y0 -= 14
        c.drawString(
            72,
            y0,
            f"Window: {params.get('start_date')} to {params.get('end_date')}",
        )
        y0 -= 14
        c.drawString(
            72,
            y0,
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

    # ---- Page 1: project summary + role mix ----
    header()
    y = h - 180  # a bit lower to account for wrapped titles

    # Optional project window (once, in the summary)
    project_window = (params.get("project_window") or "").strip()
    if project_window:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(72, y, "RFP project window:")
        y -= 16
        c.setFont("Helvetica", 10)
        for line in wrap_text(project_window, 92):
            if y < 80:
                c.showPage()
                header()
                y = h - 130
                c.setFont("Helvetica", 10)
            c.drawString(72, y, line)
            y -= 14
        y -= 10

    # Project summary (already includes company-level context in P2)
    proj_summary = (params.get("project_summary") or "").strip()
    if proj_summary:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, y, "Project summary:")
        y -= 18
        c.setFont("Helvetica", 10)

        paragraphs = [p.strip() for p in proj_summary.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [proj_summary]

        for p in paragraphs:
            for line in wrap_text(p, 92):
                if y < 80:
                    c.showPage()
                    header()
                    y = h - 130
                    c.setFont("Helvetica", 10)
                c.drawString(72, y, line)
                y -= 14
            y -= 8

        y -= 10

    # NOTE: company_requirements block removed on purpose –
    # they are already baked into the narrative summary above.

    # Role mix
    role_mix = params.get("role_mix") or {}
    if not isinstance(role_mix, dict):
        role_mix = {}
    if not role_mix:
        role_mix = params.get("role_counts", {})

    if role_mix:
        if y < 110:
            c.showPage()
            header()
            y = h - 130
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, y, "Role mix suggested by RFP:")
        y -= 18
        c.setFont("Helvetica", 10)
        for bucket, count in role_mix.items():
            if y < 80:
                c.showPage()
                header()
                y = h - 130
                c.setFont("Helvetica", 10)
            c.drawString(80, y, f"- {bucket}: {count} role(s)")
            y -= 14

    c.showPage()

    # ---- Detailed per-candidate pages ----
    window_baseline = params.get("window_baseline", 1) or 1

    for idx, r in enumerate(results, start=1):
        header()
        y = h - 130

        c.setFont("Helvetica-Bold", 12)
        display_name = r.get("display_name") or format_employee_label(r["emp_id"])
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

        strengths = profile.get("strengths") or [
            h["skill"] for h in r.get("highlights", []) if h.get("met")
        ]
        gaps = profile.get("gaps") or [
            h["skill"] for h in r.get("highlights", []) if not h.get("met")
        ]

        # Strengths
        c.setFont("Helvetica-Bold", 11)
        c.drawString(72, y, "Strengths:")
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

        # Gaps
        c.setFont("Helvetica-Bold", 11)
        c.drawString(72, y, "Gaps:")
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
            c.drawString(
                80,
                y,
                "• No major gaps identified against extracted must-have skills.",
            )
            y -= 14

        y -= 8
        if y < 80:
            c.showPage()
            header()
            y = h - 130

        # Availability impact
        c.setFont("Helvetica-Bold", 11)
        c.drawString(72, y, "Availability impact:")
        y -= 16
        c.setFont("Helvetica", 10)
        avail_ratio = r["hours"] / window_baseline
        if avail_ratio >= 0.75:
            msg = "• High availability; candidate can likely take on a full lead role."
        elif avail_ratio >= 0.45:
            msg = (
                "• Moderate availability; may need to balance this assignment "
                "with existing workload."
            )
        else:
            msg = (
                "• Limited availability; may only be suitable for partial "
                "support on this project."
            )
        for line in wrap_text(msg, 92):
            if y < 80:
                c.showPage()
                header()
                y = h - 130
                c.setFont("Helvetica", 10)
            c.drawString(80, y, line)
            y -= 14

        # Overall recommendation
        fit = (
            profile.get("project_fit_summary")
            or profile.get("candidate_background_summary")
            or r.get("project_fit_summary")
            or ""
        ).strip()

        if fit:
            y -= 8
            if y < 100:
                c.showPage()
                header()
                y = h - 130
            c.setFont("Helvetica-Bold", 11)
            c.drawString(72, y, "Overall recommendation:")
            y -= 16
            c.setFont("Helvetica", 10)
            for line in wrap_text(fit, 92):
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



# ---------------------------------------------------------------------------
# Main pipeline for results page
# ---------------------------------------------------------------------------

def fetch_ics_bytes(url: str) -> bytes:
    if not url:
        return b""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.content
    except Exception:
        return b""

def run_results_pipeline() -> Tuple[List[Dict[str, Any]], Dict[str, Any], int, str]:
    start_dt = dt.datetime.combine(start_date, dt.time(8, 0)).replace(tzinfo=UTC)
    end_dt = dt.datetime.combine(end_date, dt.time(17, 0)).replace(tzinfo=UTC)

    wd_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    working_days: Set[int] = {wd_map[d] for d in workdays_l if d in wd_map}

    # ----- Build RFP text -----
    if req_files:
        parts = [extract_text_from_upload(f) for f in req_files]
        job_text = "\n\n".join(p for p in parts if p)
    elif req_url:
        job_text = read_text_from_url(req_url)
    else:
        job_text = ""

    # ----- LLM project profile (this already includes a project_name) -----
    project_profile = build_project_profile(job_text)

    # Use LLM-derived project_title first; fall back to old heuristic ONLY if empty
    project_name = project_profile.get("project_name") or infer_project_name_from_inputs(
        req_files, req_url, job_text
    )

    # ----- Calendar source(s) -----
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

    # Window capacity in hours (used for availability % and Readiscore)
    window_baseline = total_work_hours(start_dt, end_dt, working_days, max_hours)

    def availability_for_employee(tags: List[str]) -> int:
        """
        Compute remaining hours for one employee over the window.
        If no calendar is configured, assume fully available.
        """
        if not calendars:
            return window_baseline
        cal_bytes = calendars[0]["_bytes"]
        return remaining_hours_for_employee(
            cal_bytes,
            tags,
            start_dt,
            end_dt,
            working_days,
            max_hours,
        )

    # ----- Build candidate profiles -----
    candidates: List[Dict[str, Any]] = []
    for up in resumes_raw:
        stem = filename_stem(getattr(up, "name", getattr(up, "filename", "employee")))
        text = extract_text_from_upload(up)

        display_name = infer_employee_display_name(stem, text)
        calendar_tags = build_employee_calendar_tags(display_name, stem)

        cand_profile = build_candidate_profile(text, project_profile)

        # Try to use LLM-computed percentage; fall back to simple overlap
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
                "display_name": display_name,
                "calendar_tags": calendar_tags,
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

    # ----- Compute Readiscore + highlights -----
    results: List[Dict[str, Any]] = []
    for c in candidates:
        avail = availability_for_employee(c.get("calendar_tags", []))
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
                "display_name": c.get("display_name", c["id"]),
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

    return results, project_profile, window_baseline, project_name




# ---------------------------------------------------------------------------
# Render results (bucketed tiles)
# ---------------------------------------------------------------------------

with st.spinner("Analyzing documents and pulling calendar availability..."):
    results, project_profile, window_baseline, project_name = run_results_pipeline()

# Dynamic title *after* we know the project name
title_text = "ReadiReport"
if project_name:
    title_text = f"ReadiReport: {project_name}"
st.title(title_text)

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
            display_name = r.get("display_name") or format_employee_label(r["emp_id"])

            # Build highlight lines with ✓ / ⚠ / ❌
            hl = r.get("highlights", [])
            lines = []
            for h in hl:
                status = h.get("status")
                if status == "yes":
                    icon = "✅"
                elif status == "maybe":
                    icon = "⚠️"
                else:
                    icon = "❌"
                lines.append(f"{icon} {h.get('skill','')}")
            highlights_html = (
                "<br>".join(lines)
                if lines
                else "No specific highlights identified yet."
            )

            st.markdown(
                f"""
<div class="teamreadi-card"
     style="
       background-color:#082A4C;
       border-radius:22px;
       padding:16px 18px 14px;
       margin-bottom:18px;
       box-shadow:0 8px 16px rgba(0,0,0,0.25);
       color:white;
       width:260px;
       min-height:280px;
       margin-left:auto;
       margin-right:auto;
       display:flex;
       flex-direction:column;
">
  <!-- Name -->
  <div style="
      font-size:1.3rem;
      font-weight:800;
      color:#FF8A1E;
      margin-bottom:6px;
      text-transform:uppercase;
  ">
    {display_name}
  </div>

  <!-- Divider under name -->
  <div style="height:1px;background-color:rgba(255,255,255,0.25);margin:4px 0 10px;"></div>

  <!-- Icon + ReadiScore -->
  <div style="display:flex;align-items:center;margin:4px 0 10px;">
    <div style="
    width:100px;
    height:100px;
    display:flex;
    align-items:center;
    justify-content:center;
    margin-right:14px;
">
  <img src="data:image/png;base64,{WORKER_ICON}" style="width:92px;height:92px;" />
</div>

    <div>
      <div style="font-size:2.1rem;font-weight:900;line-height:1.1;color:#FF8A1E;">
        {int(r["readiscore"]*100)}%
      </div>
      <div style="font-size:0.9rem;font-weight:600;opacity:0.95;">
        ReadiScore
      </div>
    </div>
  </div>

  <!-- Divider line -->
  <div style="height:1px;background-color:rgba(255,255,255,0.25);margin:4px 0 8px;"></div>

  <!-- Skill & availability -->
  <div style="font-size:0.9rem;opacity:0.95;">
    Skill Match: {int(r["skillfit"]*100)}%<br>
    Total Time Available: {r["hours"]} hrs
  </div>

  <!-- Role -->
  <div style="font-size:0.9rem;margin-top:4px;opacity:0.95;">
    Ideal Fit: {r.get("role_title","")}
  </div>

  <!-- Highlights -->
  <div style="
      font-size:0.9rem;
      margin-top:10px;
      font-weight:700;
      color:#FF8A1E;
  ">
    Highlights
  </div>
  <div style="font-size:0.8rem;margin-top:2px;opacity:0.95;">
    {highlights_html}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# PDF + bottom buttons
# ---------------------------------------------------------------------------

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
    "project_name": project_name,
    "project_summary": project_profile.get("project_summary", ""),
    "project_window": project_profile.get("project_window", ""),
    "project_location": project_profile.get("project_location", ""),
    "company_requirements": project_profile.get("company_requirements", []),
    "role_mix": project_profile.get("role_mix_by_bucket", {}),
    "role_counts": role_counts,
}

pdf_data = build_pdf(results, params)
st.download_button(
    "Download Full PDF Report",
    data=pdf_data,
    file_name="teamreadi_results.pdf",
    mime="application/pdf",
)

from streamlit_extras.switch_page_button import switch_page

# ---- Return to Start ----
if st.button("Return to Start"):
    st.write(
        "<script>parent.window.location.href='/'</script>",
        unsafe_allow_html=True,
    )


