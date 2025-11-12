# streamlit_app.py
# TeamReadi — Streamlit prototype with OpenAI skill extraction

import os, io, re, json, datetime as dt
from collections import defaultdict

import streamlit as st
import pandas as pd
import fitz              # PyMuPDF
from docx import Document
from icalendar import Calendar

# ---------- Page setup ----------
st.set_page_config(page_title="TeamReadi — Results", layout="wide")
st.title("TeamReadi — Ranked Results")

# ---------- Config / Taxonomy ----------
SKILL_TAXONOMY = {
    "Scheduling": ["primavera", "p6", "scheduling", "resource leveling"],
    "Cost Estimating": ["estimating", "rsmeans", "cost model", "quantity takeoff", "qto"],
    "Blueprint Interpretation": ["blueprint", "plan reading", "shop drawings", "drawings"],
    "Stormwater": ["stormwater", "swppp", "erosion control"],
    "Welding": ["welding", "cwi", "smaw", "gtaw", "mig"],
    "AutoCAD": ["autocad", "cad"],
    "Project Management": ["pm", "project management", "rfi", "submittal", "change order"],
}

WORK_HOURS_PER_DAY = 8
WORK_DAYS = {0,1,2,3,4}  # Mon–Fri

# ---------- Helpers: text extraction ----------
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

def extract_resume_text(upload) -> str:
    name = upload.name.lower()
    data = upload.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    if name.endswith(".docx"):
        return extract_text_from_docx(data)
    return data.decode(errors="ignore")

def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9+ ]", " ", s.lower())

# ---------- Availability math ----------
def daterange_days(start, end):
    d = start.date()
    while d <= end.date():
        yield d
        d += dt.timedelta(days=1)

def total_work_hours(start: dt.datetime, end: dt.datetime) -> int:
    hours = 0
    for d in daterange_days(start, end):
        if d.weekday() in WORK_DAYS:
            hours += WORK_HOURS_PER_DAY
    return hours

def parse_ics_busy_hours(ics_bytes: bytes, start: dt.datetime, end: dt.datetime) -> float:
    cal = Calendar.from_ical(ics_bytes)
    busy = dt.timedelta()
    for comp in cal.walk():
        if comp.name != "VEVENT":
            continue
        dtstart = comp.get("dtstart").dt
        dtend = comp.get("dtend").dt
        if isinstance(dtstart, dt.date) and not isinstance(dtstart, dt.datetime):
            dtstart = dt.datetime.combine(dtstart, dt.time.min)
        if isinstance(dtend, dt.date) and not isinstance(dtend, dt.datetime):
            dtend = dt.datetime.combine(dtend, dt.time.min)
        s = max(start, dtstart)
        e = min(end, dtend)
        if e > s:
            busy += (e - s)
    total = total_work_hours(start, end)
    return min(busy.total_seconds()/3600, total)

def compute_availability(ics_files, start, end):
    total = total_work_hours(start, end)
    if not ics_files:
        return total
    busy = 0.0
    for up in ics_files:
        busy += parse_ics_busy_hours(up.read(), start, end)
    busy = min(busy, total)
    return max(total - busy, 0)

# ---------- Fallback keyword skill detection ----------
def rules_flags(resume_text: str, required_skills: list[str]) -> dict[str, bool]:
    t = normalize(resume_text)
    flags = {}
    for s in required_skills:
        words = [s] + SKILL_TAXONOMY.get(s, [])
        flags[s] = any(w.lower() in t for w in words)
    return flags

# ---------- OpenAI skill extraction ----------
def llm_flags_openai(resume_text: str, job_text: str, required_skills: list[str]) -> dict[str, bool]:
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = st.secrets.get("MODEL_NAME", "gpt-4o-mini") or os.getenv("MODEL_NAME", "gpt-4o-mini")
    if not api_key:
        return rules_flags(resume_text, required_skills)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        props = {s: {"type":"boolean"} for s in required_skills}
        schema = {
            "name":"SkillFlags",
            "schema":{"type":"object","properties":props,"required":list(props.keys()),"additionalProperties":False},
            "strict":True
        }
        prompt = (
            "You are screening a construction resume against a job posting.\n"
            "Return ONLY booleans for each required skill. True only if the resume clearly evidences the skill "
            "(projects, tools, certifications, responsibilities). Be conservative.\n\n"
            f"JOB TEXT:\n{job_text}\n\nRESUME:\n{resume_text}\n"
        )
        resp = client.responses.create(
            model=model,
            input=[{"role":"user","content":prompt}],
            response_format={"type":"json_schema","json_schema":schema},
            temperature=0
        )
        try:
            return json.loads(resp.output[0].content[0].text)
        except Exception:
            return json.loads(getattr(resp, "output_text", "{}"))
    except Exception:
        return rules_flags(resume_text, required_skills)

# ---------- UI: Inputs ----------
colA, colB = st.columns([3,2])
with colA:
    resumes = st.file_uploader("Upload resumes (PDF/DOCX/TXT)", type=["pdf","docx","txt"], accept_multiple_files=True)
    proj_text = st.text_area("Paste project criteria / job post", height=160, placeholder="Paste the RFP / scope / required skills...")
with colB:
    ics_files = st.file_uploader("Upload calendars (.ics) – optional", type=["ics"], accept_multiple_files=True)
    start_date = st.date_input("Evaluation window start", value=dt.date.today())
    end_date = st.date_input("Evaluation window end", value=dt.date.today() + dt.timedelta(days=30))
    end_date = max(end_date, start_date)

job_norm = normalize(proj_text)
detected_required = []
for canon, syns in SKILL_TAXONOMY.items():
    if any(w in job_norm for w in [canon.lower(), *[s.lower() for s in syns]]):
        detected_required.append(canon)
required = st.multiselect("Required skills for this job", options=list(SKILL_TAXONOMY.keys()), default=detected_required)

alpha = st.slider("Weight on Skills (α)", 0.0, 1.0, 0.7, 0.05, help="ReadiScore = α·skills_match + (1–α)·availability_norm")

if st.button("Generate rankings") and resumes:
    start_dt = dt.datetime.combine(start_date, dt.time(8,0))
    end_dt = dt.datetime.combine(end_date, dt.time(17,0))
    window_hours = total_work_hours(start_dt, end_dt)
    avail_hours = compute_availability(ics_files, start_dt, end_dt)

    results = []
    for up in resumes:
        text = extract_resume_text(up)
        emp_id = re.sub(r"\..*$","", up.name)  # filename stem
        flags = llm_flags_openai(text, proj_text, required)
        skills_frac = sum(1 for s,v in flags.items() if v) / max(len(required), 1) if required else 0.0
        avail_frac = avail_hours / max(window_hours, 1) if window_hours else 0.0
        readiscore = alpha*skills_frac + (1 - alpha)*avail_frac
        results.append({
            "emp_id": emp_id,
            "readiscore": round(readiscore, 4),
            "skills_matched": round(skills_frac, 4),
            "hours": int(avail_hours),
            "skills": flags
        })

    results = sorted(results, key=lambda r: r["readiscore"], reverse=True)

    # ---------- Pagination ----------
    PAGE_SIZE = 12
    num_pages = max(1, (len(results) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = st.number_input("Page", min_value=1, max_value=num_pages, value=1, step=1)
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    page_results = results[start:end]
    st.caption(f"Showing {start+1}–{min(end, len(results))} of {len(results)} candidates "
               f"(window {window_hours} hrs, available {int(avail_hours)} hrs)")

    # ---------- Styles & Card renderer ----------
    st.markdown("""
    <style>
    .card {background: linear-gradient(180deg,#08264b 0%,#0b365f 100%); color:#fff; border-radius:14px;
           padding:16px 16px 12px; height:260px; box-shadow:0 6px 18px rgba(0,0,0,.25); font-family:ui-sans-serif;}
    .badge {display:inline-block; background:#ffbf00; color:#001b3a; font-weight:700; padding:2px 8px; border-radius:8px;}
    .header {font-weight:800; letter-spacing:.6px; font-size:.75rem; color:#a9c7ff;}
    .label {color:#d9e7ff; opacity:.9; font-size:.9rem;}
    .value {font-weight:800; font-size:1.4rem;}
    .skills {margin-top:8px;}
    .skill-row {display:flex; align-items:center; gap:6px; margin:3px 0;}
    .ok {width:10px; height:10px; background:#22c55e; border-radius:2px;}
    .no {width:10px; height:10px; background:#ef4444; border-radius:2px;}
    .divider {height:1px; background:rgba(255,255,255,.15); margin:8px 0 10px;}
    .small {font-size:.85rem;}
    </style>
    """, unsafe_allow_html=True)

    def card_html(r):
        skills_html = "".join(
            f'<div class="skill-row"><span class="{ "ok" if ok else "no"}"></span>'
            f'<span class="small">{name}</span></div>'
            for name, ok in r["skills"].items()
        )
        return f"""
        <div class="card">
          <div class="header">RESULTS RANKING</div>
          <div style="display:flex;align-items:center; justify-content:space-between; margin-top:6px;">
            <div><div class="badge">{r["emp_id"]}</div><div class="small" style="opacity:.85;margin-top:2px;">ReadiScore™</div></div>
            <div class="value">{int(r["readiscore"]*100)}%</div>
          </div>
          <div class="divider"></div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
            <div><div class="label">Skills matched</div><div class="value">{int(r["skills_matched"]*100)}%</div></div>
            <div><div class="label">Time available</div><div class="value">{r["hours"]} hrs</div></div>
          </div>
          <div class="divider"></div>
          <div class="label" style="margin-bottom:4px;">Skills</div>
          <div class="skills">{skills_html}</div>
        </div>
        """

    COLS = 3
    cols = st.columns(COLS, gap="large")
    for idx, r in enumerate(page_results):
        with cols[idx % COLS]:
            st.markdown(card_html(r), unsafe_allow_html=True)
