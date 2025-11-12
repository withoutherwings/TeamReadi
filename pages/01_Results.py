# pages/01_Results.py — Auto-run pipeline (LLM parsing + semantic + calendar + PDF)
import os, io, re, json, requests
import datetime as dt
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import pandas as pd
import fitz                    # PyMuPDF
from docx import Document
from icalendar import Calendar
from dateutil.tz import UTC
from rapidfuzz import fuzz

# ---------- UI shell ----------
st.set_page_config(page_title="TeamReadi — Results", layout="wide")
st.title("TeamReadi — Ranked Results")

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
    if name.endswith(".pdf"):  return extract_text_from_pdf(data)
    if name.endswith(".docx"): return extract_text_from_docx(data)
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

def total_work_hours(start: dt.datetime, end: dt.datetime, working_days: set[int], max_hours_per_day: int) -> int:
    total = 0
    for d in daterange_days(start, end):
        if d.weekday() in working_days:
            total += max_hours_per_day
    return total

def fetch_ics_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.content

def busy_blocks_from_ics(ics_bytes: bytes,
                         window_start: dt.datetime,
                         window_end: dt.datetime,
                         working_days: set[int]) -> List[Tuple[dt.datetime, dt.datetime]]:
    # Basic non-recurrence extraction (robust enough for first pass).
    # You can extend with dateutil.rrule expansion if you expect many RRULE calendars.
    cal = Calendar.from_ical(ics_bytes)
    blocks = []
    for comp in cal.walk("VEVENT"):
        dtstart = comp.get("dtstart").dt
        dtend   = comp.get("dtend").dt
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
    merged = []
    for s,e in blocks:
        if not merged or s > merged[-1][1]:
            merged.append([s,e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s,e) for s,e in merged]

def remaining_hours_from_ics(ics_bytes: bytes,
                             window_start: dt.datetime,
                             window_end: dt.datetime,
                             working_days: set[int],
                             max_hours_per_day: int) -> int:
    baseline = total_work_hours(window_start, window_end, working_days, max_hours_per_day)
    busy = sum((e - s).total_seconds() for s,e in busy_blocks_from_ics(ics_bytes, window_start, window_end, working_days)) / 3600.0
    return max(0, int(round(baseline - busy)))

# ---------- LLM + embeddings ----------
def get_openai():
    from openai import OpenAI
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key) if api_key else None

JOB_SCHEMA = {
    "name": "JobSpec",
    "schema": {
        "type":"object","additionalProperties":False,
        "properties":{
            "title":{"type":"string"},
            "summary":{"type":"string"},
            "must_have_skills":{"type":"array","items":{"type":"string"}},
            "nice_to_have_skills":{"type":"array","items":{"type":"string"}},
            "certifications_required":{"type":"array","items":{"type":"string"}},
            "years_experience_min":{"type":"integer"},
            "location":{"type":"string"},
            "other_hard_requirements":{"type":"array","items":{"type":"string"}}
        },
        "required":["must_have_skills"]
    },
    "strict":True
}

PROFILE_SCHEMA = {
    "name": "CandidateProfile",
    "schema": {
        "type":"object","additionalProperties":False,
        "properties":{
            "name":{"type":"string"},
            "emails":{"type":"array","items":{"type":"string"}},
            "summary":{"type":"string"},
            "skills":{"type":"array","items":{"type":"string"}},
            "certifications":{"type":"array","items":{"type":"string"}},
            "roles":{"type":"array","items":{"type":"string"}},
            "years_experience":{"type":"integer"}
        },
        "required":["skills"]
    },
    "strict":True
}

def llm_extract_job(client, job_text: str) -> Dict[str,Any]:
    if not client:
        words = sorted(set(re.findall(r"[A-Za-z]{3,}", (job_text or "").lower())))[:20]
        return {"title":"","summary":job_text[:1000],"must_have_skills":words,
                "nice_to_have_skills":[],"certifications_required":[],"years_experience_min":0,
                "location":"","other_hard_requirements":[]}
    prompt = "Extract a concise hiring spec from the JOB TEXT.\n\nJOB TEXT:\n" + (job_text or "")
    r = client.responses.create(
        model=st.secrets.get("MODEL_NAME","gpt-4o-mini"),
        input=[{"role":"user","content":prompt}],
        response_format={"type":"json_schema","json_schema":JOB_SCHEMA},
        temperature=0
    )
    try:
        return json.loads(r.output[0].content[0].text)
    except Exception:
        return json.loads(getattr(r,"output_text","{}") or "{}")

def llm_extract_profile(client, resume_text: str) -> Dict[str,Any]:
    if not client:
        words = list(set(re.findall(r"[A-Za-z]{3,}", (resume_text or "").lower())))[:50]
        emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", resume_text or "")
        return {"name":"","emails":emails,"summary":resume_text[:800],"skills":words,"certifications":[],"roles":[],"years_experience":0}
    prompt = "Extract candidate profile (name, emails, concise skills & certs, years).\n\nRESUME:\n" + (resume_text or "")
    r = client.responses.create(
        model=st.secrets.get("MODEL_NAME","gpt-4o-mini"),
        input=[{"role":"user","content":prompt}],
        response_format={"type":"json_schema","json_schema":PROFILE_SCHEMA},
        temperature=0
    )
    try:
        return json.loads(r.output[0].content[0].text)
    except Exception:
        return json.loads(getattr(r,"output_text","{}") or "{}")

def embed_text(client, text: str) -> List[float]:
    if not client: return []
    e = client.embeddings.create(model=st.secrets.get("EMBED_MODEL","text-embedding-3-large"), input=(text or "")[:8000])
    return e.data[0].embedding

def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b: return 0.0
    import math
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return (dot/(na*nb)) if na and nb else 0.0

# ---------- AI-assisted calendar↔candidate matching (silent) ----------
def name_sim(a,b):  # robust fuzzy similarity
    return max(fuzz.WRatio(a,b), fuzz.token_set_ratio(a,b), fuzz.partial_ratio(a,b)) / 100.0

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

def ics_identity(ics_bytes: bytes) -> Dict[str,Any]:
    name=""; emails=set()
    try:
        cal = Calendar.from_ical(ics_bytes)
        if cal.get("x-wr-calname"):
            name = str(cal.get("x-wr-calname"))
        for comp in cal.walk("VEVENT"):
            s = str(comp.get("organizer",""))
            emails.update(EMAIL_RE.findall(s.lower()))
            for att in comp.getall("attendee", []):
                emails.update(EMAIL_RE.findall(str(att).lower()))
    except Exception:
        pass
    return {"name":name, "emails":list(emails)}

def match_score(cand: Dict[str,Any], cal: Dict[str,Any]) -> float:
    email_hit = 1.0 if set(map(str.lower, cand.get("emails",[]))) & set(map(str.lower, cal.get("emails",[]))) else 0.0
    stem_sim  = name_sim(cand["stem"], cal["stem"])
    name_s    = name_sim(cand.get("name",""), cal.get("name",""))
    return max(0.0, min(1.0, 0.55*email_hit + 0.30*stem_sim + 0.15*name_s))

def auto_map_calendars(candidates: List[Dict[str,Any]], calendars: List[Dict[str,Any]]) -> Dict[str, Optional[str]]:
    assigned, used = {}, set()
    for c in candidates:
        best_id, best_s = None, 0.0
        for k in calendars:
            s = match_score(c, k)
            if s > best_s and k["id"] not in used:
                best_id, best_s = k["id"], s
        # assign even if low confidence; it’s all behind the scenes
        if best_id:
            assigned[c["id"]] = best_id
            used.add(best_id)
        else:
            assigned[c["id"]] = None
    return assigned

# ---------- PDF report ----------
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

def build_pdf(results, params) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    w, h = LETTER; y = h - 72
    def line(txt, dy=18, size=11, bold=False):
        nonlocal y; c.setFont("Helvetica-Bold" if bold else "Helvetica", size); c.drawString(72, y, txt); y -= dy
    line("TeamReadi — Ranked Results", size=16, bold=True, dy=22)
    line(f"Window: {params['start_date']} to {params['end_date']}")
    line(f"Workdays: {', '.join(params['workdays'])}   Max hrs/day: {params['max_hours']}   α: {params['alpha']}")
    y -= 6
    data = [["Rank","Candidate","ReadiScore","SkillFit","Avail. hrs"]]
    for i,r in enumerate(results, start=1):
        data.append([i, r["emp_id"], f"{int(r['readiscore']*100)}%", f"{int(r['skillfit']*100)}%", f"{r['hours']}"])
    t = Table(data, colWidths=[40,170,90,90,90])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), colors.Color(0.05,0.22,0.37)), ("TEXTCOLOR",(0,0),(-1,0), colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("ALIGN",(2,1),(-1,-1),"CENTER"),
        ("GRID",(0,0),(-1,-1),0.4, colors.grey), ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.whitesmoke, colors.lightgrey]),
    ]))
    wtbl, htbl = t.wrapOn(c, w-144, y-72); t.drawOn(c, 72, y-htbl); c.showPage(); c.save()
    return buf.getvalue()

# ---------- Orchestrator (runs automatically) ----------
with st.spinner("Analyzing inputs with AI and calendars…"):
    ss = st.session_state

    # Landing inputs
    resumes_raw = [type("Mem", (), {"name": x["name"], "read": lambda self= None, d=x["data"]: d}) for x in ss.get("resumes",[])]
    req_files   = [type("Mem", (), {"name": x["name"], "read": lambda self= None, d=x["data"]: d}) for x in ss.get("req_files",[])]
    req_url     = ss.get("req_url","")
    cal_method  = ss.get("cal_method","Calendar Link")
    cal_link    = ss.get("cal_link","")
    cal_upload  = ss.get("cal_upload")
    start_date  = dt.date.fromisoformat(ss.get("start_date", str(dt.date.today())))
    end_date    = dt.date.fromisoformat(ss.get("end_date", str(dt.date.today())))
    workdays_l  = ss.get("workdays", ["Mon","Tue","Wed","Thu","Fri"])
    max_hours   = int(ss.get("max_hours", 8))
    alpha       = float(ss.get("alpha", 0.7))

    # Window / masks
    start_dt = dt.datetime.combine(start_date, dt.time(8,0)).replace(tzinfo=UTC)
    end_dt   = dt.datetime.combine(end_date, dt.time(17,0)).replace(tzinfo=UTC)
    wd_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
    working_days = {wd_map[d] for d in workdays_l if d in wd_map}

    # Build job text (paste + files + url)
    job_parts = []
    for f in req_files:
        job_parts.append(extract_text_from_any(f))
    if req_url: job_parts.append(read_text_from_url(req_url))
    job_text = "\n\n".join([p for p in job_parts if p])

    # LLM client
    client = get_openai()

    # Extract job struct + embedding
    job_struct = llm_extract_job(client, job_text)
    job_emb    = embed_text(client, job_text)

    # Build candidates (profile + embedding)
    candidates = []
    for up in resumes_raw:
        stem = filename_stem(up.name)
        text = extract_text_from_any(up)
        prof = llm_extract_profile(client, text)
        prof_emb = embed_text(client, text)
        candidates.append({
            "id": stem, "fname": up.name, "stem": stem,
            "name": prof.get("name",""), "emails": prof.get("emails",[]),
            "profile": prof, "emb": prof_emb,
        })

    # Gather calendars (urls/uploads) as bytes “list”
    calendars = []
    # If a single link/upload represents a single shared calendar, we’ll still map best per candidate.
    if cal_method == "Calendar Link" and cal_link:
        try:
            b = fetch_ics_bytes(cal_link)
            calendars.append({"id": filename_stem(cal_link) or "calendar_link",
                              "fname": cal_link, "stem": filename_stem(cal_link),
                              "ident": ics_identity(b), "_bytes": b})
        except Exception:
            pass
    elif cal_method == "Manual Entry / Upload" and cal_upload:
        b = cal_upload["data"]
        calendars.append({"id": filename_stem(cal_upload["name"]),
                          "fname": cal_upload["name"], "stem": filename_stem(cal_upload["name"]),
                          "ident": ics_identity(b), "_bytes": b})

    # If users upload multiple calendars on landing later, just extend `calendars` with each.

    # If no calendars, availability = baseline
    window_baseline = total_work_hours(start_dt, end_dt, working_days, max_hours)

    # Auto map calendars to candidates (quietly)
    cand_identities = [{"id": c["id"], "stem": c["stem"], "name": c["name"], "emails": c["emails"]} for c in candidates]
    cal_identities  = [{"id": k["id"], "stem": k["stem"], "name": k["ident"]["name"], "emails": k["ident"]["emails"]} for k in calendars]
    mapping = {}
    if cal_identities:
        def _score(c, k):
            email_hit = 1.0 if set(map(str.lower, c.get("emails",[]))) & set(map(str.lower, k.get("emails",[]))) else 0.0
            stem_sim  = max(fuzz.WRatio(c["stem"], k["stem"]), fuzz.token_set_ratio(c["stem"], k["stem"]), fuzz.partial_ratio(c["stem"], k["stem"])) / 100.0
            name_sim  = max(fuzz.WRatio(c.get("name",""), k.get("name","")), fuzz.token_set_ratio(c.get("name",""), k.get("name","")), fuzz.partial_ratio(c.get("name",""), k.get("name",""))) / 100.0
            return max(0.0, min(1.0, 0.55*email_hit + 0.30*stem_sim + 0.15*name_sim))
        used = set()
        for c in cand_identities:
            best, best_s = None, 0.0
            for k in cal_identities:
                if k["id"] in used: continue
                s = _score(c,k)
                if s > best_s:
                    best, best_s = k["id"], s
            if best:
                mapping[c["id"]] = best; used.add(best)
            else:
                mapping[c["id"]] = None
    else:
        mapping = {c["id"]: None for c in candidates}

    # Compute availability and skill fit; blend to ReadiScore
    def score_candidate(profile: Dict[str,Any], job: Dict[str,Any], emb_sim: float) -> float:
        must = set(s.lower() for s in job.get("must_have_skills", []))
        skills = set(s.lower() for s in profile.get("skills", []))
        if must and not (must & skills):
            base = 0.0
        else:
            mh_overlap = len(must & skills)/max(len(must),1) if must else 0.6
            cert_req = set(s.lower() for s in job.get("certifications_required", []))
            certs = set(s.lower() for s in profile.get("certifications", []))
            cert_match = 1.0 if (not cert_req or cert_req.issubset(certs)) else 0.0
            years_ok = 1.0
            if job.get("years_experience_min",0) > 0:
                years_ok = 1.0 if profile.get("years_experience",0) >= job["years_experience_min"] else 0.6
            base = (0.5*mh_overlap + 0.2*cert_match + 0.3*emb_sim) * years_ok
        return max(0.0, min(1.0, base))

    results = []
    for c in candidates:
        # Skill fit
        emb_sim  = cosine(c["emb"], job_emb)
        skillfit = score_candidate(c["profile"], job_struct, emb_sim)

        # Availability
        cal_id = mapping.get(c["id"])
        if cal_id:
            cal_obj = next((x for x in calendars if x["id"] == cal_id), None)
            avail = remaining_hours_from_ics(cal_obj["_bytes"], start_dt, end_dt, working_days, max_hours) if cal_obj else window_baseline
        else:
            avail = window_baseline  # no calendar: assume fully available in window

        avail_frac = avail / max(window_baseline, 1)
        readiscore = alpha*skillfit + (1 - alpha)*avail_frac

        results.append({
            "emp_id": c["id"],
            "skillfit": round(skillfit, 4),
            "hours": int(avail),
            "readiscore": round(readiscore, 4),
        })

# ---------- Render results ----------
results = sorted(results, key=lambda r: r["readiscore"], reverse=True)
df = pd.DataFrame([{
    "Rank": i+1,
    "Candidate": r["emp_id"],
    "ReadiScore %": int(r["readiscore"]*100),
    "SkillFit %": int(r["skillfit"]*100),
    "Avail. hrs": r["hours"],
} for i, r in enumerate(results)])
st.dataframe(df, use_container_width=True)
st.caption(f"α blends SkillFit with Availability. Availability window and workday mask set on the previous page.")

# ---------- PDF + Return ----------
params = {
    "start_date": str(st.session_state.get("start_date")),
    "end_date": str(st.session_state.get("end_date")),
    "workdays": st.session_state.get("workdays", ["Mon","Tue","Wed","Thu","Fri"]),
    "max_hours": st.session_state.get("max_hours", 8),
    "alpha": float(st.session_state.get("alpha", 0.7)),
}
st.download_button("Download PDF report", data=build_pdf(results, params),
                   file_name="teamreadi_results.pdf", mime="application/pdf")

def _reset_and_return():
    for k in ("resumes","req_files","req_url","cal_method","cal_link","random_target",
              "cal_upload","start_date","end_date","workdays","max_hours","alpha"):
        if k in st.session_state: del st.session_state[k]
    st.switch_page("streamlit_app.py")

st.button("Return to Start", on_click=_reset_and_return)
