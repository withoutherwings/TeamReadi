# app.py — TeamReadi Landing (collects inputs, then routes to Results)
import base64, datetime as dt
import streamlit as st

import os
from openai import OpenAI

# --- fixed weighting (no slider anywhere) ---
SKILL_WEIGHT = 0.70  # 70% skills, 30% availability

# Load your API key automatically (Streamlit Secrets preferred; env as fallback)
API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.set_page_config(page_title="TeamReadi", page_icon="✅", layout="centered")
    st.error("OPENAI_API_KEY is not set. Add it in Streamlit Secrets or as an environment variable.")
    st.stop()

client = OpenAI(api_key=API_KEY)
EMBED_MODEL = "text-embedding-3-large"

# ---- final layout: left banner (no border), right form (bordered) ----
col1, col2 = st.columns([1,1], gap="large")

with col1:
    st.image("TeamReadi Side Banner.png", use_column_width=True)  # no frame, no CSS

with col2:
    st.markdown("""
    <style>
    :root{ --navy:#001f3f; --orange:#ff7a00; --panel-h: clamp(560px,80vh,900px); }
    .form-card{
        background:#fff; border:3px solid var(--navy); border-radius:20px;
        height:var(--panel-h); box-shadow:0 4px 20px rgba(0,0,0,.10);
        display:flex; flex-direction:column; overflow:hidden;
    }
    .form-header{ background:linear-gradient(135deg,#123e78, #1e5799 30%, #ed9a3f 100%);
        color:#fff; padding:18px 22px 22px 22px; }
    .form-title{ font-size:22px; font-weight:750; margin:0 0 4px 0; }
    .form-sub{ opacity:.9; font-size:13px; margin:0 0 10px 0; }
    .steps{ display:flex; gap:26px; align-items:center; font-size:12px; }
    .step{ display:flex; gap:10px; align-items:center; color:#e9eef6; }
    .bullet{ width:22px; height:22px; border-radius:999px; background:rgba(255,255,255,.18);
             display:grid; place-items:center; font-weight:700; }
    .step.active .bullet{ background:#fff; color:#1e457a; } .step.active{ color:#fff; }
    .form-body{ padding:18px 22px 22px 22px; overflow:auto; }
    .dropzone{ border:2px dashed #d9dee7; border-radius:12px; padding:14px;
               background:linear-gradient(180deg,#fff 0%,#fbfcfe 100%); }
    div[role="radiogroup"] label{ border:2px solid #e6e6e6; border-radius:10px; padding:8px 10px; margin-right:10px; }
    div[role="radiogroup"] label:has(input:checked){ border-color:var(--orange); box-shadow:0 0 0 2px rgba(255,122,0,.12); }
    .stTextInput input, .stDateInput input, .stTextArea textarea{ border:1px solid #cfcfcf; border-radius:10px; }
    .stButton>button{ background:var(--orange) !important; color:#fff !important; border:none !important;
                      border-radius:12px !important; font-weight:750 !important; width:100%; height:3rem; }
    .stButton>button:hover{ background:#e66a00 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="form-header">
      <div class="form-title">Generate New Report</div>
      <div class="form-sub">Upload resumes, import calendars, set availability, and generate a demo in minutes.</div>
      <div class="steps">
        <div class="step active"><div class="bullet">1</div><div>Resumes</div></div>
        <div class="step"><div class="bullet">2</div><div>Calendar</div></div>
        <div class="step"><div class="bullet">3</div><div>Review</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="form-body">', unsafe_allow_html=True)

    # Upload Resumes
    st.markdown('<div class="dropzone">', unsafe_allow_html=True)
    resumes = st.file_uploader("Drag & drop files here, or Browse",
        type=["pdf","doc","docx","txt"], accept_multiple_files=True, label_visibility="collapsed")
    st.caption("PDF, DOC, DOCX")
    st.markdown('</div>', unsafe_allow_html=True)

    # Project Requirements (upload + URL)
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="dropzone">', unsafe_allow_html=True)
    req_files = st.file_uploader("Drag & drop project requirements here, or Browse",
        type=["pdf","doc","docx","txt","md"], accept_multiple_files=True, key="req_files",
        label_visibility="collapsed")
    st.caption("Limit 200MB per file • PDF, DOC, DOCX, TXT, MD")
    st.markdown('</div>', unsafe_allow_html=True)
    req_url = st.text_input("Paste job/RFP URL here (optional)")

    # Calendar + Availability (unchanged from your current version)…
    calendar_method = st.radio("", ["Calendar Link","Manual Entry / Upload","Randomize Hours"],
                               index=0, horizontal=True, label_visibility="collapsed")
    cal_link = cal_upload = None; random_target = None
    if calendar_method == "Calendar Link":
        cal_link = st.text_input("Paste calendar link here (Google, Outlook, etc.)")
    elif calendar_method == "Manual Entry / Upload":
        cal_upload = st.file_uploader("Upload .ics / CSV / spreadsheet", type=["ics","csv","xls","xlsx"], key="cal_csv")
    else:
        random_target = st.slider("Average utilization target (%)", 10, 100, 60)

    c1, c2 = st.columns(2)
    with c1:  start_date = st.date_input("Start Date")
    with c2:  end_date   = st.date_input("End Date")
    workdays = st.multiselect("Working Days", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                              default=["Mon","Tue","Wed","Thu","Fri"])
    max_hours = st.number_input("Maximum work hours per day", 1, 12, 8, 1)

    if st.button("Get Readi!", use_container_width=True):
        _stash_files("resumes", resumes); _stash_files("req_files", req_files)
        st.session_state.update({
            "req_url": req_url or "", "cal_method": calendar_method, "cal_link": cal_link or "",
            "random_target": random_target,
            "cal_upload": None if cal_upload is None else {"name": cal_upload.name, "data": cal_upload.getvalue()},
            "start_date": str(start_date), "end_date": str(end_date),
            "workdays": workdays, "max_hours": int(max_hours), "alpha": float(SKILL_WEIGHT),
        })
        st.switch_page("pages/01_Results.py")

    st.markdown('</div>', unsafe_allow_html=True)  # end form-body
    st.markdown('</div>', unsafe_allow_html=True)  # end form-card


