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

# ------------ layout: banner + ONE uniform report form (equal height) ------------

st.set_page_config(page_title="TeamReadi", page_icon="✅", layout="wide")

# helper to embed image as base64 so we can fully control CSS sizing
def _img_b64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

BANNER_PATH = "TeamReadi Side Banner.png"   # <- your left graphic
banner_b64 = _img_b64(BANNER_PATH)

st.markdown("""
<style>
:root{
  --navy:#001f3f;
  --orange:#ff7a00;
  /* Equal height for both columns: scales with screen, capped for large monitors */
  --panel-h: clamp(560px, 80vh, 900px);
}

/* Shared framed box (used by banner + form) */
.frame{
  border:3px solid var(--navy);
  border-radius:20px;
  background:#fff;
  height:var(--panel-h);
  width:100%;
  box-shadow:0 4px 20px rgba(0,0,0,.10);
  overflow:hidden;              /* keep content inside rounded corners */
}

/* Banner image must scale INSIDE the frame */
.frame .banner{
  width:100%;
  height:100%;
  object-fit:contain;           /* keep proportions, no cropping, no stretch */
  display:block;
}

/* Form panel styling */
.form-card{
  background:linear-gradient(180deg,#ffffff 0%,#f9fafc 100%);
  height:var(--panel-h);        /* exact same height as banner */
  border:3px solid var(--navy);
  border-radius:20px;
  padding:2rem;
  box-shadow:0 4px 20px rgba(0,0,0,.10);
  display:flex;
  flex-direction:column;
  gap:.75rem;
}

/* Inputs */
.form-card .stTextInput input,
.form-card .stDateInput input,
.form-card .stTextArea textarea{
  border:1px solid #cfcfcf;
  border-radius:10px;
}

/* Radio: orange outline highlight */
div[role="radiogroup"] label{
  border:2px solid #e6e6e6;
  border-radius:10px;
  padding:8px 10px;
  margin-right:10px;
}
div[role="radiogroup"] label:has(input:checked){
  border-color:var(--orange);
  box-shadow:0 0 0 2px rgba(255,122,0,.12);
}

/* Button */
.stButton>button{
  background-color:var(--orange) !important;
  color:#fff !important;
  border:none !important;
  border-radius:12px !important;
  font-weight:700 !important;
  width:100%;
  height:3rem;
}
.stButton>button:hover{ background-color:#e66a00 !important; }
</style>
""", unsafe_allow_html=True)

# Two columns — equal height via CSS variables above
col1, col2 = st.columns([1,1], gap="large")

with col1:
    # Use HTML <img> so CSS can set object-fit/height exactly
    if banner_b64:
        st.markdown(
            f"<div class='frame'><img class='banner' src='data:image/png;base64,{banner_b64}' alt='TeamReadi banner'/></div>",
            unsafe_allow_html=True
        )
    else:
        # fallback if file missing
        st.markdown("<div class='frame'></div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='form-card'>", unsafe_allow_html=True)
    st.markdown("### Generate New Report")

    # Upload Resumes
    resumes = st.file_uploader(
        "Upload Resumes",
        type=["pdf","doc","docx","txt"],
        accept_multiple_files=True,
        help="Drag & drop files or browse"
    )

    # Project Requirements
    req_files = st.file_uploader(
        "Upload Project Requirements",
        type=["pdf","doc","docx","txt","md"],
        accept_multiple_files=True,
        key="req_files",
        help="RFP / job criteria"
    )
    req_url = st.text_input("Or paste job/RFP URL here")

    # Calendar
    calendar_method = st.radio(
        "Import Calendar",
        ["Calendar Link", "Manual Entry / Upload", "Randomize Hours"],
        horizontal=True
    )
    cal_link = cal_upload = None
    random_target = None
    if calendar_method == "Calendar Link":
        cal_link = st.text_input("Paste calendar link (Google, Outlook, etc.)", placeholder="https://…")
    elif calendar_method == "Manual Entry / Upload":
        cal_upload = st.file_uploader("Upload .ics / CSV / spreadsheet", type=["ics","csv","xls","xlsx"], key="cal_csv")
    else:
        random_target = st.slider("Average utilization target (%)", 10, 100, 60)

    # Availability
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start Date", value=dt.date.today())
    with c2:
        end_date = st.date_input("End Date", value=dt.date.today())

    workdays = st.multiselect(
        "Working Days",
        ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        default=["Mon","Tue","Wed","Thu","Fri"]
    )
    max_hours = st.number_input("Maximum work hours per day", min_value=1, max_value=12, value=8, step=1)

    st.markdown("<div style='margin-top:.5rem'></div>", unsafe_allow_html=True)
    if st.button("Get Readi!", use_container_width=True):
        # assumes _stash_files exists elsewhere in your code
        _stash_files("resumes", resumes)
        _stash_files("req_files", req_files)
        st.session_state.update({
            "req_url": req_url or "",
            "cal_method": calendar_method,
            "cal_link": cal_link or "",
            "random_target": random_target,
            "cal_upload": None if cal_upload is None else {"name": cal_upload.name, "data": cal_upload.getvalue()},
            "start_date": str(start_date),
            "end_date": str(end_date),
            "workdays": workdays,
            "max_hours": int(max_hours),
            "alpha": float(SKILL_WEIGHT),
        })
        st.switch_page("pages/01_Results.py")

    st.markdown("</div>", unsafe_allow_html=True)
