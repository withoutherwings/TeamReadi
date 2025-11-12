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

st.set_page_config(page_title="TeamReadi", page_icon="✅", layout="wide")

# ------------ styles ------------
NAVY   = "#0b2749"
ORANGE = "#f89c1c"
LIGHT  = "#f8fafc"

st.markdown(f"""
<style>
.block-container {{
  padding-top: 1.5rem !important;
  max-width: 1500px;
}}

[data-testid="stHorizontalBlock"] > div:first-child,
[data-testid="column"] > div:first-child {{
  display: flex;
  justify-content: center;
  align-items: center;
}}

[data-testid="stImage"] img {{
  max-height: 82vh;
  width: auto;
  object-fit: contain;
  border-radius: 12px;
}}

/* Form container styling */
.form-card {{
  border: 3px solid {NAVY};
  border-radius: 18px;
  background: #fff;
  padding: 2rem;
  box-shadow: 0 3px 10px rgba(11,39,73,0.1);
  max-width: 720px;
  margin: auto;
}}

/* Section headers */
.form-header {{
  background: {NAVY};
  color: #fff;
  font-weight: 700;
  font-size: 1.1rem;
  padding: .65rem 1rem;
  border-radius: 10px;
  margin-bottom: .8rem;
}}

/* File upload area */
.stFileUploader > label div[data-testid="stFileUploaderDropzone"] {{
  border: 2px dashed rgba(11,39,73,.2);
  background: {LIGHT};
  border-radius: 12px;
}}
.stFileUploader > label div[data-testid="stFileUploaderDropzone"]:hover {{
  border-color: {ORANGE};
  background: #fff8f0;
}}

/* Inputs and date fields */
.stTextInput input, .stNumberInput input, .stDateInput input {{
  border-radius: 8px;
  border: 1.5px solid #e5e7eb;
  height: 42px;
}}

/* Radio buttons as pills */
.stRadio > div > label {{
  border: 1.5px solid #dcdfe4;
  border-radius: 999px;
  padding: .3rem .9rem;
  margin-right: .4rem;
  font-weight: 500;
  color: {NAVY};
}}
.stRadio > div > label:hover {{
  border-color: {ORANGE};
  background: #fff8f0;
}}
.stRadio input:checked + div + label, 
.stRadio input:checked + label {{
  border-color: {ORANGE};
  background: #fff8f0;
  color: {NAVY};
  font-weight: 600;
}}

/* CTA Button */
div.stButton > button {{
  width: 100%;
  height: 50px;
  border-radius: 999px;
  background: {ORANGE};
  color: white;
  font-weight: 700;
  border: 2px solid {ORANGE};
  box-shadow: 0 2px 6px rgba(248,156,28,.25);
}}
div.stButton > button:hover {{
  background: #e88c0f;
  border-color: #e88c0f;
}}

</style>
""", unsafe_allow_html=True)

# ------------ layout: banner + form ------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.image("TeamReadi Side Banner.png", use_column_width=True)

with col2:
    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    st.markdown('<div class="form-header">Upload Resumes</div>', unsafe_allow_html=True)
    resumes = st.file_uploader(
        "Resumes (PDF/DOC/DOCX/TXT)",
        type=["pdf","doc","docx","txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.markdown('<div class="form-header">Project Requirements</div>', unsafe_allow_html=True)
    req_files = st.file_uploader(
        "RFP / Job criteria (PDF/DOC/DOCX/TXT/MD)",
        type=["pdf","doc","docx","txt","md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="req_files"
    )
    req_url = st.text_input("Or paste job/RFP URL here")

    st.markdown('<div class="form-header">Import Calendar</div>', unsafe_allow_html=True)
    cal_method = st.radio("Method", ["Calendar Link","Manual Entry / Upload","Randomize Hours"], horizontal=True)
    cal_link = cal_upload = None
    random_target = None
    if cal_method == "Calendar Link":
        cal_link = st.text_input("Paste calendar link (Google, Outlook, etc.)", placeholder="https://…")
    elif cal_method == "Manual Entry / Upload":
        cal_upload = st.file_uploader("Upload .ics / CSV / spreadsheet", type=["ics","csv","xls","xlsx"], key="cal_csv")
    else:
        random_target = st.slider("Average utilization target (%)", 10, 100, 60)

    st.markdown('<div class="form-header">Availability Parameters</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start Date", value=dt.date.today())
    with c2:
        end_date = st.date_input("End Date", value=dt.date.today())
    workdays = st.multiselect("Working Days", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                              default=["Mon","Tue","Wed","Thu","Fri"])
    max_hours = st.number_input("Maximum work hours per day", min_value=1, max_value=12, value=8, step=1)

    if st.button("Get Readi!"):
        _stash_files("resumes", resumes)
        _stash_files("req_files", req_files)
        st.session_state.update({
            "req_url": req_url or "",
            "cal_method": cal_method,
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
