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
LIGHT  = "#f6f7fb"

st.markdown(f"""
<style>
/* Page spacing */
.block-container {{ padding-top: 1.25rem !important; }}

/* Keep columns equal and tidy on desktop */
@media (min-width: 992px) {{
  [data-testid="column"] > div:first-child {{ 
    max-width: 720px;    /* cap each column width */
    margin: 0 auto; 
  }}
}}

/* Banner image: fit screen height, not too tall */
[data-testid="stImage"] img {{
  max-height: 82vh;     /* never exceed viewport height */
  width: auto;
  height: auto;
  object-fit: contain;
  border-radius: 12px;
}}

/* Form card look (no gradient) */
.form-card {{
  border: 2px solid {NAVY};
  border-radius: 16px;
  padding: 1.25rem;
  background: #fff;
  box-shadow: 0 2px 12px rgba(0,0,0,.06);
}}

/* Section headers like your original */
.form-header {{
  background: {NAVY};
  color: #fff;
  font-weight: 700;
  font-size: 1.05rem;
  padding: .6rem 1rem;
  border-radius: 10px;
  margin: .25rem 0 .9rem 0;
}}

/* Inputs: softer, rounded */
.stTextInput > div > div > input,
.stNumberInput input,
.stDateInput input {{
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  height: 44px;
}}

/* File uploader: dashed light card with orange focus */
.stFileUploader > label div[data-testid="stFileUploaderDropzone"] {{
  border: 2px dashed rgba(11,39,73,.18);
  background: {LIGHT};
  border-radius: 14px;
}}
.stFileUploader > label div[data-testid="stFileUploaderDropzone"]:hover {{
  border-color: {ORANGE};
}}

/* Radio "pills" */
.stRadio > div > label {{
  border: 1.5px solid #e5e7eb;
  border-radius: 999px;
  padding: .35rem .7rem;
  margin-right: .4rem;
}}
.stRadio > div > label:hover {{
  border-color: {ORANGE};
}}
.stRadio input:checked + div + label, 
.stRadio input:checked + label {{
  border-color: {ORANGE};
  box-shadow: 0 0 0 2px rgba(248,156,28,.15) inset;
}}

/* CTA button in orange */
div.stButton > button {{
  width: 100%;
  height: 48px;
  border-radius: 999px;
  font-weight: 700;
  border: 2px solid {ORANGE};
  color: #fff;
  background: {ORANGE};
}}
div.stButton > button:hover {{
  filter: brightness(.95);
}}

[data-testid="stVerticalBlock"] {{ align-items: flex-start; }}
</style>
""", unsafe_allow_html=True)

# ------------ layout: banner + form ------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    # scale automatically but never exceed ~viewport height (CSS above)
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
