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

# ------------ layout: banner + ONE uniform report form ------------
PANEL_H = 800  # 👈 make this match your banner's visible height (px)

st.markdown(f"""
<style>
:root {{
  --navy: #001f3f;
  --orange: #ff7a00;
  --panel-h: {PANEL_H}px;
}}
.form-card {{
    background: linear-gradient(180deg, #ffffff 0%, #f9fafc 100%);
    border: 3px solid var(--navy);
    border-radius: 20px;
    padding: 2rem;
    width: 100%;
    max-width: 560px;
    min-height: var(--panel-h);
    box-shadow: 0 4px 20px rgba(0,0,0,0.10);
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}}
.form-card h3 {{
    margin: 0 0 .25rem 0;
    text-align: center;
    color: var(--navy);
}}
/* Inputs */
.form-card .stTextInput input,
.form-card .stDateInput input,
.form-card .stSelectbox div[data-baseweb="select"] > div,
.form-card .stTextArea textarea {{
    border: 1px solid #cfcfcf;
    border-radius: 10px;
}}
/* Radio: orange outline highlight style */
div[role="radiogroup"] label {{
    border: 2px solid #e6e6e6;
    border-radius: 10px;
    padding: 8px 10px;
    margin-right: 10px;
}}
div[role="radiogroup"] input:checked + div + label,
div[role="radiogroup"] label:has(input:checked) {{
    border-color: var(--orange);
    box-shadow: 0 0 0 2px rgba(255,122,0,.12);
}}
/* Button */
.stButton>button {{
    background-color: var(--orange) !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: 700 !important;
    width: 100%;
    height: 3rem;
}}
.stButton>button:hover {{ background-color: #e66a00 !important; }}
</style>
""", unsafe_allow_html=True)

# Side-by-side
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    # Left banner (shown as-is). Adjust PANEL_H above so the right panel matches it.
    st.image("TeamReadi Side Banner.png", use_column_width=True)

with col2:
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
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

    st.markdown("<div style='margin-top: .5rem'></div>", unsafe_allow_html=True)
    if st.button("Get Readi!", use_container_width=True):
        # assumes you have _stash_files defined elsewhere in your app
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
