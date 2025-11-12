# app.py — TeamReadi Landing (collects inputs, then routes to Results)
import base64, datetime as dt
import streamlit as st

import os
from openai import OpenAI

# app.py — TeamReadi "Generate New Report" UI clone
import datetime as dt
import streamlit as st

st.set_page_config(page_title="TeamReadi", page_icon="🕒", layout="wide")

# ---- THEME COLORS (tweak here if needed) ----
NAVY = "#10233D"       # deep navy
NAVY_600 = "#1A2F4C"
NAVY_700 = "#0B1F3A"
ORANGE = "#FF8A1E"     # primary orange
ORANGE_600 = "#F27B00"
MUTED = "#98A2B3"      # gray for help text
BG = "#F7F8FA"         # soft page bg
CARD_BG = "#FFFFFF"

# ---- GLOBAL CSS ----
st.markdown(
    f"""
    <style>
      .main {{ background:{BG}; }}
      /* remove default padding so our card sits tight */
      .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1180px; }}
      /* two-column heights align */
      .equal-col > div {{ height: 100%; }}

      /* RIGHT CARD (form) */
      .card {{
        background: {CARD_BG};
        border-radius: 14px;
        box-shadow: 0 6px 24px rgba(16, 35, 61, 0.12);
        border: 1px solid rgba(16, 35, 61, 0.06);
        overflow: hidden;
      }}
      .card-header {{
        background: linear-gradient(135deg, {NAVY_700} 0%, {ORANGE} 100%);
        color: white;
        padding: 18px 22px;
        font-weight: 700;
        letter-spacing: .2px;
        font-size: 1.05rem;
      }}
      .card-body {{ padding: 18px; }}

      /* section titles */
      .sec-title {{
        color: {NAVY};
        font-weight: 700;
        margin: 8px 0 6px;
      }}
      .hint {{ color: {MUTED}; font-size: 0.85rem; margin-top: -2px; }}

      /* uploader boxes look like dashed areas in the mock */
      .stFileUploader > div {{ border: 1px dashed rgba(16,35,61,.25) !important; border-radius: 10px; }}
      .stFileUploadDropzone {{
        background: #FBFCFE !important;
      }}

      /* radio/checkbox paddings */
      .stRadio > label, .stCheckbox > label {{ color: {NAVY}; }}
      .days-row .stCheckbox {{ margin-right: .6rem; }}

      /* button */
      div.stButton > button {{
        width: 100%;
        padding: 12px 16px;
        border-radius: 10px;
        border: 0;
        background: {ORANGE};
        color: #fff;
        font-weight: 800;
        letter-spacing: .2px;
      }}
      div.stButton > button:hover {{ background: {ORANGE_600}; }}

      /* inputs */
      .stTextInput input, .stNumberInput input, .stDateInput input {{
        border-radius: 10px !important;
      }}

      /* remove random top "box" spacing users sometimes see */
      header[data-testid="stHeader"] {{ background: transparent; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- LAYOUT: LEFT BANNER | RIGHT FORM ----
left, right = st.columns([0.9, 1.3], gap="large")

with left:
    st.markdown("<div class='equal-col'>", unsafe_allow_html=True)
    st.image("TeamReadi Side Banner.png", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>Generate New Report</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-body'>", unsafe_allow_html=True)

    with st.form("generate_form", clear_on_submit=False):
        # --- Upload Resumes ---
        st.markdown("<div class='sec-title'>Upload Resumes</div>", unsafe_allow_html=True)
        resumes = st.file_uploader(
            "Drag & drop files here, or browse",
            type=["pdf", "doc", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        st.markdown("<div class='hint'>PDF, DOC, DOCX</div>", unsafe_allow_html=True)

        # --- Upload Project Details ---
        st.markdown("<div class='sec-title' style='margin-top:10px;'>Upload project details</div>", unsafe_allow_html=True)
        proj = st.file_uploader(
            "Drag & drop files here, or browse",
            type=["pdf", "doc", "docx", "csv", "xlsx"],
            accept_multiple_files=True,
            key="proj",
            label_visibility="collapsed",
        )
        st.markdown("<div class='hint'>PDF, DOC, DOCX, CSV, XLSX</div>", unsafe_allow_html=True)

        st.markdown("---")

        # --- Import Calendar ---
        st.markdown("<div class='sec-title'>Import Calendar</div>", unsafe_allow_html=True)
        cal_mode = st.radio(
            "Calendar import mode",
            options=["Calendar link", "Manual entry / upload", "Randomize hours"],
            horizontal=True,
            label_visibility="collapsed",
        )

        cal_url = None
        cal_file = None
        randomize_seed = None

        if cal_mode == "Calendar link":
            cal_url = st.text_input("Paste a public or shared calendar URL (Google, Outlook, etc.)", placeholder="https://...")
        elif cal_mode == "Manual entry / upload":
            cal_file = st.file_uploader(
                "Upload CSV or spreadsheet (columns: employee_id, start, end, hours)", type=["csv", "xlsx"], key="cal_csv"
            )
            st.markdown("<div class='hint'>CSV or spreadsheet import</div>", unsafe_allow_html=True)
        else:
            randomize_seed = st.number_input("Great for demos: seed", min_value=0, value=42, step=1)

        st.markdown("---")

        # --- Availability Parameters ---
        st.markdown("<div class='sec-title'>Availability parameters</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("Start date", value=dt.date.today())
        with c2:
            end_date = st.date_input("End date", value=dt.date.today() + dt.timedelta(days=30))

        # Working days
        st.markdown("<div class='sec-title' style='margin-top:8px;'>Working days</div>", unsafe_allow_html=True)
        dcols = st.columns(7)
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        defaults = [True, True, True, True, True, False, False]
        workdays = []
        for i, col in enumerate(dcols):
            with col:
                workdays.append(st.checkbox(day_labels[i], value=defaults[i], key=f"d{i}"))

        # Max hours/day
        max_daily = st.number_input("Maximum work hours per day", min_value=1.0, max_value=24.0, value=8.0, step=0.5)
        st.markdown("<div class='hint'>Defaults to 8; narrow input (1–12) if needed.</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Get Readi!")

    st.markdown("</div></div>", unsafe_allow_html=True)

# ---- BACKEND STUB (all inputs operable; wire to your logic here) ----
if 'results' not in st.session_state:
    st.session_state.results = None

if submitted:
    st.success("Inputs captured. (This is where your scoring + scheduling logic runs.)")
    st.session_state.results = {
        "resume_files": [f.name for f in resumes] if resumes else [],
        "project_files": [f.name for f in proj] if proj else [],
        "calendar_mode": cal_mode,
        "calendar_url": cal_url,
        "calendar_file": (cal_file.name if cal_file else None),
        "random_seed": randomize_seed,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "workdays": [d for d, keep in zip(day_labels, workdays) if keep],
        "max_hours_per_day": float(max_daily),
    }

# Optional: echo the captured state (hide in production)
with st.expander("Debug: current form payload"):
    st.write(st.session_state.results)

