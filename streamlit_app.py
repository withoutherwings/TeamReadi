# app.py — TeamReadi Landing (complete replacement)

import os, json, base64, datetime as dt
import streamlit as st
from openai import OpenAI

# TeamReadi backend imports
from backend.pipeline import run_teamreadi_pipeline
from backend.calendar_backend import llm_explain_employee

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="TeamReadi", page_icon="🕒", layout="wide")

# ----- Fixed weighting (used later in scoring logic) -----
SKILL_WEIGHT = 0.70

# ----- OpenAI setup -----
API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = st.secrets.get("MODEL_NAME", "gpt-4o")
EMBED_MODEL = st.secrets.get("EMBED_MODEL", "text-embedding-3-large")

if not API_KEY:
    st.error("OPENAI_API_KEY is not set.")
    st.stop()

client = OpenAI(api_key=API_KEY)

# ---------------- STYLING ----------------
st.markdown(
    """
<style>
  /* Page + container */
  .main {
    background:#ffffff;
  }
  .block-container {
    padding-top:1.2rem;
    padding-bottom:2rem;
    max-width:1200px;
  }
  header[data-testid="stHeader"] { background:transparent; }

  /* Unified report card */
  .report-card {
    background:#FFFFFF;
    border-radius:18px;
    box-shadow:0 12px 32px rgba(16,35,61,0.15);
    border:1px solid rgba(16,35,61,0.10);
    overflow:hidden;
  }
  .report-header {
    background:#0F233D;          /* solid navy */
    color:#ffffff;
    padding:18px 24px;
    font-weight:750;
    font-size:1.1rem;
    letter-spacing:.02em;
  }
  .report-body {
    padding:18px 24px 22px 24px;
    background:#F9FAFB;
  }

  /* Section titles */
  .sec-title {
    color:#10233D;
    font-weight:700;
    margin:6px 0 4px;
    font-size:0.98rem;
  }
  .sec-subtitle {
    font-size:0.82rem;
    color:#64748B;
  }

  /* File uploaders: keep card style but not too tall */
  .stFileUploader > div {
    border:1px dashed rgba(15,35,61,0.26) !important;
    border-radius:12px !important;
    padding-top:0.6rem !important;
    padding-bottom:0.6rem !important;
    background:#FFFFFF !important;
  }

  /* Text / date / number inputs: always visible borders */
  .stTextInput input, .stDateInput input {
    border-radius:10px !important;
    border:1px solid #D0D7E2 !important;
    background:#FFFFFF !important;
  }
  .stNumberInput input {
    border-radius:10px !important;
    border:1px solid #D0D7E2 !important;
    background:#FFFFFF !important;
    text-align:center;
  }
  .stTextInput input:focus, .stDateInput input:focus, .stNumberInput input:focus {
    outline:none !important;
    border:1px solid #FF8A1E !important;
    box-shadow:0 0 0 1px rgba(255,138,30,0.25) !important;
  }

  /* Radio / checkbox labels */
  .stRadio > label, .stCheckbox > label {
    color:#10233D;
  }

  /* Thin divider lines inside card */
  .report-divider {
    border-top:1px solid #E2E8F0;
    margin:14px 0 16px 0;
  }

  /* Centered CTA row */
  .cta-row {
    display:flex;
    justify-content:center;
    margin-top:1.4rem;
  }

  /* Orange Get Readi button (form submit) */
  .cta-row button[data-testid="baseButton-primaryFormSubmit"],
  .cta-row button[kind="formSubmit"] {
      background:#FF8A1E !important;
      color:#ffffff !important;
      border-radius:12px !important;
      padding:10px 26px !important;
      border:0 !important;
      font-weight:800 !important;
      letter-spacing:.03em !important;
      box-shadow:0 3px 0 rgba(0,0,0,0.10) !important;
      width:auto !important;
      min-width:150px;
  }
  .cta-row button[data-testid="baseButton-primaryFormSubmit"]:hover,
  .cta-row button[kind="formSubmit"]:hover {
      background:#F27B00 !important;
  }

  /* Make hours input compact and centered under its label */
  .hours-row {
    display:flex;
    justify-content:center;
    margin-top:4px;
  }
  .hours-row > div {
    max-width:220px;
    width:100%;
  }

</style>
""",
    unsafe_allow_html=True,
)

# ---------------- LAYOUT ----------------
# 1/3 banner, 2/3 form
left, right = st.columns([1, 2], gap="large")

# ---- LEFT: marketing banner ----
with left:
    st.image("TeamReadi Side Banner.png", use_container_width=True)

# ---- RIGHT: unified card (header + form) ----
with right:
    st.markdown("<div class='report-card'>", unsafe_allow_html=True)
    st.markdown("<div class='report-header'>Generate New Report</div>", unsafe_allow_html=True)
    st.markdown("<div class='report-body'>", unsafe_allow_html=True)

    # ---------- SINGLE FORM ----------
    with st.form("generate_form", clear_on_submit=False):

        # --- Upload Resumes ---
        st.markdown("<div class='sec-title'>Upload Resumes</div>", unsafe_allow_html=True)
        resumes = st.file_uploader(
            "Drag & drop files here, or browse",
            type=["pdf", "doc", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        st.caption("PDF, DOC, DOCX, TXT")

        # --- Project Requirements ---
        st.markdown("<div class='sec-title' style='margin-top:10px;'>Project Requirements</div>", unsafe_allow_html=True)
        proj = st.file_uploader(
            "Drag & drop files here, or browse",
            type=["pdf", "doc", "docx", "txt", "md"],
            accept_multiple_files=True,
            key="proj",
            label_visibility="collapsed",
        )
        req_url = st.text_input("Or paste job / RFP URL", placeholder="https://…")

        st.markdown("<div class='report-divider'></div>", unsafe_allow_html=True)

        # --- Import Calendar ---
        st.markdown("<div class='sec-title'>Import Calendar</div>", unsafe_allow_html=True)
        cal_mode = st.radio(
            "Calendar import mode",
            options=["Calendar link", "Randomize hours (demo mode)"],
            horizontal=True,
            index=0,
            label_visibility="collapsed",
        )

        cal_url = ""
        randomize_seed = None

        if cal_mode.startswith("Calendar link"):
            cal_url = st.text_input(
                "Paste a public/shared calendar URL (Google, Outlook, etc.)",
                placeholder="https://calendar.google.com/calendar/ical/…"
            )
        else:
            randomize_seed = st.slider("Average utilization target (%)", 10, 100, 60)

        st.markdown("<div class='report-divider'></div>", unsafe_allow_html=True)

        # --- Availability Parameters ---
        st.markdown("<div class='sec-title'>Availability Parameters</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("Start date", value=dt.date.today())
        with c2:
            end_date = st.date_input("End date", value=dt.date.today() + dt.timedelta(days=30))

        st.markdown("<div class='sec-title' style='margin-top:10px;'>Working days</div>", unsafe_allow_html=True)
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        defaults = [True, True, True, True, True, False, False]
        dcols = st.columns(7)
        workdays_checks = []
        for i, col in enumerate(dcols):
            with col:
                workdays_checks.append(st.checkbox(day_labels[i], value=defaults[i], key=f"d{i}"))
        selected_days = [d for d, keep in zip(day_labels, workdays_checks) if keep]

        # Maximum hours: label immediately above compact, centered input
        st.markdown("<div class='sec-title' style='margin-top:12px;'>Maximum work hours per day</div>", unsafe_allow_html=True)
        st.markdown("<div class='hours-row'>", unsafe_allow_html=True)
        max_daily = st.number_input(
            "",
            min_value=1.0, max_value=12.0,
            value=8.0, step=1.0,
            key="max_daily_hours",
            label_visibility="collapsed"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Centered orange CTA ----
        st.markdown("<div class='cta-row'>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Get Readi!")
        st.markdown("</div>", unsafe_allow_html=True)

    # close report-body + report-card divs
    st.markdown("</div></div>", unsafe_allow_html=True)

# ---------------- HANDLE SUBMIT ----------------
if submitted:
    # Save inputs to session for the Results page
    st.session_state["resumes"] = [
        {"name": f.name, "data": f.getvalue()} for f in (resumes or [])
    ]
    st.session_state["req_files"] = [
        {"name": f.name, "data": f.getvalue()} for f in (proj or [])
    ]
    st.session_state.update(
        {
            "req_url": req_url or "",
            "cal_method": cal_mode,
            "cal_link": cal_url or "",
            "random_target": randomize_seed,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "workdays": selected_days,
            "max_hours": float(max_daily),
            "alpha": SKILL_WEIGHT,  # can be swapped to a slider later
        }
    )
    # Navigate to results page (expects pages/01_Results.py)
    st.switch_page("pages/01_Results.py")

# Optional debug panel
with st.expander("Debug: current form payload"):
    st.write(
        {
            "resumes": [f["name"] for f in st.session_state.get("resumes", [])],
            "project_files": [f["name"] for f in st.session_state.get("req_files", [])],
            "cal_method": st.session_state.get("cal_method"),
            "cal_link": st.session_state.get("cal_link"),
            "random_target": st.session_state.get("random_target"),
            "start_date": st.session_state.get("start_date"),
            "end_date": st.session_state.get("end_date"),
            "workdays": st.session_state.get("workdays"),
            "max_hours": st.session_state.get("max_hours"),
        }
    )
