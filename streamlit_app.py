# app.py — TeamReadi Landing

import os, json, base64, datetime as dt
import streamlit as st
from openai import OpenAI

# TeamReadi backend imports
from backend.pipeline import run_teamreadi_pipeline
from backend.calendar_backend import llm_explain_employee

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="TeamReadi", page_icon="🕒", layout="wide")

SKILL_WEIGHT = 0.70

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
  /* Make whole app white */
  body, .stApp, .main {
      background:#ffffff !important;
  }
  .block-container {
      padding-top:0.8rem;
      padding-bottom:1.4rem;
      max-width:1200px;
  }

  /* Just align the row, don't clamp its height */
  [data-testid="stHorizontalBlock"] {
      align-items:flex-start;
  }

  /* --- LEFT COLUMN (banner) --- */
  [data-testid="column"]:first-child {
      display:flex;
      align-items:stretch;
  }
  [data-testid="column"]:first-child > div {
      flex:1;
      display:flex;
      align-items:center;
      justify-content:center;
  }
  [data-testid="column"]:first-child [data-testid="stImage"] img {
      width:100%;
      height:auto;
      object-fit:contain;
  }

  /* --- RIGHT COLUMN (form card) --- */
  [data-testid="stForm"] {
      background:#FFFFFF !important;
      border-radius:24px !important;
      border:1px solid rgba(16,35,61,.12) !important;
      box-shadow:0 12px 30px rgba(16,35,61,.10) !important;
      padding:0 !important;
  }
  [data-testid="stForm"] > div {
      padding:0 !important;
      background:transparent !important;
  }

  .card-header {
      background:#0F243D;
      color:#ffffff;
      padding:16px 26px;
      font-weight:700;
      letter-spacing:.2px;
      font-size:1.1rem;
  }
  .card-body {
      padding:18px 24px 18px 24px;
      background:#F9FAFB;
  }

  .sec-title {
      color:#10233D;
      font-weight:700;
      margin:6px 0 4px;
      font-size:0.96rem;
  }
  .subtle {
      color:#6B7280;
      font-size:0.8rem;
      margin-top:2px;
  }

  .stFileUploader > div {
      border-radius:18px !important;
      border:1px dashed rgba(16,35,61,.25) !important;
      background:#FFFFFF !important;
  }

  .stTextInput > div > input {
      border-radius:10px !important;
      border:1px solid rgba(16,35,61,.25) !important;
      background:#FFFFFF !important;
  }

  .stNumberInput input, .stDateInput input {
      border-radius:10px !important;
  }

  /* Orange Get Readi button */
  button[kind="formSubmit"],
  [data-testid="baseButton-primaryFormSubmit"],
  [data-testid="baseButton-secondaryFormSubmit"] {
      width:100%;
      max-width:190px;
      padding:12px 16px;
      border-radius:10px !important;
      border:0 !important;
      background:#FF8A1E !important;
      color:#ffffff !important;
      font-weight:800;
      letter-spacing:.2px;
      box-shadow:0 4px 10px rgba(255,138,30,.35);
  }
  button[kind="formSubmit"]:hover,
  [data-testid="baseButton-primaryFormSubmit"]:hover,
  [data-testid="baseButton-secondaryFormSubmit"]:hover {
      background:#F27B00 !important;
  }

  header[data-testid="stHeader"] { background:transparent; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- LAYOUT: 50 / 50 ----------------
banner_col, form_col = st.columns([1, 1], gap="large")

with banner_col:
    st.image("TeamReadi Side Banner.png")

with form_col:
    # Use the form itself as the card; header + body are inside
    with st.form("generate_form", clear_on_submit=False):
        st.markdown("<div class='card-header'>Generate New Report</div>", unsafe_allow_html=True)
        st.markdown("<div class='card-body'>", unsafe_allow_html=True)

        # --- Upload Resumes ---
        st.markdown("<div class='sec-title'>Upload Resumes</div>", unsafe_allow_html=True)
        resumes = st.file_uploader(
            "Drag & drop files here, or browse",
            type=["pdf", "doc", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        st.markdown("<div class='subtle'>PDF, DOC, DOCX, TXT</div>", unsafe_allow_html=True)

        # --- Project Requirements ---
        st.markdown("<div class='sec-title' style='margin-top:10px;'>Project Requirements</div>",
                    unsafe_allow_html=True)
        proj = st.file_uploader(
            "Drag & drop files here, or browse",
            type=["pdf", "doc", "docx", "txt", "md"],
            accept_multiple_files=True,
            key="proj",
            label_visibility="collapsed",
        )
        req_url = st.text_input("Or paste job / RFP URL", placeholder="https://...")

        st.markdown("<hr />", unsafe_allow_html=True)

        # --- Import Calendar ---
        st.markdown("<div class='sec-title'>Import Calendar</div>", unsafe_allow_html=True)
        cal_mode = st.radio(
            "Calendar import mode",
            options=["Calendar link", "Randomize hours (demo mode)"],
            horizontal=True,
            label_visibility="collapsed",
        )

        cal_url = ""
        randomize_seed = None

        if cal_mode == "Calendar link":
            cal_url = st.text_input(
                "Paste a public/shared calendar URL (Google, Outlook, etc.)",
                placeholder="https://calendar.google.com/calendar/ical/…",
            )
        else:
            randomize_seed = st.slider("Average utilization target (%)", 10, 100, 60)

        st.markdown("<hr />", unsafe_allow_html=True)

        # --- Availability Parameters ---
        st.markdown("<div class='sec-title'>Availability Parameters</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("Start date", value=dt.date.today())
        with c2:
            end_date = st.date_input("End date", value=dt.date.today() + dt.timedelta(days=30))

        # Working days
        st.markdown("<div class='sec-title' style='margin-top:8px;'>Working days</div>",
                    unsafe_allow_html=True)
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        defaults = [True, True, True, True, True, False, False]
        dcols = st.columns(7)
        workdays_checks = []
        for i, col in enumerate(dcols):
            with col:
                workdays_checks.append(
                    st.checkbox(day_labels[i], value=defaults[i], key=f"d{i}")
                )
        selected_days = [d for d, keep in zip(day_labels, workdays_checks) if keep]

        # Maximum hours per day – narrow + centered
        st.markdown(
            "<div class='sec-title' style='margin-top:12px;'>Maximum work hours per day</div>",
            unsafe_allow_html=True,
        )
        mh_left, mh_center, mh_right = st.columns([2, 1, 2])
        with mh_center:
            max_daily = st.number_input(
                "",
                min_value=1.0,
                max_value=12.0,
                value=8.0,
                step=1.0,
                label_visibility="collapsed",
            )

        # Centered orange CTA directly under the number input
        cta_left, cta_mid, cta_right = st.columns([2, 1, 2])
        with cta_mid:
            submitted = st.form_submit_button("Get Readi!")

        # close card-body div
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- HANDLE SUBMIT ----------------
if "submitted" in locals() and submitted:
    st.session_state["resumes"] = [{"name": f.name, "data": f.getvalue()} for f in (resumes or [])]
    st.session_state["req_files"] = [{"name": f.name, "data": f.getvalue()} for f in (proj or [])]

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
            "alpha": SKILL_WEIGHT,
        }
    )

    st.switch_page("pages/01_Results.py")
