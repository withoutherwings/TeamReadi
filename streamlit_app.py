# app.py — TeamReadi Landing (reset to cohesive card layout)

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
  /* Page background + main container */
  .main { background:#ffffff; }
  .block-container {
      padding-top:1.2rem;
      padding-bottom:2rem;
      max-width:1200px;
  }

  /* Card shell: header + body in ONE card */
  .card {
      background:#FFFFFF;
      border-radius:24px;
      border:1px solid rgba(16,35,61,.12);
      box-shadow:0 12px 30px rgba(16,35,61,.10);
      overflow:hidden;  /* ensures rounded edges apply to everything inside */
  }
  .card-header {
      background:#0F243D;
      color:#ffffff;
      padding:18px 28px;
      font-weight:700;
      letter-spacing:.2px;
      font-size:1.15rem;
      border-radius:24px 24px 0 0;  /* rounded top corners */
  }
  .card-body {
      padding:20px 26px 22px 26px;
      border-radius:0 0 24px 24px;   /* rounded bottom corners */
      background:#F9FAFB;
  }

  .sec-title {
      color:#10233D;
      font-weight:700;
      margin:10px 0 6px;
      font-size:0.98rem;
  }
  .subtle {
      color:#6B7280;
      font-size:0.8rem;
      margin-top:2px;
  }

  /* Uploaders */
  .stFileUploader > div {
      border-radius:18px !important;
      border:1px dashed rgba(16,35,61,.25) !important;
      background:#FFFFFF !important;
  }

  /* Text inputs (URLs etc.) */
  .stTextInput > div > input {
      border-radius:10px !important;
      border:1px solid rgba(16,35,61,.25) !important;
      background:#FFFFFF !important;
  }

  /* Number + date inputs */
  .stNumberInput input, .stDateInput input {
      border-radius:10px !important;
  }

  /* Orange submit button inside form */
  div.stForm button[kind="formSubmit"],
  div.stForm [data-testid="baseButton-primaryFormSubmit"],
  div.stForm [data-testid="baseButton-secondaryFormSubmit"] {
      width:100%;
      max-width:190px;
      padding:12px 16px;
      border-radius:10px;
      border:0 !important;
      background:#FF8A1E !important;
      color:#ffffff !important;
      font-weight:800;
      letter-spacing:.2px;
      box-shadow:0 4px 10px rgba(255,138,30,.35);
  }
  div.stForm button[kind="formSubmit"]:hover,
  div.stForm [data-testid="baseButton-primaryFormSubmit"]:hover,
  div.stForm [data-testid="baseButton-secondaryFormSubmit"]:hover {
      background:#F27B00 !important;
  }

  header[data-testid="stHeader"] { background:transparent; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- LAYOUT: 1/3 BANNER | 2/3 FORM ----------------
banner_col, form_col = st.columns([1, 2], gap="large")

with banner_col:
    st.image("TeamReadi Side Banner.png", use_container_width=True)

with form_col:
    # ONE unified card: header + body
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>Generate New Report</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-body'>", unsafe_allow_html=True)

    # ---------------- FORM ----------------
    with st.form("generate_form", clear_on_submit=False):
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
        st.markdown(
            "<div class='sec-title' style='margin-top:14px;'>Project Requirements</div>",
            unsafe_allow_html=True,
        )
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
        st.markdown(
            "<div class='sec-title' style='margin-top:10px;'>Working days</div>",
            unsafe_allow_html=True,
        )
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
            "<div class='sec-title' style='margin-top:16px;'>Maximum work hours per day</div>",
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

    # close body + card
    st.markdown("</div></div>", unsafe_allow_html=True)

# ---------------- HANDLE SUBMIT ----------------
if submitted:
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
