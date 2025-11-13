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

# ---------------- GLOBAL STYLING ----------------
st.markdown("""
<style>

    /* Page background */
    .main { background:#ffffff !important; }

    .block-container {
        padding-top:1.2rem !important;
        padding-bottom:2rem !important;
        max-width:1450px !important;
    }

    /* Card container */
    .card {
        background:#FFFFFF;
        border-radius:14px;
        overflow:hidden;
        border:1px solid rgba(16,35,61,.08);
        box-shadow:0 10px 28px rgba(16,35,61,.10);
    }

    .card.narrow {
        max-width:1000px;
        margin:0 auto;
    }

    /* Header Bar */
    .card-header {
        background:#10233D !important;  /* solid navy */
        color:#fff;
        padding:20px 26px;
        font-weight:700;
        border-top-left-radius:14px;
        border-top-right-radius:14px;
        letter-spacing:.2px;
        font-size:1.15rem;
    }

    .card-body {
        padding:22px;
        background:#ffffff;
    }

    .sec-title {
        color:#10233D;
        font-weight:700;
        margin:12px 0 6px;
        font-size:1rem;
    }

    /* URL bars and number inputs always visible */
    .stTextInput input,
    .stNumberInput input {
        border-radius:10px !important;
        border:1px solid #FF8A1E !important;
        background:#fff !important;
        box-shadow:none !important;
    }

    /* File upload boxes */
    .stFileUploader > div {
        border:1px dashed rgba(16,35,61,.25) !important;
        border-radius:10px !important;
    }

    /* Orange CTA button */
    div.stForm button[kind="formSubmit"],
    div.stForm [data-testid="baseButton-primaryFormSubmit"],
    div.stForm [data-testid="baseButton-secondaryFormSubmit"] {
        width:auto !important;
        padding:12px 32px !important;
        border-radius:10px !important;
        border:0 !important;
        background:#FF8A1E !important;
        color:#ffffff !important;
        font-weight:800 !important;
        letter-spacing:.2px !important;
        display:block !important;
        margin:0 auto !important;
    }

    div.stForm button:hover {
        background:#F27B00 !important;
    }

    header[data-testid="stHeader"] { background:transparent !important; }

</style>
""", unsafe_allow_html=True)



# ---------------- LAYOUT ----------------
left, right = st.columns([1, 2], gap="large")     # 1/3 | 2/3 layout


# LEFT BANNER
with left:
    st.image("TeamReadi Side Banner.png", use_container_width=True)



# RIGHT FORM
with right:
    st.markdown("<div class='card narrow'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>Generate New Report</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-body'>", unsafe_allow_html=True)

    # ---------------- SINGLE FORM ----------------
    with st.form("generate_form", clear_on_submit=False):

        # UPLOAD RESUMES
        st.markdown("<div class='sec-title'>Upload Resumes</div>", unsafe_allow_html=True)
        resumes = st.file_uploader(
            "Drag & drop files here, or browse",
            type=["pdf", "doc", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        st.caption("PDF, DOC, DOCX, TXT")

        # PROJECT REQUIREMENTS
        st.markdown("<div class='sec-title'>Project Requirements</div>", unsafe_allow_html=True)
        proj = st.file_uploader(
            "Drag & drop files here, or browse",
            type=["pdf", "doc", "docx", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        req_url = st.text_input("Or paste job / RFP URL", placeholder="https://...")

        st.markdown("---")

        # CALENDAR OPTIONS
        st.markdown("<div class='sec-title'>Import Calendar</div>", unsafe_allow_html=True)
        cal_mode = st.radio(
            "Calendar import mode",
            ["Calendar link", "Randomize hours (demo mode)"],
            horizontal=True,
            label_visibility="collapsed",
        )

        cal_url = ""
        randomize_seed = None

        if cal_mode == "Calendar link":
            cal_url = st.text_input(
                "Paste a public/shared calendar URL (Google, Outlook, etc.)",
                placeholder="https://calendar.google.com/calendar/ical/…",
                help="Must be publicly available."
            )
        else:
            randomize_seed = st.slider(
                "Average utilization target (%)",
                10, 100, 60
            )

        st.markdown("---")

        # AVAILABILITY
        st.markdown("<div class='sec-title'>Availability Parameters</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("Start date", value=dt.date.today())
        with c2:
            end_date = st.date_input("End date", value=dt.date.today() + dt.timedelta(days=30))

        st.markdown("<div class='sec-title' style='margin-top:8px;'>Working days</div>", unsafe_allow_html=True)
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        defaults = [True, True, True, True, True, False, False]

        dcols = st.columns(7)
        workdays = []
        for i, col in enumerate(dcols):
            with col:
                workdays.append(st.checkbox(day_labels[i], value=defaults[i], key=f"d{i}"))

        # NARROW MAX HOURS INPUT
        st.markdown("Maximum work hours per day")
        h1, hmid, h3 = st.columns([1, 1, 1])
        with hmid:
            max_daily = st.number_input(
                "",
                min_value=1.0, max_value=12.0, value=8.0, step=1.0,
                key="max_daily_hours",
            )

        st.write("")

        # CTA BUTTON (centered automatically)
        submitted = st.form_submit_button("Get Readi!")

    st.markdown("</div></div>", unsafe_allow_html=True)



# ---------------- STORE DATA & ROUTE ----------------
if submitted:
    st.session_state["resumes"] = [{"name": f.name, "data": f.getvalue()} for f in (resumes or [])]
    st.session_state["req_files"] = [{"name": f.name, "data": f.getvalue()} for f in (proj or [])]

    st.session_state.update({
        "req_url": req_url,
        "cal_method": cal_mode,
        "cal_link": cal_url,
        "random_target": randomize_seed,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "workdays": [d for d, keep in zip(day_labels, workdays) if keep],
        "max_hours": float(max_daily),
        "alpha": SKILL_WEIGHT,
    })

    st.switch_page("pages/01_Results.py") 
