# app.py — TeamReadi Landing (collects inputs, then routes to Results)
import os, json, base64, datetime as dt
import streamlit as st
from openai import OpenAI

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="TeamReadi", page_icon="🕒", layout="wide")

# ----- Fixed weighting for scoring -----
SKILL_WEIGHT = 0.70  # 70% skills, 30% availability

# ----- OpenAI setup -----
API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = st.secrets.get("MODEL_NAME", "gpt-4o")
EMBED_MODEL = st.secrets.get("EMBED_MODEL", "text-embedding-3-large")

if not API_KEY:
    st.error("OPENAI_API_KEY is not set. Add it in .streamlit/secrets.toml or as an env var.")
    st.stop()

client = OpenAI(api_key=API_KEY)

# ---------------- STYLING ----------------
st.markdown(
    """
<style>
/* Make entire app background pure white */
[data-testid="stAppViewContainer"],
.main,
html, body {
    background:#ffffff !important;
}

/* Center content and keep width similar to your mockup */
.block-container {
    max-width:1100px !important;
    padding-top:1.5rem;
    padding-bottom:2rem;
}

/* Card header + body share same border & shadow (no separate bubble) */
.tr-card {
    margin-top:0;
}

.tr-header {
    background:#10233D;
    color:#fff;
    padding:18px 26px;
    font-weight:700;
    font-size:1.15rem;
    letter-spacing:.2px;
    border-radius:18px 18px 0 0;
    border:1px solid rgba(16,35,61,.08);
    border-bottom:none;
    box-shadow:0 10px 28px rgba(16,35,61,.10);
    margin-bottom:0;
}

/* Light gray only inside the form */
.tr-body {
    background:#F8F9FC;
    padding:22px 24px 26px;
    border-radius:0 0 18px 18px;
    border:1px solid rgba(16,35,61,.08);
    border-top:none;
    box-shadow:0 10px 28px rgba(16,35,61,.10);
}

/* Section titles */
.sec-title {
    color:#10233D;
    font-weight:700;
    margin:8px 0 6px;
    font-size:1.0rem;
}

/* Make uploaders look like cards */
.stFileUploader > div {
    border:1px dashed rgba(16,35,61,.25)!important;
    border-radius:12px;
}

/* Text & number inputs: clear white boxes with visible borders */
.stTextInput input,
.stNumberInput input {
    border-radius:12px !important;
    border:1px solid #CDD4E1 !important;
    background:#ffffff !important;
}

/* Prevent checkbox labels (Mon, Tue, ...) from splitting onto 2 lines */
div[data-testid="stCheckbox"] label {
    white-space:nowrap;
}

/* Orange form submit button, centered */
div.stForm button[kind="formSubmit"],
div.stForm [data-testid="baseButton-primaryFormSubmit"] {
    min-width:160px;
    padding:10px 20px;
    border-radius:10px;
    border:0 !important;
    background:#FF8A1E !important;
    color:#ffffff !important;
    font-weight:800;
    letter-spacing:.2px;
    box-shadow:0 2px 0 rgba(0,0,0,.06);
}
div.stForm button[kind="formSubmit"]:hover,
div.stForm [data-testid="baseButton-primaryFormSubmit"]:hover {
    background:#F27B00 !important;
}

/* Remove default app header tint */
header[data-testid="stHeader"] { background:transparent; }

/* Equal treatment for banner column */
.equal-col img {
    width:100%;
    height:auto;
    object-fit:contain;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- LAYOUT: 1/2 banner, 1/2 form ----------------
left, right = st.columns([1, 1], gap="large")

# Left: banner
with left:
    st.markdown("<div class='equal-col'>", unsafe_allow_html=True)
    st.image("TeamReadi Side Banner.png", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Right: header + form inside one rounded card
with right:
    st.markdown("<div class='tr-card'>", unsafe_allow_html=True)
    st.markdown("<div class='tr-header'>Generate New Report</div>", unsafe_allow_html=True)
    st.markdown("<div class='tr-body'>", unsafe_allow_html=True)

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
        req_url = st.text_input("Or paste job / RFP URL")

        st.markdown("---")

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
                help="Must be publicly available.",
            )
        else:
            randomize_seed = st.slider("Average utilization target (%)", 10, 100, 60)

        st.markdown("---")

        # --- Availability Parameters ---
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
        workdays_checks = []
        for i, col in enumerate(dcols):
            with col:
                workdays_checks.append(st.checkbox(day_labels[i], value=defaults[i], key=f"d{i}"))
        selected_days = [d for d, keep in zip(day_labels, workdays_checks) if keep]

        # --- Maximum work hours per day (label + centered input) ---
        mid1, mid2, mid3 = st.columns([1, 2, 1])
        with mid2:
            # Force a clean 2-line label instead of messy wrapping
            st.markdown(
                "<div class='sec-title' style='margin-top:8px;'>Maximum work hours per<br>day</div>",
                unsafe_allow_html=True,
            )
            max_daily = st.number_input(
                "",
                min_value=1.0,
                max_value=12.0,
                value=8.0,
                step=1.0,
                help=None,
            )

        # --- Centered orange Get Readi! button under the 8.00 box ---
        g1, g2, g3 = st.columns([1, 2, 1])
        with g2:
            submitted = st.form_submit_button("Get Readi!")

    # close body + card
    st.markdown("</div></div>", unsafe_allow_html=True)

# ---------------- HANDLE SUBMIT ----------------
if "submitted" in locals() and submitted:
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
            "alpha": SKILL_WEIGHT,
        }
    )
    st.switch_page("pages/01_Results.py")
