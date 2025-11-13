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

/* Outer card that holds the whole form */
.tr-card {
    background:#FFFFFF;
    border-radius:18px;
    border:1px solid rgba(16,35,61,.08);
    box-shadow:0 10px 28px rgba(16,35,61,.10);
    overflow:hidden; /* ensures children respect rounded corners */
}

/* Light gray body area inside that card */
.tr-body {
    background:#F8F9FC;
    padding:22px 24px 26px;
}

/* Embedded main title INSIDE the form card */
.tr-main-title {
    display:inline-block;
    background:#10233D;
    color:#ffffff;
    padding:10px 18px;
    border-radius:12px;
    font-weight:800;
    font-size:1.1rem;
    margin-bottom:16px;
}

/* Make the Streamlit form itself transparent so we only see our card */
form[data-testid="stForm"] {
    background:transparent !important;
    border:none !important;
    box-shadow:none !important;
}
form[data-testid="stForm"] > div {
    background:transparent !important;
    padding:0 !important;
    border:none !important;
    box-shadow:none !important;
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

/* Text & number inputs: visible white boxes with border */
.stTextInput input,
.stNumberInput input {
    border-radius:12px !important;
    border:1px solid #CDD4E1 !important;
    background:#ffffff !important;
}

/* Date inputs: same style so Start/End date boxes are visible */
[data-testid="stDateInput"] input {
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

# Right: one card containing embedded title + body
with right:
    st.markdown("<div class='tr-card'><div class='tr-body'>", unsafe_allow_html=True)

    # Embedded title at the very top of the card
    st.markdown("<div class='tr-main-title'>Generate New Report</div>", unsafe_allow_html=True)

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

        # --- Project R
