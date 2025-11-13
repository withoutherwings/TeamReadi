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

/* ONE card: header + body share same rounded border & shadow */
.tr-card {
    background:#FFFFFF;                       /* outer card white */
    border-radius:18px;
    border:1px solid rgba(16,35,61,.08);
    box-shadow:0 10px 28px rgba(16,35,61,.10);
    overflow:hidden;                          /* rounds header and body together */
}

/* Dark navy bar at the very top of that card */
.tr-header {
    background:#10233D;
    color:#fff;
    padding:18px 26px;
    font-weight:700;
    font-size:1.15rem;
    letter-spacing:.2px;
}

/* Light gray body area inside that same card */
.tr-body {
    background:#F8F9FC;
    padding:22px 24px 26px;
}

/* KILL Streamlit's default form "card" so it doesn't create a second bubble */
form[data-testid="stForm"] {
    background:transparent !important;
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

/* Text & number*
