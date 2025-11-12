# app.py — TeamReadi Landing (collects inputs, then routes to Results)
import os, json, base64, datetime as dt
import streamlit as st
from openai import OpenAI

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="TeamReadi", page_icon="🕒", layout="wide")

# ----- Fixed weighting -----
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
st.markdown("""
<style>
  .main { background:#F7F8FA; }
  .block-container { padding-top:1.2rem; padding-bottom:2rem; max-width:1180px; }

  /* CARD STYLE */
  .card { background:#FFF; border-radius:14px; box-shadow:0 6px 24px rgba(16,35,61,.12);
          border:1px solid rgba(16,35,61,.06); overflow:hidden; }
  .card.narrow { max-width:860px; margin-left:auto; margin-right:auto; }
  .card-header { background:linear-gradient(135deg,#0B1F3A 0%, #FF8A1E 100%);
                 color:#fff; padding:18px 22px; font-weight:700; letter-spacing:.2px; font-size:1.05rem; }
  .card-body { padding:18px; }

  .sec-title { color:#10233D; font-weight:700; margin:8px 0 6px; }
  .stFileUploader > div { border:1px dashed rgba(16,35,61,.25)!important; border-radius:10px; }

  /* FORM SUBMIT BUTTON (orange, full-width, centered via columns) */
  div.stForm button[kind="formSubmit"] {
      width:100%; padding:12px 16px; border-radius:10px; border:0;
      background:#FF8A1E; color:#fff; font-weight:800; letter-spacing:.2px;
  }
  div.stForm button[kind="formSubmit"]:hover { background:#F27B00; }

  header[data-testid="stHeader"] { background:transparent; }
</style>
""", unsafe_allow_html=True)

# ---------------- LAYOUT ----------------
left, right = st.columns([0.9, 1.3], gap="large")

with left:
    st.image("TeamReadi Side Banner.png", use_container_width=True)

with right:
    st.markdown("<div class='card narrow'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>Generate New Report</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-body'>", unsafe_allow_html=True)

    # ---------- SINGLE FORM (only one!) ----------
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
            options=["Calendar link", "Manual entry / upload", "Randomize hours"],
            horizontal=True,
            label_visibility="collapsed",
        )

        cal_url = ""
        cal_file = None
        randomize_seed = None

        if cal_mode == "Calendar link":
            cal_url = st.text_input(
                "Paste a public or shared calendar URL (Google, Outlook, etc.)",
                placeholder="https://...", help="Must be publicly available."
            )
        elif cal_mode == "Manual entry / upload":
            cal_file = st.file_uploader("Upload .ics file", type=["ics"], key="cal_ics")
            st.caption("ICS only for now.")
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

        max_daily = st.number_input(
            "Maximum work hours per day", min_value=1.0, max_value=12.0, value=8.0, step=1.0,
            help=None  # no default hint text
        )

        # Centered orange CTA
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            submitted = st.form_submit_button("Get Readi!")

    st.markdown("</div></div>", unsafe_allow_html=True)

# ---------------- HANDLE SUBMIT ----------------
if submitted:
    # Save inputs to session for the Results page
    st.session_state["resumes"] = [{"name": f.name, "data": f.getvalue()} for f in (resumes or [])]
    st.session_state["req_files"] = [{"name": f.name, "data": f.getvalue()} for f in (proj or [])]
    st.session_state.update({
        "req_url": req_url or "",
        "cal_method": cal_mode,
        "cal_link": cal_url or "",
        "random_target": randomize_seed,
        "cal_upload": None if cal_file is None else {"name": cal_file.name, "data": cal_file.getvalue()},
        "start_date": str(start_date),
        "end_date": str(end_date),
        "workdays": selected_days,
        "max_hours": float(max_daily),
        "alpha": SKILL_WEIGHT,  # or swap to a slider in the UI if you prefer
    })
    # Navigate to results page
    st.switch_page("pages/01_Results.py")
