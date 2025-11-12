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
  /* PAGE BACKGROUND – force everything white */
  html, body, .stApp, [data-testid="stAppViewContainer"], .main {
    background-color:#ffffff !important;
  }
  .block-container {
    padding-top:1.2rem;
    padding-bottom:2rem;
    max-width:1180px;
  }

  /* CARD */
  .card{
    background:#FFFFFF;
    border-radius:14px;
    overflow:hidden;                           /* header + body share same shell */
    border:1px solid rgba(16,35,61,.08);
    box-shadow:0 10px 28px rgba(16,35,61,.10); /* subtle shadow around the form only */
  }
  .card.narrow{
    max-width:640px;                           /* narrower so it feels lighter */
    margin-left:auto;
    margin-right:auto;
  }

  /* HEADER – solid navy, rounded top corners, attached to card */
  .card-header{
    background:#10233D;
    color:#fff;
    padding:18px 22px;
    font-weight:700;
    letter-spacing:.2px;
    font-size:1.05rem;
    border-top-left-radius:14px;
    border-top-right-radius:14px;
  }

  .card-body{ padding:18px; }

  /* SECTION TITLES & UPLOADS */
  .sec-title{
    color:#10233D;
    font-weight:700;
    margin:8px 0 6px;
  }
  .stFileUploader > div{
    border:1px dashed rgba(16,35,61,.25) !important;
    border-radius:10px;
  }

  /* ORANGE SUBMIT – force Streamlit’s form button to our brand color */
  div.stForm button[kind="formSubmit"],
  div.stForm button,
  div.stForm [data-testid="baseButton-primaryFormSubmit"],
  div.stForm [data-testid="baseButton-secondaryFormSubmit"]{
      width:100%;
      padding:12px 16px;
      border-radius:10px;
      border:0 !important;
      background:#FF8A1E !important;
      color:#ffffff !important;
      font-weight:800;
      letter-spacing:.2px;
      box-shadow:0 2px 0 rgba(0,0,0,.06);
  }
  div.stForm button:hover,
  div.stForm [data-testid="baseButton-primaryFormSubmit"]:hover,
  div.stForm [data-testid="baseButton-secondaryFormSubmit"]:hover{
      background:#F27B00 !important;
  }

  /* Remove default Streamlit header tint */
  header[data-testid="stHeader"]{
    background:transparent;
  }
</style>
""", unsafe_allow_html=True)

# ---------------- LAYOUT ----------------
# Slightly more balanced split; form column won’t dominate
left, right = st.columns([1, 1.1], gap="large")

with left:
    st.image("TeamReadi Side Banner.png", use_container_width=True)

with right:
    st.markdown("<div class='card narrow'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>Generate New Report</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-body'>", unsafe_allow_html=True)

    # ---------- SINGLE FORM ----------
    with st.form("generate_form", clear_on_submit=False):

        # two columns *inside* the card so the form isn't as tall
        left_col, right_col = st.columns(2, gap="large")

        # -------- LEFT COLUMN: Resumes, Requirements, Calendar --------
        with left_col:
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
            st.markdown("<div class='sec-title' style='margin-top:10px;'>Project Requirements</div>",
                        unsafe_allow_html=True)
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
                # FIX: actually show a CSV/XLSX uploader for manual mode
                cal_file = st.file_uploader(
                    "Upload CSV or spreadsheet (employee_id, start, end, hours)",
                    type=["csv", "xlsx"],
                    key="cal_csv",
                )
                st.caption("CSV / XLSX only. One row per busy block.")
            else:
                randomize_seed = st.slider("Average utilization target (%)", 10, 100, 60)

        # -------- RIGHT COLUMN: Availability --------
        with right_col:
            st.markdown("<div class='sec-title'>Availability Parameters</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                start_date = st.date_input("Start date", value=dt.date.today())
            with c2:
                end_date = st.date_input("End date", value=dt.date.today() + dt.timedelta(days=30))

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

            max_daily = st.number_input(
                "Maximum work hours per day",
                min_value=1.0,
                max_value=12.0,
                value=8.0,
                step=1.0,
                help=None,
            )

        # -------- BOTTOM: centered CTA spanning the full card --------
        st.markdown("")  # small spacer
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
        "alpha": SKILL_WEIGHT,
    })
    st.switch_page("pages/01_Results.py")
