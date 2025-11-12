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

  /* FORM SUBMIT BUTTON */
  div.stForm button[kind="formSubmit"] {
      width:100%;
      padding:12px 16px;
      border-radius:10px;
      border:0;
      background:#FF8A1E;
      color:#fff;
      font-weight:800;
      letter-spacing:.2px;
  }
  div.stForm button[kind="formSubmit"]:hover { background:#F27B00; }

  header[data-testid="stHeader"] { background:transparent; }
</style>
""", unsafe_allow_html=True)


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

