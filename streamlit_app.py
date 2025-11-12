# app.py  — TeamReadi Landing (collects inputs, then routes to Results)
import base64, datetime as dt
import streamlit as st

import os
from openai import OpenAI

# Load your API key automatically from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define which embedding model you’ll use
EMBED_MODEL = "text-embedding-3-large"


try:
    test = client.embeddings.create(model=EMBED_MODEL, input="Project management in construction")
    st.success(f"Connected! Embedding length: {len(test.data[0].embedding)}")
except Exception as e:
    st.error(f"Connection failed: {e}")


st.set_page_config(page_title="TeamReadi", page_icon="✅", layout="centered")

# ------------ helpers ------------
def _logo_b64(names=("teamreadi-logo.png","TeamReadi Logo.png","TeamReadi_Logo.png","TeamReadi_Icon.png")):
    for p in names:
        try:
            return base64.b64encode(open(p,"rb").read()).decode("utf-8")
        except Exception:
            pass
    return None

def _stash_files(key, uploads):
    st.session_state[key] = []
    for up in (uploads or []):
        st.session_state[key].append({"name": up.name, "data": up.getvalue()})

# ------------ styles ------------
st.markdown("""
<style>
.block-container{padding-top:2rem !important} body{background:#f8fafc}
.card{border-radius:1rem;overflow:hidden;box-shadow:0 10px 30px rgba(15,47,95,.12);background:white}
.card-head{padding:1.25rem;color:#fff;background:linear-gradient(135deg,#0f2f5f,#174a8b 40%,#ff9b28)}
.stepbar{display:flex;align-items:center;gap:.5rem;margin-top:.75rem;font-size:.72rem}
.dot{width:24px;height:24px;border-radius:9999px;display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,.2);font-weight:700}
.line{flex:1;height:2px;background:rgba(255,255,255,.25);margin:0 .5rem}
.hint{font-size:.75rem;opacity:.95}
.cta{background:#f58b1f;color:#fff;border-radius:.75rem;padding:.75rem 1.25rem;font-weight:600;border:0}
.cta:hover{background:#ff9b28}
</style>
""", unsafe_allow_html=True)

# ---------- header / logo ----------
logo = _logo_b64()

# responsive, shadow-free logo
if logo:
    st.markdown(
        f"""
        <style>
          .tr-logo img {{
            width: 360px;               /* default (desktop) */
            box-shadow: none;           /* no drop shadow */
            background: none;
          }}
          @media (max-width: 768px) {{
            .tr-logo img {{ width: 280px; }}  /* tablets */
          }}
          @media (max-width: 480px) {{
            .tr-logo img {{ width: 220px; }}  /* phones */
          }}
        </style>
        <div class="tr-logo" style="text-align:center; margin-top:-20px;">
          <img src="data:image/png;base64,{logo}" alt="TeamReadi logo"/>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    # Fallback if base64 helper can't find the file
    st.image("public/teamreadi-logo.png", width=360)


# ------------ form ------------
with st.container():
    # Resumes
    st.subheader("Upload Resumes", divider="gray")
    resumes = st.file_uploader(
        "Resumes (PDF/DOC/DOCX/TXT)",
        type=["pdf","doc","docx","txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    # Project requirements
    st.subheader("Project Requirements", divider="gray")
    req_files = st.file_uploader(
        "RFP / Job criteria (PDF/DOC/DOCX/TXT/MD)",
        type=["pdf","doc","docx","txt","md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="req_files"
    )
    req_url = st.text_input("Or paste job/RFP URL here")

    # Calendar
    st.subheader("Import Calendar", divider="gray")
    cal_method = st.radio("Method", ["Calendar Link","Manual Entry / Upload","Randomize Hours"], horizontal=True)
    cal_link = cal_upload = None
    random_target = None
    if cal_method == "Calendar Link":
        cal_link = st.text_input(
            "Paste calendar link (Google, Outlook, etc.)",
            placeholder="https://…",
            help="Must be publicly available."
        )
    elif cal_method == "Manual Entry / Upload":
        cal_upload = st.file_uploader("Upload .ics / CSV / spreadsheet", type=["ics","csv","xls","xlsx"], key="cal_csv")
    else:
        random_target = st.slider("Average utilization target (%)", 10, 100, 60)

    # Availability
    st.subheader("Availability Parameters", divider="gray")
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start Date", value=dt.date.today())
    with c2:
        end_date = st.date_input("End Date", value=dt.date.today())
    workdays = st.multiselect(
        label="Working Days",
        options=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        default=["Mon","Tue","Wed","Thu","Fri"]
    )
    max_hours = st.number_input("Maximum work hours per day", min_value=1, max_value=12, value=8, step=1)

    # Blend control
    alpha = st.slider("Weight on Skills (α)", 0.0, 1.0, 0.7, 0.05,
                      help="ReadiScore = α·SkillFit + (1–α)·AvailabilityFit")

st.markdown("</div>", unsafe_allow_html=True)  # close card

# ------------ CTA: stash & go ------------
if st.button("Get Readi!", type="primary", use_container_width=True):
    _stash_files("resumes", resumes)
    _stash_files("req_files", req_files)
    st.session_state.update({
        "req_url": req_url or "",
        "cal_method": cal_method,
        "cal_link": cal_link or "",
        "random_target": random_target,
        "cal_upload": None if cal_upload is None else {"name": cal_upload.name, "data": cal_upload.getvalue()},
        "start_date": str(start_date),
        "end_date": str(end_date),
        "workdays": workdays,
        "max_hours": int(max_hours),
        "alpha": float(alpha),
    })
    st.switch_page("pages/01_Results.py")
