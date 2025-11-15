"""
roles_backend.py
Infer resume roles and project-fit narratives using OpenAI.
"""

import json
import os
from typing import Dict, Any

from openai import OpenAI


# ---------- OpenAI client helper ----------

def _get_client() -> OpenAI | None:
    """
    Match how app.py loads the key: st.secrets first, then env.
    """
    try:
        import streamlit as st  # type: ignore
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None

    if not key:
        key = os.getenv("OPENAI_API_KEY")

    if not key:
        return None

    return OpenAI(api_key=key)


# ---------- Role buckets & fallback ----------

ROLE_BUCKETS = [
    "PM/Admin",
    "Support/Coordination",
    "Field/Operator",
    "Out-of-scope",
]


def _fallback_role(resume_text: str) -> Dict[str, Any]:
    """
    Cheap heuristic if the LLM call fails.
    Looks at keywords and assigns a rough bucket.
    """
    text = (resume_text or "").lower()

    if any(w in text for w in ["project manager", "project engineer", "submittal", "rfi", "scheduling", "estimating"]):
        bucket = "PM/Admin"
        title = "Project Engineer / Admin (fallback)"
    elif any(w in text for w in ["supervisor", "logistics", "coordinator", "operations", "safety"]):
        bucket = "Support/Coordination"
        title = "Operations / Support (fallback)"
    elif any(w in text for w in ["operator", "laborer", "equipment", "paving", "flagger", "crew"]):
        bucket = "Field/Operator"
        title = "Field Operator / Labor (fallback)"
    else:
        bucket = "Out-of-scope"
        title = "Not obviously related to construction field work (fallback)"

    return {
        "bucket": bucket,
        "role_title": title,
        "project_fit_summary": "Heuristic classification; unable to run full AI analysis.",
        "unsuitable_reason": "" if bucket != "Out-of-scope" else "Experience is largely outside construction / field operations.",
    }


# ---------- Main role inference function ----------

def infer_resume_role(job_text: str, resume_text: str) -> Dict[str, Any]:
    """
    Given the full project description (job_text) and a single resume text,
    infer:
      - which ROLE BUCKET the person belongs in
      - a human-readable role title
      - a short memo about how they fit (or don't fit) this project

    Returns a dict:
    {
        "bucket": "PM/Admin" | "Support/Coordination" | "Field/Operator" | "Out-of-scope",
        "role_title": "Project Engineer / APM",
        "project_fit_summary": "Short narrative about fit.",
        "unsuitable_reason": "If not suitable, why."
    }
    """
    client = _get_client()
    if not client:
        return _fallback_role(resume_text)

    prompt = f"""
You are assisting with staffing for a construction project.

First, read the PROJECT DESCRIPTION and understand what types of ROLES and responsibilities it requires.

Then, read the RESUME and infer what kind of role this person actually fits.

You must:
1. Assign the candidate to ONE of these buckets:
   - "PM/Admin"            (project manager, assistant PM, project engineer, construction admin)
   - "Support/Coordination" (operations supervisor, logistics, documentation support, safety-only roles)
   - "Field/Operator"      (laborer, equipment operator, crew member, working hands-on in the field)
   - "Out-of-scope"        (experience not relevant to construction project staffing)

2. Suggest a concise role_title describing what they would realistically be hired as
   (e.g., "Assistant Project Manager", "Traffic Control Coordinator (trainable)", "General Labor (trainable)").

3. Write a project_fit_summary explaining:
   - What the project is asking for in plain language
   - How this person's background DOES or DOES NOT align with that
   - Whether you'd consider them for any role on this project, and if so which one.

4. If they are not really suitable for this project at all, include a clear unsuitable_reason.

PROJECT DESCRIPTION:
\"\"\"{job_text}\"\"\"

RESUME:
\"\"\"{resume_text}\"\"\"

Return ONLY JSON with:
{{
  "bucket": "...",
  "role_title": "...",
  "project_fit_summary": "...",
  "unsuitable_reason": "..."   // empty string if suitable
}}
"""

    try:
    resp = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in construction staffing. "
                    "You classify candidates into role buckets and explain fit for a given project. "
                    "Respond only with valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=800,
    )

    raw = resp.choices[0].message.content or ""

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _fallback_role(resume_text)

    bucket = data.get("bucket") or ""
    if bucket not in ROLE_BUCKETS:
        bucket = "Out-of-scope"

    return {
        "bucket": bucket,
        "role_title": data.get("role_title", "").strip() or "Unspecified role",
        "project_fit_summary": data.get("project_fit_summary", "").strip() or "",
        "unsuitable_reason": data.get("unsuitable_reason", "").strip(),
    }

except Exception:
    return _fallback_role(resume_text)
