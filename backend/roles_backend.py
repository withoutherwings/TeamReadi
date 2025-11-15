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
    Given the full project description (job_text) and resume text,
    infer the correct role bucket + role title + project-fit narrative.
    Always returns a JSON-safe dict, falling back to heuristics if LLM fails.
    """
    client = _get_client()
    if not client:
        return _fallback_role(resume_text)

    system_msg = (
        "You are an expert in construction staffing. "
        "You classify candidates into role buckets and explain fit for a given project. "
        "Return ONLY valid JSON — no markdown, no commentary."
    )

    user_prompt = f"""
PROJECT DESCRIPTION:
\"\"\"{job_text}\"\"\"


RESUME:
\"\"\"{resume_text}\"\"\"


You MUST return JSON shaped exactly like this:
{{
  "bucket": "PM/Admin" | "Support/Coordination" | "Field/Operator" | "Out-of-scope",
  "role_title": "short human-readable job title",
  "project_fit_summary": "3–6 sentences describing exactly how the person fits or doesn't fit this project",
  "unsuitable_reason": "reason if not suitable, otherwise empty string"
}}
"""

    try:
        resp = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=800,
            temperature=0,
        )

        raw = resp.choices[0].message.content or ""
        data = json.loads(raw)  # throws JSONDecodeError if malformed

        bucket = data.get("bucket", "")
        if bucket not in ROLE_BUCKETS:
            bucket = "Out-of-scope"

        return {
            "bucket": bucket,
            "role_title": data.get("role_title", "").strip() or "Unspecified role",
            "project_fit_summary": data.get("project_fit_summary", "").strip(),
            "unsuitable_reason": data.get("unsuitable_reason", "").strip(),
        }

    except Exception:
        return _fallback_role(resume_text)
