"""
skills_backend.py
Project requirements ↔ resume matching using OpenAI.
"""

import json
import os
import re
from typing import List, Dict, Any

from openai import OpenAI


# ---------- OpenAI client helper ----------

def _get_client() -> OpenAI | None:
    """Match how app.py loads the key: st.secrets first, then env."""
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


# ---------- Fallback helpers (only if API really fails) ----------

def _fallback_requirements(project_text: str) -> List[Dict[str, Any]]:
    text = (project_text or "").lower()
    tokens = re.findall(r"[a-z]{4,}", text)

    uniq: List[str] = []
    for t in tokens:
        if t not in uniq:
            uniq.append(t)
        if len(uniq) >= 8:
            break

    if not uniq:
        uniq = ["construction", "experience", "coordination"]

    reqs: List[Dict[str, Any]] = []
    for i, word in enumerate(uniq, start=1):
        reqs.append(
            {
                "id": f"S{i}",
                "label": word.capitalize(),
                "description": f"Evidence of {word} in the project description.",
                "importance": 2,
            }
        )
    return reqs


def _fallback_resume_score(requirements: List[Dict[str, Any]], resume_text: str) -> Dict[str, Any]:
    text = (resume_text or "").lower()
    per_skill = []
    total_weight = 0.0
    got_weight = 0.0

    for req in requirements:
        label = str(req.get("label", "")).lower()
        importance = float(req.get("importance", 2) or 2)
        total_weight += importance

        match_status = "no_match"
        evidence = ""

        if label and label in text:
            match_status = "strong_match"
            idx = text.find(label)
            snippet = resume_text[max(idx - 40, 0): idx + len(label) + 40]
            evidence = snippet.strip()
            got_weight += importance

        per_skill.append(
            {
                "id": req.get("id", ""),
                "label": req.get("label", ""),
                "match_status": match_status,
                "evidence_snippet": evidence,
            }
        )

    skill_pct = (got_weight / total_weight * 100.0) if total_weight > 0 else 0.0

    return {"per_skill": per_skill, "skill_match_pct": skill_pct}


# ---------- Project requirement extraction ----------

def extract_project_requirements(project_text: str) -> List[Dict[str, Any]]:
    """
    Use the LLM to extract 5–10 workforce requirements from the project description.
    Falls back to heuristics only if the LLM fails.
    Output format:
    [
      {"id": "S1", "label": "...", "description": "...", "importance": 1–3},
      ...
    ]
    """
    client = _get_client()
    if not client or not project_text.strip():
        return _fallback_requirements(project_text)

    system_msg = (
        "You extract staffing requirements from construction project descriptions and RFPs. "
        "Return only JSON — no markdown — in the format {\"requirements\": [...]}."
    )

    user_prompt = f"""
Project Description / Scope:
\"\"\"{project_text}\"\"\"


Return exactly this JSON schema:
{{
  "requirements": [
    {{
      "id": "S1",
      "label": "4–12 word capability phrase (NOT a location or agency name)",
      "description": "1–2 sentence explanation of what the project needs",
      "importance": 1 | 2 | 3
    }},
    ...
  ]
}}

Rules:
- 5–10 items
- Must be multi-word capability phrases (3+ words minimum, ideally 4–12)
- Must describe what a PERSON must know or be able to do
- No locations, owners, agencies, contractors, or dates as labels
- At least 2 items must have importance = 3
"""

    try:
        resp = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=900,
        )

        raw = resp.choices[0].message.content or ""
        data = json.loads(raw)

        reqs = data.get("requirements", [])
        if not isinstance(reqs, list) or not reqs:
            return _fallback_requirements(project_text)

        cleaned = []
        for i, r in enumerate(reqs, start=1):
            label = str(r.get("label", "")).strip()
            if not label or len(label.split()) < 3:
                continue
            cleaned.append(
                {
                    "id": r.get("id") or f"S{i}",
                    "label": label,
                    "description": r.get("description", "").strip(),
                    "importance": int(r.get("importance", 2) or 2),
                }
            )

        return cleaned or _fallback_requirements(project_text)

    except Exception:
        return _fallback_requirements(project_text)



# ---------- Resume scoring ----------

def score_resume_against_requirements(requirements: List[Dict[str, Any]], resume_text: str) -> Dict[str, Any]:
    """
    Compare resume against project requirements using the LLM.
    Response JSON format:
    {
      "per_skill": [
        {"id": "S1", "label": "...", "match_status": "strong_match|partial_match|no_match", "evidence_snippet": "..."},
        ...
      ],
      "skill_match_pct": number
    }
    """
    client = _get_client()
    if not client or not resume_text.strip():
        return _fallback_resume_score(requirements, resume_text)

    req_json = json.dumps(requirements)

    system_msg = (
        "You evaluate construction resumes against project requirements. "
        "Respond only with valid JSON: {\"per_skill\": [...], \"skill_match_pct\": number}."
    )

    user_prompt = f"""
Project requirements (JSON):
{req_json}

Resume:
\"\"\"{resume_text}\"\"\"

Rules:
- For EACH requirement, assign:
    - strong_match  = full credit
    - partial_match = half credit
    - no_match      = zero credit
- Provide 1 evidence_snippet for strong/partial matches
- skill_match_pct is weighted by requirement.importance (1,2,3)
"""

    try:
        resp = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=900,
        )

        raw = resp.choices[0].message.content or ""
        data = json.loads(raw)

        # Validate structure
        if "per_skill" not in data or "skill_match_pct" not in data:
            return _fallback_resume_score(requirements, resume_text)

        return data

    except Exception:
        return _fallback_resume_score(requirements, resume_text)

# ---------- Backwards-compatible wrapper for older pipeline code ----------

def extract_resume_skills(*args, **kwargs):
    """
    Backwards-compatible helper used by backend.pipeline.build_candidate_profile.

    Accepts either:
      - (project_text, resume_text)
      - (requirements_list, resume_text)

    Returns:
      {
        "per_skill": [...],
        "skill_match_pct": number
      }
    using the same LLM / fallback logic as score_resume_against_requirements.
    """
    if len(args) < 2:
        raise ValueError(
            "extract_resume_skills expects at least project/context and resume_text"
        )

    first, resume_text = args[0], args[1]

    # If the first arg already looks like a requirements list, use it directly.
    if isinstance(first, list) and first and isinstance(first[0], dict) and "label" in first[0]:
        requirements = first
    else:
        # Otherwise treat it as project text and extract requirements from it.
        project_text = str(first) if first is not None else ""
        requirements = extract_project_requirements(project_text)

    return score_resume_against_requirements(requirements, resume_text)
