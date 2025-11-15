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
    Returns a list like:
    [
        {"id": "S1", "label": "...", "description": "...", "importance": 1-3},
        ...
    ]
    """
    client = _get_client()
    if not client or not project_text or not project_text.strip():
        return _fallback_requirements(project_text)

    prompt = f"""
You are assisting with construction staffing for project pursuits.

You will read a construction project description or RFP and identify the 5–10
most important SKILLS, CAPABILITIES, or EXPERIENCE requirements needed to staff the work.

VERY IMPORTANT RULES FOR EACH REQUIREMENT:
- It MUST be a multi-word capability phrase (at least 3 words, ideally 4–12).
  Examples:
    - "Experience managing state DOT highway resurfacing projects"
    - "QA/QC documentation and submittal management"
    - "Construction safety management and work zone traffic control"
- It MUST NOT be:
  - A single word (like "State" or "Transportation")
  - A location (like "North Carolina" or "Utah")
  - An agency or owner name (like "Department of Transportation")
- Think in terms of what the PERSON must be able to do or know,
  not who the OWNER is or where the project is located.

For each requirement, return:
- id: a short ID like "S1", "S2", ...
- label: 3–12 word capability label (see examples above)
- description: 1–2 sentence explanation
- importance: 1 (nice to have), 2 (important), 3 (critical)

Project description:
\"\"\"{project_text}\"\"\"

Respond as JSON with a single key "requirements" whose value is a list of objects.
Do not include any text outside the JSON.
"""

    try:
resp = client.chat.completions.create(
    model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
    messages=[
        {
            "role": "system",
            "content": (
                "You extract staffing requirements from construction RFPs. "
                "Always respond with valid JSON: {\"requirements\": [...]} and nothing else."
            ),
        },
        {"role": "user", "content": prompt},
    ],
    temperature=0,
    max_tokens=900,
)

raw = resp.choices[0].message.content or ""
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    return _fallback_requirements(project_text)


        reqs = data.get("requirements", [])
        if not isinstance(reqs, list) or not reqs:
            return _fallback_requirements(project_text)

        def _good_label(lbl: str) -> bool:
            words = [w for w in lbl.strip().split() if w]
            # must be multi-word and not purely a location / owner style thing
            return len(words) >= 3

        cleaned: List[Dict[str, Any]] = []
        for i, r in enumerate(reqs, start=1):
            if not isinstance(r, dict):
                continue
            label = str(r.get("label") or "").strip()
            if not label or not _good_label(label):
                continue  # drop junk like "State", "North", etc.
            cleaned.append(
                {
                    "id": r.get("id") or f"S{i}",
                    "label": label,
                    "description": r.get("description") or "",
                    "importance": int(r.get("importance") or 2),
                }
            )

        return cleaned or _fallback_requirements(project_text)

    except Exception:
        return _fallback_requirements(project_text)


# ---------- Resume scoring ----------

def score_resume_against_requirements(requirements: List[Dict[str, Any]], resume_text: str) -> Dict[str, Any]:
    client = _get_client()
    if not client:
        return _fallback_resume_score(requirements, resume_text)

    req_json = json.dumps(requirements)

    prompt = f"""
You are matching a construction employee resume to project requirements.

Project requirements (JSON array of requirements objects):
{req_json}

Employee resume:
\"\"\"{resume_text}\"\"\"

For EACH requirement, decide:
- match_status: "strong_match", "partial_match", or "no_match"
- evidence_snippet: short phrase or sentence from the resume that supports your decision (if any).

Then compute an overall skill_match_pct from 0–100, where:
- strong_match ≈ full credit
- partial_match ≈ half credit
- no_match ≈ zero credit,
weighted by importance (3=critical, 2=important, 1=nice to have).

Respond as JSON with:
- "per_skill": list of {{id, label, match_status, evidence_snippet}}
- "skill_match_pct": number
Do not include any text outside the JSON.
"""

    try:
       resp = client.chat.completions.create(
    model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
    messages=[
        {
            "role": "system",
            "content": "You evaluate resumes against project requirements and respond ONLY with JSON.",
        },
        {"role": "user", "content": prompt},
    ],
    temperature=0,
    max_tokens=900,
)

raw = resp.choices[0].message.content or ""
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    return _fallback_resume_score(requirements, resume_text)

if "per_skill" not in data or "skill_match_pct" not in data:
    return _fallback_resume_score(requirements, resume_text)

return data


    except Exception:
        return _fallback_resume_score(requirements, resume_text)
