"""
skills_backend.py
Project requirements ↔ resume matching using OpenAI.

Two main entrypoints:

- extract_project_requirements(project_text: str) -> list[dict]
- score_resume_against_requirements(requirements: list[dict], resume_text: str) -> dict
"""

import json
import os
import re
from typing import List, Dict, Any

from openai import OpenAI

# =========================
# OpenAI client setup
# =========================

def _get_api_key() -> str | None:
    """
    Try to pull OPENAI_API_KEY from Streamlit secrets first, then env vars.
    This matches how app.py is configured.
    """
    # Streamlit may not be available when running tests, so guard import.
    try:
        import streamlit as st  # type: ignore
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    # Fallback: plain environment variable
    return os.getenv("OPENAI_API_KEY")


API_KEY = _get_api_key()
client = OpenAI(api_key=API_KEY) if API_KEY else None


# =========================
# Fallback helper functions
# =========================

def _fallback_requirements(project_text: str) -> List[Dict[str, Any]]:
    """
    Very simple backup if the LLM call or JSON parsing fails.
    We just grab some keywords from the project text and treat them
    as "requirements" so the rest of the pipeline can still run.
    """
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
                "importance": 2,  # treat as "important"
            }
        )
    return reqs


def _fallback_resume_score(requirements: List[Dict[str, Any]], resume_text: str) -> Dict[str, Any]:
    """
    Simple heuristic scoring used if the LLM output cannot be parsed.
    Looks for requirement label words in the resume text.
    """
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
            # crude: if phrase appears, call it a strong match
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

    if total_weight <= 0:
        skill_pct = 0.0
    else:
        skill_pct = (got_weight / total_weight) * 100.0

    return {
        "per_skill": per_skill,
        "skill_match_pct": skill_pct,
    }


# =========================
# Project requirement extraction
# =========================

def extract_project_requirements(project_text: str) -> List[Dict[str, Any]]:
    """
    Read the RFP / project description and return a list of key requirements.

    Intended shape:
    [
        {"id": "S1", "label": "...", "description": "...", "importance": 1-3},
        ...
    ]

    This function is defensive: if the LLM call fails or returns invalid JSON,
    it falls back to a simple keyword-based requirements list.
    """

    # If we have no API key or no text, just use fallback
    if not client or not project_text or not project_text.strip():
        return _fallback_requirements(project_text)

    prompt = f"""
You are assisting with construction staffing for project pursuits.

Given this project description, identify the 5–10 most important skills,
credentials, or experience requirements that are explicitly or implicitly requested.

For each requirement, return:
- id: a short ID like "S1", "S2", ...
- label: 4–10 word label
- description: 1–2 sentence explanation
- importance: 1 (nice to have), 2 (important), 3 (critical)

Project description:
\"\"\"{project_text}\"\"\"

Respond as JSON with a single key "requirements" whose value is a list of objects.
"""

    try:
        resp = client.responses.create(
            model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
            input=[
                {
                    "role": "system",
                    "content": (
                        "You extract staffing requirements from construction RFPs. "
                        "Always respond with valid JSON: "
                        "{\"requirements\": [...]} and nothing else."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_output_tokens=700,
        )

        raw = resp.output[0].content[0].text or ""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Model responded but not in clean JSON
            return _fallback_requirements(project_text)

        reqs = data.get("requirements", [])
        if not isinstance(reqs, list) or not reqs:
            return _fallback_requirements(project_text)

        # Ensure each requirement has expected keys; fill in defaults if missing
        cleaned: List[Dict[str, Any]] = []
        for i, r in enumerate(reqs, start=1):
            if not isinstance(r, dict):
                continue
            cleaned.append(
                {
                    "id": r.get("id") or f"S{i}",
                    "label": r.get("label") or f"Requirement {i}",
                    "description": r.get("description") or "",
                    "importance": int(r.get("importance") or 2),
                }
            )
        return cleaned or _fallback_requirements(project_text)

    except Exception:
        # Any API or network error → fallback
        return _fallback_requirements(project_text)


# =========================
# Resume scoring
# =========================

def score_resume_against_requirements(requirements: List[Dict[str, Any]], resume_text: str) -> Dict[str, Any]:
    """
    For a single resume, decide strong / partial / no match for each requirement.

    Returns:
    {
        "per_skill": [
            {"id": "...", "label": "...", "match_status": "strong_match"|"partial_match"|"no_match",
             "evidence_snippet": "..."},
            ...
        ],
        "skill_match_pct": float
    }

    This is also defensive: if the LLM output cannot be parsed as JSON, we fall back
    to a simple keyword-based matcher so the app still produces useful results.
    """

    # If we have no client, go straight to fallback
    if not client:
        return _fallback_resume_score(requirements, resume_text)

    req_json = json.dumps(requirements)

    prompt = f"""
You are matching a construction employee resume to project requirements.

Project requirements (JSON array of requirements):
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
"""

    try:
        resp = client.responses.create(
            model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
            input=[
                {
                    "role": "system",
                    "content": "You evaluate resumes against project requirements and respond ONLY with JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_output_tokens=900,
        )

        raw = resp.output[0].content[0].text or ""

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return _fallback_resume_score(requirements, resume_text)

        # Validate minimal structure
        if "per_skill" not in data or "skill_match_pct" not in data:
            return _fallback_resume_score(requirements, resume_text)

        return data

    except Exception:
        return _fallback_resume_score(requirements, resume_text)
