"""
skills_backend.py
Project requirements ↔ resume matching using OpenAI.
"""

import json
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_project_requirements(project_text: str) -> list[dict]:
    """
    Read the RFP / project description and return a list of key requirements.

    Returns a list like:
    [
        {"id": "S1", "label": "...", "description": "...", "importance": 1-3},
        ...
    ]
    """
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

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": "You extract staffing requirements from construction RFPs."},
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=700,
    )

    raw = resp.output[0].content[0].text
    data = json.loads(raw)
    return data["requirements"]


def score_resume_against_requirements(requirements: list[dict], resume_text: str) -> dict:
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
    """
    req_json = json.dumps(requirements)

    prompt = f"""
You are matching a construction employee resume to project requirements.

Project requirements (JSON):
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

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": "You evaluate resumes against project requirements."},
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=900,
    )

    raw = resp.output[0].content[0].text
    data = json.loads(raw)
    return data
