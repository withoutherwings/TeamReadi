"""TeamReadi backend pipeline.

LLM-powered helpers:

- build_project_profile(project_text)
- build_candidate_profile(resume_text, project_profile)
- compute_skill_match(project_must_have, candidate_skills)

We *don’t* rely on OpenAI JSON mode here, because the current library
version is routing through Responses.create in a way that doesn’t accept
`response_format`. Instead, we enforce JSON in the prompt and parse it
manually, with safe fallbacks.
"""

import os
import json
from typing import Dict, List, Any, Set

from openai import OpenAI

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

client = OpenAI(api_key=API_KEY)


# ---------------------------------------------------------------------------
# Simple helpers
# ---------------------------------------------------------------------------

def _normalize_phrase_list(items: List[str]) -> Set[str]:
    return {str(x).strip().lower() for x in (items or []) if str(x).strip()}


def compute_skill_match(project_must_have: List[str], candidate_skills: List[str]) -> float:
    """Fallback overlap-based score, 0–100."""
    proj_set = _normalize_phrase_list(project_must_have)
    cand_set = _normalize_phrase_list(candidate_skills)
    if not proj_set:
        return 0.0
    overlap = proj_set & cand_set
    pct = 100.0 * len(overlap) / len(proj_set)
    return round(pct, 1)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _llm_json(prompt: str) -> Dict[str, Any]:
    """
    Call the chat model and *attempt* to parse JSON from the response.

    We tell the model to return JSON only. If parsing fails, we fall back
    to an empty dict and let callers handle defaults.
    """
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
        )
        text = resp.choices[0].message.content.strip()
        # Strip possible markdown fences
        if text.startswith("```"):
            text = text.strip("`")
            # after stripping backticks, there may still be leading "json\n"
            if text.lower().startswith("json"):
                text = text.split("\n", 1)[1]
        return json.loads(text)
    except Exception as e:
        print(f"[TeamReadi] _llm_json error: {e}")
        return {}


def build_project_profile(project_text: str) -> Dict[str, Any]:
    """
    Parse the RFP / project description into a project profile:

    {
      "project_summary": str,
      "must_have_skills": [str, ...],
      "nice_to_have_skills": [str, ...],
    }
    """
    trimmed = (project_text or "").strip()
    if not trimmed:
        return {
            "project_summary": "",
            "must_have_skills": [],
            "nice_to_have_skills": [],
        }

    trimmed = trimmed[:8000]

    prompt = f"""
You are helping a construction project team staff a role.

You will be given the full text of an RFP / project description.

1. Read the RFP carefully.
2. Write:
   - "project_summary": 3–5 sentences describing scope, context, major tasks.
   - "must_have_skills": 5–10 SHORT phrases for truly essential skills/experience.
   - "nice_to_have_skills": 3–8 SHORT phrases that are helpful but not required.

Return ONLY a valid JSON object with exactly these keys:
"project_summary", "must_have_skills", "nice_to_have_skills".

RFP TEXT:
---
{trimmed}
---
"""
    data = _llm_json(prompt) or {}

    project_summary = (data.get("project_summary") or "").strip()
    must_have = [str(s).strip() for s in (data.get("must_have_skills") or []) if str(s).strip()]
    nice_to_have = [str(s).strip() for s in (data.get("nice_to_have_skills") or []) if str(s).strip()]

    return {
        "project_summary": project_summary,
        "must_have_skills": must_have,
        "nice_to_have_skills": nice_to_have,
    }


def build_candidate_profile(resume_text: str, project_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate one candidate resume against the project profile.

    Return:
    {
      "candidate_summary": str,
      "candidate_skills": [str, ...],
      "strengths": [str, ...],
      "gaps": [str, ...],
      "matched_must_have_skills": [str, ...],
      "missing_must_have_skills": [str, ...],
      "skill_match_percent": float
    }
    """
    trimmed_resume = (resume_text or "").strip()
    trimmed_resume = trimmed_resume[:8000]

    project_summary = project_profile.get("project_summary", "")
    must_have_skills = project_profile.get("must_have_skills", []) or []
    nice_to_have_skills = project_profile.get("nice_to_have_skills", []) or []

    prompt = f"""
You are evaluating a construction professional for a specific project.

You are given:

PROJECT SUMMARY:
{project_summary}

MUST-HAVE SKILLS:
{must_have_skills}

NICE-TO-HAVE SKILLS:
{nice_to_have_skills}

CANDIDATE RESUME:
---
{trimmed_resume}
---

Your tasks:

1. Decide which MUST-HAVE skills the candidate clearly meets.
2. Decide which MUST-HAVE skills are clearly missing or too weak.
3. Extract 10–20 SHORT "candidate_skills" phrases that describe this candidate,
   focusing on project-relevant hard and soft skills.
4. Write "candidate_summary": 1–3 sentences about how well this person fits this project.
5. Write 3–6 SHORT bullet phrases under "strengths" focused on this project.
6. Write 3–6 SHORT bullet phrases under "gaps" focused on this project.
7. Compute "skill_match_percent" = 100 * (# MUST-HAVE skills met) / (total MUST-HAVE).

Return ONLY a JSON object with keys:
"candidate_summary",
"candidate_skills",
"strengths",
"gaps",
"matched_must_have_skills",
"missing_must_have_skills",
"skill_match_percent".
"""
    data = _llm_json(prompt) or {}

    candidate_summary = (data.get("candidate_summary") or "").strip()
    candidate_skills = [str(s).strip() for s in (data.get("candidate_skills") or []) if str(s).strip()]
    strengths = [str(s).strip() for s in (data.get("strengths") or []) if str(s).strip()]
    gaps = [str(s).strip() for s in (data.get("gaps") or []) if str(s).strip()]
    matched = [str(s).strip() for s in (data.get("matched_must_have_skills") or []) if str(s).strip()]
    missing = [str(s).strip() for s in (data.get("missing_must_have_skills") or []) if str(s).strip()]
    skill_match_percent = data.get("skill_match_percent")

    # Robust fallback for score
    try:
        skill_match_percent = float(skill_match_percent)
    except (TypeError, ValueError):
        if must_have_skills:
            matched_norm = _normalize_phrase_list(matched)
            must_norm = _normalize_phrase_list(must_have_skills)
            pct = 100.0 * len(matched_norm & must_norm) / max(1, len(must_norm))
            skill_match_percent = round(pct, 1)
        else:
            skill_match_percent = compute_skill_match(must_have_skills, candidate_skills)

    return {
        "candidate_summary": candidate_summary,
        "candidate_skills": candidate_skills,
        "strengths": strengths,
        "gaps": gaps,
        "matched_must_have_skills": matched,
        "missing_must_have_skills": missing,
        "skill_match_percent": float(skill_match_percent),
    }
