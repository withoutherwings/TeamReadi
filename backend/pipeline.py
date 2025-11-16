"""TeamReadi backend pipeline.

LLM-powered helpers:

- build_project_profile(project_text)
- build_candidate_profile(resume_text, project_profile)
- compute_skill_match(project_must_have, candidate_skills)
"""

import os
import json
from typing import Dict, List, Any, Set

from openai import OpenAI

USE_LLM = False  # <-- flip to True only when you want real runs

def _llm_json(prompt: str) -> dict:
    """
    Call OpenAI to get structured JSON. When USE_LLM is False,
    return a cheap stub so we don't burn tokens while debugging.
    """
    if not USE_LLM:
        # Minimal stub so the rest of the code doesn't crash
        return {
            "project_summary": "",
            "must_have_skills": [],
            "nice_to_have_skills": [],
            "role_mix_by_bucket": {
                "PM/Admin": 1,
                "Support/Coordination": 1,
                "Field/Operator": 1,
            },
        }

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

        # Strip possible markdown code fences
        if text.startswith("```"):
            text = text.strip()
            lines = text.splitlines()
            if len(lines) >= 2:
                if lines[0].lstrip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines)

        data = json.loads(text)
        if not isinstance(data, dict):
            return {}
        return data

    except Exception as e:
        print(f"[TeamReadi] _llm_json error: {e}")
        return {}


# ---------------------------------------------------------------------------
# Project profile
# ---------------------------------------------------------------------------

def build_project_profile(project_text: str) -> Dict[str, Any]:
    """
    Parse the RFP / project description into a project profile:

    {
      "project_summary": str,
      "must_have_skills": [str, ...],
      "nice_to_have_skills": [str, ...],
      "role_mix_by_bucket": {bucket: int}
    }
    """
    trimmed = (project_text or "").strip()
    if not trimmed:
        return {
            "project_summary": "",
            "must_have_skills": [],
            "nice_to_have_skills": [],
            "role_mix_by_bucket": {},
        }

    trimmed = trimmed[:8000]

    prompt = f"""
You are helping a construction project team staff a role.

You will be given the full text of an RFP / project description.

1. Read the RFP carefully.
2. Write:
   - "project_summary": THREE paragraphs, separated by a blank line:
       * Paragraph 1: a concise, factual overview for staffing decisions:
         owner, facility, city/state, contract type (if given), and the total
         expected performance period or construction duration, using dates
         or approximate months if stated. Avoid marketing or funding language.
       * Paragraph 2: the key scope and technical requirements (major systems,
         phases, deliverables, constraints, and coordination requirements).
       * Paragraph 3: role-specific requirements for the construction manager /
         project manager and key team members, including any required
         certifications, SDVOSB or other set-aside status, bonding requirements,
         and federal or regulatory frameworks (e.g., FAR, VAAR, VA, EHRM).
   - "must_have_skills": 5–10 SHORT phrases for truly essential skills or
     experience that are clearly required by this RFP (e.g., "FAR compliance",
     "federal VA project experience", "bid guarantees and performance bonds").
   - "nice_to_have_skills": 3–8 SHORT phrases that are helpful but not strictly
     required.
   - "role_mix_by_bucket": an object whose keys are the fixed buckets
     "PM/Admin", "Support/Coordination", and "Field/Operator", and whose
     values are integers estimating how many people in each bucket the project
     will realistically need at peak
     (e.g. {{"PM/Admin": 1, "Support/Coordination": 1, "Field/Operator": 2}}).

Return ONLY a valid JSON object with exactly these keys:
"project_summary", "must_have_skills", "nice_to_have_skills",
"role_mix_by_bucket".

RFP TEXT:
---
{trimmed}
---
"""
    data = _llm_json(prompt) or {}
    if not isinstance(data, dict):
        data = {}

    # --- project_summary: coerce list -> string safely ---
    raw_summary = data.get("project_summary", "")
    if isinstance(raw_summary, list):
        raw_summary = "\n\n".join(
            str(p).strip() for p in raw_summary if str(p).strip()
        )
    project_summary = str(raw_summary or "").strip()

    # --- must_have_skills: ensure list of strings ---
    raw_must = data.get("must_have_skills") or []
    if isinstance(raw_must, str):
        raw_must = [raw_must]
    must_have = [str(s).strip() for s in raw_must if str(s).strip()]

    # --- nice_to_have_skills: ensure list of strings ---
    raw_nice = data.get("nice_to_have_skills") or []
    if isinstance(raw_nice, str):
        raw_nice = [raw_nice]
    nice_to_have = [str(s).strip() for s in raw_nice if str(s).strip()]

    # --- role mix: make sure it’s a dict ---
    role_mix = data.get("role_mix_by_bucket") or {}
    if not isinstance(role_mix, dict):
        role_mix = {}

    return {
        "project_summary": project_summary,
        "must_have_skills": must_have,
        "nice_to_have_skills": nice_to_have,
        "role_mix_by_bucket": role_mix,
    }


# ---------------------------------------------------------------------------
# Candidate profile
# ---------------------------------------------------------------------------

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
    trimmed_resume = (resume_text or "").strip()[:8000]

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
   - Treat organizational set-aside conditions (e.g., SDVOSB certification,
     woman-owned business status, bonding capacity, etc.) as attributes of the
     bidding firm, not the individual employee. Do NOT mark these as "met" or
     "missing" for a candidate unless the resume explicitly shows that this
     person owns or leads such a certified business.
   - You may still mention these firm-level conditions in the project context
     in the candidate_summary if it helps explain overall fit, but do not count
     them as individual strengths or gaps.
2. Decide which MUST-HAVE skills are clearly missing or too weak at the
   individual level (e.g., no federal VA project experience, no similar
   infrastructure or EHRM background, no relevant schedule/cost control).
3. Extract 10–20 SHORT "candidate_skills" phrases that describe this candidate,
   focusing on project-relevant hard and soft skills.
4. Write "candidate_summary": 1–3 sentences about how well this person fits
   this specific project.
5. Write 3–6 SHORT bullet phrases under "strengths" focused on this project.
6. Write 3–6 SHORT bullet phrases under "gaps" focused on this project.
7. Compute "skill_match_percent" = 100 * (# MUST-HAVE skills met) /
   (total MUST-HAVE).

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
    if not isinstance(data, dict):
        data = {}

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
