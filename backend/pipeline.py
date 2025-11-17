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

USE_LLM = True  # <-- flip to True only when you want real runs

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

# ---- Heuristics to separate company-level vs person-level requirements ----

COMPANY_LEVEL_KEYWORDS = [
    "sdvosb",
    "service-disabled veteran",
    "service disabled veteran",
    "woman-owned",
    "women owned",
    "wosb",
    "hubzone",
    "8(a)",
    "8a ",
    "small business set-aside",
    "set-aside",
    "dbe ",
    "mbe ",
    "wbe ",
    "far compliance",
    "federal acquisition regulation",
    "far part",
    "cage code",
    "sam.gov",
    "sam registration",
    "bonding capacity",
    "insurance coverage",
]


def _is_company_level_requirement(text: str) -> bool:
    t = str(text).lower()
    return any(kw in t for kw in COMPANY_LEVEL_KEYWORDS)


def _split_person_vs_company(requirements) -> tuple[list[str], list[str]]:
    person: list[str] = []
    company: list[str] = []
    for r in requirements or []:
        s = str(r).strip()
        if not s:
            continue
        if _is_company_level_requirement(s):
            company.append(s)
        else:
            person.append(s)
    return person, company


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

from typing import Dict, Any  # make sure this import exists at top


def build_project_profile(job_text: str) -> Dict[str, Any]:
    """
    Use the LLM (or a stub when USE_LLM is False) to summarize the RFP and
    extract structured fields for the rest of the pipeline.

    We intentionally keep *person-level* skill requirements separate from
    company-level business/certification requirements so that candidates
    are not penalized for SDVOSB status, FAR compliance, etc.

    NEW: we ask for three clearly separated summary paragraphs:
      - p1 = overview (what / where / why)
      - p2 = owner / company-level constraints & requirements
      - p3 = what the project team / key personnel will actually do
    """
    if not job_text:
        return {
            "project_name": "",
            "project_summary": "",
            "project_window": "",
            "project_location": "",
            "must_have_skills": [],
            "nice_to_have_skills": [],
            "company_requirements": [],
            "role_mix_by_bucket": {},
        }

    prompt = f"""
You are analyzing a construction RFP and turning it into structured JSON
for an internal workforce-planning tool.

RFP TEXT (truncated if very long):
---
{job_text[:12000]}
---

Extract the following fields:

1. "project_name": a SHORT human-readable project name
   (e.g. "EHRM Outpatient Clinic Renovation – Asheville VAMC").
   If no clear name is given, synthesize a concise descriptive name.

2. "project_summary_p1": one paragraph giving a high-level overview:
   - what the project is,
   - where it is located,
   - the primary purpose or goal of the work,
   - any key systems or disciplines (e.g. EHRM, electrical, HVAC).

3. "project_summary_p2": one paragraph focused on OWNER / COMPANY-LEVEL
   requirements and constraints that affect who can win the work:
   - small business or SDVOSB/8(a)/HUBZone set-asides,
   - important FAR / VA / DoD compliance issues,
   - bonding, insurance, or other firm-level requirements.
   This should read like a narrative paragraph, NOT a bullet list.

4. "project_summary_p3": one paragraph focused on what the PROJECT TEAM
   and key personnel will actually be responsible for:
   - major construction tasks and phases,
   - coordination with the owner or facility (e.g. working in an active VAMC),
   - any critical schedule, phasing, or access constraints.

5. "project_window": a brief description of the expected duration / period
   of performance as described in the RFP
   (e.g. "Design NTP Feb 2026; substantial completion by Oct 2027").
   If not stated, use "Not clearly stated in RFP."

6. "project_location": city/state or facility name if given
   (e.g. "Chicago, IL – Jesse Brown VAMC").

7. "must_have_skills": an array of SHORT, PERSON-LEVEL skill or experience
   requirements for individual team members.
   - DO NOT include company-level business requirements such as:
     SDVOSB / WOSB / 8(a) / HUBZone status, small-business set-asides,
     bonding capacity, insurance limits, or generic "FAR compliance".

8. "nice_to_have_skills": similar array of PERSON-LEVEL nice-to-have skills.
   (Same rule: ignore company-level certifications and ownership status.)

9. "company_requirements": an array of SHORT items describing OWNER /
   COMPANY-LEVEL requirements (set-asides, SDVOSB, FAR clauses, bonding,
   VA-specific rules, etc.).

10. "role_mix_by_bucket": an object with integer counts estimating how many
    people the OWNER will realistically need at peak in each of three buckets:
      - "PM/Admin"
      - "Support/Coordination"
      - "Field/Operator"
    Example: {{"PM/Admin": 1, "Support/Coordination": 1, "Field/Operator": 2}}

Return ONLY a valid JSON object with exactly these keys:
"project_name",
"project_summary_p1", "project_summary_p2", "project_summary_p3",
"project_window", "project_location",
"must_have_skills", "nice_to_have_skills",
"company_requirements",
"role_mix_by_bucket".
"""

    data = _llm_json(prompt) or {}

    # ---- Person-level vs company-level skills ----
    raw_must = data.get("must_have_skills") or []
    raw_nice = data.get("nice_to_have_skills") or []

    must_person, must_company = _split_person_vs_company(raw_must)
    nice_person, nice_company = _split_person_vs_company(raw_nice)

    # Explicit company requirements from the JSON (may overlap with split output)
    explicit_company = data.get("company_requirements") or []

    # Clean + de-duplicate company-level requirements
    company_reqs: list[str] = []
    for item in list(must_company + nice_company + explicit_company):
        s = str(item).strip()
        if s and s not in company_reqs:
            company_reqs.append(s)

    # ---- Build final multi-paragraph summary ----
    p1 = str(data.get("project_summary_p1") or "").strip()
    p2 = str(data.get("project_summary_p2") or "").strip()
    p3 = str(data.get("project_summary_p3") or "").strip()

    if any([p1, p2, p3]):
        pieces = [p for p in (p1, p2, p3) if p]
        project_summary = "\n\n".join(pieces)
    else:
        # Fallback if the model ignored the new keys
        raw_summary = data.get("project_summary", "")
        if isinstance(raw_summary, list):
            project_summary = " ".join(str(x) for x in raw_summary)
        else:
            project_summary = str(raw_summary)

    profile: Dict[str, Any] = {
        "project_name": str(data.get("project_name") or "").strip(),
        "project_summary": project_summary.strip(),
        "project_window": str(data.get("project_window") or "").strip(),
        "project_location": str(data.get("project_location") or "").strip(),
        # Only person-level items feed into skill matching and highlights:
        "must_have_skills": must_person,
        "nice_to_have_skills": nice_person,
        # Company-level requirements kept for the project summary page:
        "company_requirements": company_reqs,
        "role_mix_by_bucket": data.get("role_mix_by_bucket") or {},
    }
    return profile


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
