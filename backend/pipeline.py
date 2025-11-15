"""
pipeline.py
High-level TeamReadi pipeline:
- Project text + resumes -> skill_match results via LLM
- Calendar -> availability
- Combine into ReadiScore and sorted ranking
"""

import os
import json
import datetime as dt
from typing import Dict, List, Any, Set

from openai import OpenAI

from backend.calendar_backend import (
    fetch_calendar_hours_by_employee,
    compute_metrics_for_employee,
)

# ----------------- OpenAI client setup -----------------

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Please configure it in your environment or Streamlit secrets."
    )

client = OpenAI(api_key=API_KEY)

# You can adjust this if you want to use a different model
MODEL_NAME = os.getenv("TEAMREADI_MODEL_NAME", "gpt-4.1-mini")


# ----------------- Helper functions -----------------

def _normalize_phrase_list(skills: List[str]) -> Set[str]:
    """
    Simple normalization: lowercase, strip, remove duplicates.
    This is intentionally simple; we can upgrade to embeddings later.
    """
    norm = set()
    for s in skills:
        if not isinstance(s, str):
            continue
        tokens = s.lower().strip()
        if tokens:
            norm.add(tokens)
    return norm


def compute_skill_match(
    project_must_have: List[str],
    candidate_skills: List[str],
) -> float:
    """
    Return a percentage [0,100] representing how many must-have skills
    the candidate appears to cover. Very simple overlap for now
    (string-level, after normalization).
    """
    proj_set = _normalize_phrase_list(project_must_have)
    cand_set = _normalize_phrase_list(candidate_skills)

    if not proj_set:
        # Avoid division by zero; if you have no defined must-haves,
        # treat skill match as 0 for now (or adjust to 50 if you prefer neutral).
        return 0.0

    overlap = proj_set.intersection(cand_set)
    score = (len(overlap) / len(proj_set)) * 100.0
    return round(score, 1)


# ----------------- LLM-based project & candidate profiles -----------------

def build_project_profile(project_text: str) -> Dict[str, Any]:
    """
    Use the LLM to create a compact project profile:
    - short summary
    - must-have skills
    - nice-to-have skills
    This replaces the old 'extract_project_requirements' behavior.
    """
    # Truncate to keep the prompt manageable
    trimmed_text = project_text[:8000] if project_text else ""

    prompt = f"""
You are helping a construction management team staff a project.

You are given the full text of a project RFP / description. Read it carefully
and output a JSON object with:

- "project_summary": 3–5 sentences describing the core scope, context, and key deliverables.
- "must_have_skills": 5–10 short skill or experience phrases that are required to succeed.
- "nice_to_have_skills": 3–8 short skill or experience phrases that are beneficial but not mandatory.

Guidelines for the skill phrases:
- Keep them short, not full sentences.
- They should be grounded in the RFP text (do not hallucinate wildly).
- Examples:
  - "construction project scheduling (Primavera P6)"
  - "DOT facilities experience"
  - "contract administration and RFI management"
  - "field inspection of civil works"
  - "quantity takeoffs and cost estimating"

RFP TEXT:
\"\"\"{trimmed_text}\"\"\"
"""

    try:
        resp = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            response_format={"type": "json_object"},
        )
        raw_text = resp.output[0].content[0].text
        data = json.loads(raw_text)
    except Exception as e:
        # Fail safe: if LLM call fails, return something minimal
        # This will prevent the entire pipeline from exploding.
        print(f"[TeamReadi] build_project_profile LLM error: {e}")
        data = {
            "project_summary": trimmed_text[:500],
            "must_have_skills": [],
            "nice_to_have_skills": [],
        }

    project_summary = (data.get("project_summary") or "").strip()
    must_have = data.get("must_have_skills") or []
    nice_to_have = data.get("nice_to_have_skills") or []

    # Normalize lists to ensure strings
    must_have = [str(s).strip() for s in must_have if str(s).strip()]
    nice_to_have = [str(s).strip() for s in nice_to_have if str(s).strip()]

    return {
        "project_summary": project_summary,
        "must_have_skills": must_have,
        "nice_to_have_skills": nice_to_have,
    }


def build_candidate_profile(
    resume_text: str,
    project_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Produce a clean candidate profile:
    - candidate_summary (1–3 sentences)
    - candidate_skills (10–20 short phrases)
    - strengths (3–6 short phrases tied to project needs)
    - gaps (3–6 short phrases tied to project needs)
    This replaces the old 'score_resume_against_requirements' behavior.
    """
    trimmed_resume = resume_text[:8000] if resume_text else ""

    project_summary = project_profile.get("project_summary", "")
    must_have_skills = project_profile.get("must_have_skills", [])
    nice_to_have_skills = project_profile.get("nice_to_have_skills", [])

    prompt = f"""
You are evaluating a candidate for a construction project.

You are given:
1) A brief project summary and required skills.
2) A candidate resume.

PROJECT SUMMARY:
{project_summary}

MUST-HAVE SKILLS:
{must_have_skills}

NICE-TO-HAVE SKILLS:
{nice_to_have_skills}

CANDIDATE RESUME TEXT:
\"\"\"{trimmed_resume}\"\"\"


Return ONLY a JSON object with:

- "candidate_summary": 1–3 sentences summarizing the candidate's background relevant to this project.
- "candidate_skills": 10–20 short skills or experience phrases derived from the resume
  (grounded in the resume, not hallucinated).
- "strengths": 3–6 short phrases explaining where this candidate aligns WELL with the project needs.
- "gaps": 3–6 short phrases explaining what is MISSING relative to the project needs.

All items in lists must be short, human-readable phrases, not full sentences.
"""

    try:
        resp = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            response_format={"type": "json_object"},
        )
        raw_text = resp.output[0].content[0].text
        data = json.loads(raw_text)
    except Exception as e:
        print(f"[TeamReadi] build_candidate_profile LLM error: {e}")
        data = {
            "candidate_summary": "",
            "candidate_skills": [],
            "strengths": [],
            "gaps": [],
        }

    candidate_summary = (data.get("candidate_summary") or "").strip()
    candidate_skills = data.get("candidate_skills") or []
    strengths = data.get("strengths") or []
    gaps = data.get("gaps") or []

    candidate_skills = [str(s).strip() for s in candidate_skills if str(s).strip()]
    strengths = [str(s).strip() for s in strengths if str(s).strip()]
    gaps = [str(s).strip() for s in gaps if str(s).strip()]

    return {
        "candidate_summary": candidate_summary,
        "candidate_skills": candidate_skills,
        "strengths": strengths,
        "gaps": gaps,
    }


# ----------------- Skill results builder (replaces old skills_backend) -----------------

def build_skill_results(
    project_text: str,
    resumes: Dict[str, str],
) -> Dict[str, dict]:
    """
    project_text: full text of project RFP/spec
    resumes: {"Employee_001": "full resume text", ...}

    Returns dict:
    {
        "Employee_001": {
            "skill_match_pct": float,
            "highlights": [{"skill": str, "met": bool}, ...],
            "candidate_summary": str,
            "strengths": [str, ...],
            "gaps": [str, ...],
            "candidate_skills": [str, ...],
            "project_profile": {...}  # same object for all employees
        },
        ...
    }

    NOTE: We retain "skill_match_pct" and "highlights" to keep compatibility
    with the existing Results page & PDF logic, but now they come from the
    LLM-based profiles instead of the old skills_backend.
    """
    # 1) Derive a structured project profile once
    project_profile = build_project_profile(project_text)

    results: Dict[str, dict] = {}

    for emp_id, resume_text in resumes.items():
        # 2) Build a candidate profile for each resume
        cand_profile = build_candidate_profile(resume_text, project_profile)

        # 3) Compute numeric skill match using project must-haves vs candidate skills
        skill_match_pct = compute_skill_match(
            project_profile.get("must_have_skills", []),
            cand_profile.get("candidate_skills", []),
        )

        # 4) Build "highlights" list in the same shape as before:
        #    [{"skill": <must-have-skill>, "met": True/False}, ...]
        highlights = []
        proj_must = project_profile.get("must_have_skills", [])
        cand_skill_set = _normalize_phrase_list(
            cand_profile.get("candidate_skills", [])
        )

        for s in proj_must:
            skill_label = str(s).strip()
            if not skill_label:
                continue
            met = skill_label.lower().strip() in cand_skill_set
            highlights.append({"skill": skill_label, "met": met})

        results[emp_id] = {
            "skill_match_pct": float(skill_match_pct),
            "highlights": highlights,
            "candidate_summary": cand_profile["candidate_summary"],
            "strengths": cand_profile["strengths"],
            "gaps": cand_profile["gaps"],
            "candidate_skills": cand_profile["candidate_skills"],
            "project_profile": project_profile,
        }

    return results


# ----------------- Main pipeline entrypoint -----------------

def run_teamreadi_pipeline(
    calendar_url: str,
    start_date: dt.date,
    end_date: dt.date,
    working_days: set[int],
    hours_per_day: float,
    project_text: str,
    resumes: Dict[str, str],
    tz_name: str = "America/New_York",
):
    """
    Full end-to-end run:
    - derive skill match from project + resumes via LLM
    - derive availability from calendar
    - compute ReadiScore
    - return ranked list of employees

    The returned structure is compatible with the previous implementation:
    [
        {
            "employee_id": <str>,
            "metrics": <dict from compute_metrics_for_employee>,
            "highlights": [{"skill": str, "met": bool}, ...],
            "llm_profile": {...},       # NEW, extra info
        },
        ...
    ]
    """
    # 1) Skill side (now LLM-based)
    skill_results = build_skill_results(project_text, resumes)
    employee_ids = list(skill_results.keys())

    # 2) Calendar side (unchanged)
    calendar_data = fetch_calendar_hours_by_employee(
        ics_url=calendar_url,
        employee_ids=employee_ids,
        window_start=dt.datetime.combine(start_date, dt.time(0, 0)),
        window_end=dt.datetime.combine(end_date, dt.time(23, 59)),
        working_days=working_days,
        hours_per_day=hours_per_day,
        tz_name=tz_name,
    )

    # 3) Combine into metrics + ranking
    rows = []
    for emp_id in employee_ids:
        s_res = skill_results[emp_id]
        skill_match_pct = s_res["skill_match_pct"]
        highlights = s_res["highlights"]

        emp_data = calendar_data.get(
            emp_id,
            {"booked_hours": 0.0, "capacity_hours": 0.0, "events": []},
        )

        # compute_metrics_for_employee is assumed to still take (emp_data, skill_match_pct)
        metrics = compute_metrics_for_employee(emp_data, skill_match_pct)

        rows.append(
            {
                "employee_id": emp_id,
                "metrics": metrics,
                "highlights": highlights,
                # Extra LLM-based info that UI/PDF can optionally use:
                "llm_profile": {
                    "candidate_summary": s_res["candidate_summary"],
                    "strengths": s_res["strengths"],
                    "gaps": s_res["gaps"],
                    "candidate_skills": s_res["candidate_skills"],
                    "project_profile": s_res["project_profile"],
                },
            }
        )

    # 4) Sort by ReadiScore descending
    rows_sorted = sorted(
        rows, key=lambda r: r["metrics"]["readiscore"], reverse=True
    )

    # Debug hook if you need to inspect what the Results page receives:
    # print("=== DEBUG: TeamReadi pipeline output ===")
    # from pprint import pprint
    # pprint(rows_sorted)

    return rows_sorted
