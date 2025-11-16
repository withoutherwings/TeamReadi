"""
pipeline.py
High-level TeamReadi pipeline:
- Project text + resumes -> semantic skill_match results via LLM + embeddings
- Calendar -> availability
- Combine into ReadiScore and sorted ranking
"""

import os
import json
import math
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

# Model for LLM reasoning
MODEL_NAME = os.getenv("TEAMREADI_MODEL_NAME", "gpt-4.1-mini")

# Model for semantic similarity
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

# Similarity threshold for treating a must-have skill as "matched"
SEMANTIC_MATCH_THRESHOLD = float(os.getenv("TEAMREADI_MATCH_THRESHOLD", "0.63"))


# ----------------- Generic helpers -----------------


def _normalize_phrase_list(skills: List[str]) -> Set[str]:
    """
    Simple normalization: lowercase, strip, remove duplicates.
    Used only as a fallback when embeddings fail.
    """
    norm = set()
    for s in skills:
        if not isinstance(s, str):
            continue
        tokens = s.lower().strip()
        if tokens:
            norm.add(tokens)
    return norm


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for a list of short phrases.
    Returns a list of vectors in the same order.
    """
    if not texts:
        return []
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _compute_skill_match_lexical(
    project_must_have: List[str],
    candidate_skills: List[str],
) -> float:
    """
    Fallback: lexical overlap only. Used if embeddings fail completely.
    """
    proj_set = _normalize_phrase_list(project_must_have)
    cand_set = _normalize_phrase_list(candidate_skills)

    if not proj_set:
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

    NOTE: We intentionally allow a large chunk of text so we don't get stuck
    on just the first page of a long RFP.
    """
    if not project_text:
        trimmed_text = ""
    else:
        # Hard cap only to avoid absurdly huge prompts; 60k chars ~ many pages.
        trimmed_text = project_text[:60000]

    prompt = f"""
You are helping a construction management team staff a project.

You are given the text of a project RFP / description. Read it carefully
and output a JSON object with:

- "project_summary": 3–5 sentences describing the core scope, context, and key deliverables.
- "must_have_skills": 5–10 short skill or experience phrases that are required to succeed.
- "nice_to_have_skills": 3–8 short skill or experience phrases that are beneficial but not mandatory.

Important:
- Focus on the technical and managerial requirements of the ROLE(S),
  not just generic proposal submittal instructions.
- Infer implied skills where appropriate. For example, if the RFP calls for
  managing multiple renovation projects in occupied facilities, you may infer
  skills like "phased construction in occupied facilities" or "stakeholder coordination."
- Keep each skill phrase short (not full sentences) and grounded in the RFP.

RFP TEXT:
\"\"\"{trimmed_text}\"\"\""""

    try:
        resp = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            response_format={"type": "json_object"},
        )
        raw_text = resp.output[0].content[0].text
        data = json.loads(raw_text)
    except Exception as e:
        print(f"[TeamReadi] build_project_profile LLM error: {e}")
        data = {
            "project_summary": trimmed_text[:500],
            "must_have_skills": [],
            "nice_to_have_skills": [],
        }

    project_summary = (data.get("project_summary") or "").strip()
    must_have = data.get("must_have_skills") or []
    nice_to_have = data.get("nice_to_have_skills") or []

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
    - strengths (3–6 phrases tied to project needs)
    - gaps (3–6 phrases tied to project needs)
    """
    if not resume_text:
        trimmed_resume = ""
    else:
        # Resumes are small; this cap is generous.
        trimmed_resume = resume_text[:16000]

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
  (grounded in the resume, not hallucinated; include tools, domains, and responsibilities).
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


# ----------------- Skill results builder (semantic) -----------------


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
            "highlights": [{"skill": str, "met": bool, ...}, ...],
            "candidate_summary": str,
            "strengths": [str, ...],
            "gaps": [str, ...],
            "candidate_skills": [str, ...],
            "project_profile": {...}  # same object for all employees
        },
        ...
    }

    We keep "skill_match_pct" and "highlights" compatible with the existing
    Results page & PDF logic, but now they are based on semantic similarity.
    """
    # 1) Derive a structured project profile once
    project_profile = build_project_profile(project_text)
    proj_must = project_profile.get("must_have_skills", []) or []

    # Pre-compute embeddings for project must-have skills
    try:
        proj_embeds = _embed_texts(proj_must) if proj_must else []
        embedding_ok = bool(proj_embeds)
    except Exception as e:
        print(f"[TeamReadi] project embedding error, falling back to lexical: {e}")
        proj_embeds = []
        embedding_ok = False

    results: Dict[str, dict] = {}

    for emp_id, resume_text in resumes.items():
        # 2) Build a candidate profile for each resume
        cand_profile = build_candidate_profile(resume_text, project_profile)
        cand_skills = cand_profile.get("candidate_skills", []) or []

        # 3) If embeddings are available, do semantic matching; otherwise lexical.
        highlights = []
        skill_match_pct = 0.0

        if proj_must and cand_skills and embedding_ok:
            try:
                cand_embeds = _embed_texts(cand_skills)

                matched_count = 0
                for i, skill_label in enumerate(proj_must):
                    s_label = str(skill_label).strip()
                    if not s_label:
                        continue

                    best_sim = 0.0
                    best_match = None

                    for j, cand_vec in enumerate(cand_embeds):
                        sim = _cosine_similarity(proj_embeds[i], cand_vec)
                        if sim > best_sim:
                            best_sim = sim
                            best_match = cand_skills[j]

                    met = best_sim >= SEMANTIC_MATCH_THRESHOLD
                    if met:
                        matched_count += 1

                    # We keep "skill" and "met" for compatibility; extra fields are optional.
                    highlights.append(
                        {
                            "skill": s_label,
                            "met": met,
                            "best_match": best_match,
                            "similarity": round(best_sim, 3),
                        }
                    )

                if proj_must:
                    skill_match_pct = round(
                        100.0 * matched_count / len(proj_must), 1
                    )
            except Exception as e:
                # If something goes wrong with embeddings for this candidate, fall back to lexical.
                print(f"[TeamReadi] candidate embedding error, fallback to lexical: {e}")
                skill_match_pct = _compute_skill_match_lexical(proj_must, cand_skills)
                cand_set = _normalize_phrase_list(cand_skills)
                for s in proj_must:
                    label = str(s).strip()
                    if not label:
                        continue
                    met = label.lower().strip() in cand_set
                    highlights.append({"skill": label, "met": met})
        else:
            # No embeddings or no skills; lexical fallback.
            skill_match_pct = _compute_skill_match_lexical(proj_must, cand_skills)
            cand_set = _normalize_phrase_list(cand_skills)
            for s in proj_must:
                label = str(s).strip()
                if not label:
                    continue
                met = label.lower().strip() in cand_set
                highlights.append({"skill": label, "met": met})

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
    - derive semantic skill match from project + resumes via LLM + embeddings
    - derive availability from calendar
    - compute ReadiScore
    - return ranked list of employees

    Returned structure:
    [
        {
            "employee_id": <str>,
            "metrics": <dict from compute_metrics_for_employee>,
            "highlights": [{"skill": str, "met": bool, ...}, ...],
            "llm_profile": {
                "candidate_summary": str,
                "strengths": [str, ...],
                "gaps": [str, ...],
                "candidate_skills": [str, ...],
                "project_profile": {...}
            },
        },
        ...
    ]
    """
    # 1) Skill side (now LLM + embeddings)
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

    return rows_sorted
