"""TeamReadi backend pipeline.

This module provides a *thin* wrapper around the OpenAI API so the rest of the
app can stay simple.  The key ideas:

- `build_project_profile(project_text)`:
    Use the LLM once to turn an RFP / project description into a compact
    summary + list of must-have / nice-to-have skills.

- `build_candidate_profile(resume_text, project_profile)`:
    Use the LLM once per resume, conditioned on the project profile, to decide:
      * what skills the candidate has
      * which of the MUST-HAVE skills they meet / miss
      * a short narrative summary of their fit

- `compute_skill_match(...)`:
    Kept for backwards-compatibility, but we now prefer the LLM-supplied
    `skill_match_percent` coming out of `build_candidate_profile`.

Everything is designed so that, if the LLM call fails for any reason, we fall
back to safe defaults instead of crashing the app.
"""

import os
import json
from typing import Dict, List, Any, Set

from openai import OpenAI


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

API_KEY = os.getenv("OPENAI_API_KEY")  # Streamlit already validates this
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

client = OpenAI(api_key=API_KEY)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _normalize_phrase_list(items: List[str]) -> Set[str]:
    return {str(x).strip().lower() for x in (items or []) if str(x).strip()}


def compute_skill_match(project_must_have: List[str], candidate_skills: List[str]) -> float:
    """Simple lexical overlap score kept as a fallback.

    Returns a percentage 0–100 based on overlap between the two lists.
    """
    proj_set = _normalize_phrase_list(project_must_have)
    cand_set = _normalize_phrase_list(candidate_skills)

    if not proj_set:
        return 0.0

    overlap = proj_set.intersection(cand_set)
    score = (len(overlap) / len(proj_set)) * 100.0
    return round(score, 1)


# ---------------------------------------------------------------------------
# LLM-based project & candidate profiles
# ---------------------------------------------------------------------------

def build_project_profile(project_text: str) -> Dict[str, Any]:
    """Turn raw RFP / project text into a structured project profile.

    Output:
        {
          "project_summary": str,
          "must_have_skills": [str, ...],
          "nice_to_have_skills": [str, ...],
        }
    """
    trimmed_text = (project_text or "")[:8000]

    if not trimmed_text:
        return {
            "project_summary": "",
            "must_have_skills": [],
            "nice_to_have_skills": [],
        }

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
- Use the terminology that appears in the RFP when possible.

RFP TEXT (verbatim):
{trimmed_text}

Return ONLY valid JSON with keys exactly:
"project_summary", "must_have_skills", "nice_to_have_skills".
"""  # noqa: E501
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw_text = resp.choices[0].message.content
        data = json.loads(raw_text)
    except Exception as e:  # defensive fallback
        print(f"[TeamReadi] build_project_profile LLM error: {e}")
        data = {
            "project_summary": trimmed_text[:500],
            "must_have_skills": [],
            "nice_to_have_skills": [],
        }

    project_summary = (data.get("project_summary") or "").strip()
    must_have = [str(s).strip() for s in (data.get("must_have_skills") or []) if str(s).strip()]
    nice_to_have = [str(s).strip() for s in (data.get("nice_to_have_skills") or []) if str(s).strip()]

    return {
        "project_summary": project_summary,
        "must_have_skills": must_have,
        "nice_to_have_skills": nice_to_have,
    }


def build_candidate_profile(resume_text: str, project_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate ONE candidate resume against the project.

    Inputs:
        resume_text: full text of a single candidate CV / resume
        project_profile: output from `build_project_profile`

    Output JSON (keys are important; the UI expects these):

        {
          "candidate_summary": str,        # 1–3 sentences
          "candidate_skills": [str, ...],  # 10–20 short phrases
          "strengths": [str, ...],         # 3–6 bullets, project-specific
          "gaps": [str, ...],              # 3–6 bullets, project-specific
          "matched_must_have_skills": [str, ...],   # subset of project_profile["must_have_skills"]
          "missing_must_have_skills": [str, ...],   # subset of project_profile["must_have_skills"]
          "skill_match_percent": float     # 0–100, based ONLY on MUST-HAVE skills
        }
    """
    trimmed_resume = (resume_text or "")[:8000]

    project_summary = project_profile.get("project_summary", "")
    must_have_skills = project_profile.get("must_have_skills", []) or []
    nice_to_have_skills = project_profile.get("nice_to_have_skills", []) or []

    prompt = f"""
You are evaluating a candidate for a construction project.

You are given:
1) A brief project summary and required skills.
2) A candidate resume.

PROJECT SUMMARY:
{project_summary}

MUST-HAVE SKILLS (these drive the numeric match score):
{must_have_skills}

NICE-TO-HAVE SKILLS:
{nice_to_have_skills}

CANDIDATE RESUME TEXT:
{trimmed_resume}

Task:
1. Decide which of the MUST-HAVE skills the candidate clearly meets.
2. Decide which of the MUST-HAVE skills are clearly missing or too weak.
3. Extract 10–20 short "candidate_skills" phrases that best describe this
   person in the context of this project.
4. Write a short 1–3 sentence "candidate_summary" explaining their fit.
5. Write 3–6 short bullet "strengths" focused on project-relevant strengths.
6. Write 3–6 short bullet "gaps" focused on limitations or risks for this project.
7. Compute `skill_match_percent` as:
   100 * (number of MUST-HAVE skills met) / (total MUST-HAVE skills).

Return ONLY a JSON object with keys exactly:
- "candidate_summary"
- "candidate_skills"
- "strengths"
- "gaps"
- "matched_must_have_skills"   # list of phrases from MUST-HAVE SKILLS the candidate meets
- "missing_must_have_skills"   # list of phrases from MUST-HAVE SKILLS the candidate does NOT meet
- "skill_match_percent"        # numeric 0–100, based ONLY on MUST-HAVE skills
"""  # noqa: E501

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw_text = resp.choices[0].message.content
        data = json.loads(raw_text)
    except Exception as e:  # defensive fallback
        print(f"[TeamReadi] build_candidate_profile LLM error: {e}")
        data = {
            "candidate_summary": "",
            "candidate_skills": [],
            "strengths": [],
            "gaps": [],
            "matched_must_have_skills": [],
            "missing_must_have_skills": [],
            "skill_match_percent": None,
        }

    candidate_summary = (data.get("candidate_summary") or "").strip()
    candidate_skills = [str(s).strip() for s in (data.get("candidate_skills") or []) if str(s).strip()]
    strengths = [str(s).strip() for s in (data.get("strengths") or []) if str(s).strip()]
    gaps = [str(s).strip() for s in (data.get("gaps") or []) if str(s).strip()]
    matched = [str(s).strip() for s in (data.get("matched_must_have_skills") or []) if str(s).strip()]
    missing = [str(s).strip() for s in (data.get("missing_must_have_skills") or []) if str(s).strip()]
    skill_match_percent = data.get("skill_match_percent")

    # Fallback if the model did not return a numeric score
    try:
        skill_match_percent = float(skill_match_percent)
    except (TypeError, ValueError):
        # Compute from lists if possible, else fall back to lexical overlap
        if must_have_skills:
            matched_norm = _normalize_phrase_list(matched)
            must_norm = _normalize_phrase_list(must_have_skills)
            pct = 100.0 * (len(must_norm.intersection(matched_norm)) / max(1, len(must_norm)))
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


# ---------------------------------------------------------------------------
# Optional: high-level pipeline (still used by some experiments / tests)
# ---------------------------------------------------------------------------

def run_teamreadi_pipeline(project_text: str, resumes: Dict[str, str]) -> List[Dict[str, Any]]:
    """Legacy helper kept for completeness.

    The Streamlit page now assembles most of the pieces itself, but this
    function is still useful for debugging or offline tests.
    """
    project_profile = build_project_profile(project_text)
    rows: List[Dict[str, Any]] = []

    for emp_id, resume_text in (resumes or {}).items():
        cand_profile = build_candidate_profile(resume_text, project_profile)
        skillfit = cand_profile.get("skill_match_percent", 0.0) / 100.0

        rows.append(
            {
                "id": emp_id,
                "profile": cand_profile,
                "metrics": {
                    "skillfit": skillfit,
                    "readiscore": skillfit,  # availability is not handled here
                },
            }
        )

    rows_sorted = sorted(rows, key=lambda r: r["metrics"]["readiscore"], reverse=True)
    return rows_sorted
