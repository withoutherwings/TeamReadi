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
# Candidate profile (uses project_summary P3 to align skills)
# ---------------------------------------------------------------------------

from typing import Dict, Any


def build_candidate_profile(
    resume_text: str,
    project_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Analyze a single candidate resume against the project profile.

    Uses:
      - project_summary P3 (team responsibilities) as the "what this team does"
      - must_have_skills from the project_profile
    to classify which must-have skills are met vs missing and to compute a
    skill_match_percent that stays consistent with the matched/missing lists.
    """

    resume_text = resume_text or ""

    # Project "must-have" skills list
    proj_must = [
        str(s).strip()
        for s in project_profile.get("must_have_skills", [])
        if str(s).strip()
    ]

    # Pull the "team responsibilities" paragraph = last paragraph of summary
    summary = project_profile.get("project_summary", "") or ""
    paras = [p.strip() for p in summary.split("\n\n") if p.strip()]
    team_resp = paras[-1] if paras else summary

    prompt = f"""
You are evaluating a construction candidate resume against a specific project.

PROJECT TEAM RESPONSIBILITIES (from RFP summary, P3):
---
{team_resp}
---

PROJECT MUST-HAVE SKILLS (person-level, not company ownership):
- """ + "\n- ".join(proj_must or ["<none explicitly listed>"]) + """

RESUME TEXT:
---
{resume_text[:9000]}
---

1. Read the project responsibilities and must-have skills carefully.
2. Read the resume and infer the candidate's key skills and experiences.

Return ONLY a JSON object with these keys:

- "candidate_skills": an array of SHORT skill phrases derived from the resume
  (e.g. "healthcare CM at VA facilities", "EHRM infrastructure upgrades").
- "matched_must_have_skills": array of must-have skills (from the project list)
  that this candidate clearly demonstrates.
- "missing_must_have_skills": array of must-have skills from the project list
  that are clearly NOT demonstrated in the resume.
- "strengths": 3–6 bullet-style phrases describing this candidate's best
  alignment with the project (focus on project-relevant experience).
- "gaps": 3–6 bullet-style phrases describing the main concerns / gaps vs the
  project requirements.
- "candidate_summary": a 2–3 sentence paragraph describing how well this
  candidate fits the project and what role they are best suited for.
- "skill_match_percent": an integer from 0 to 100 representing how well the
  candidate meets the must-have skills OVERALL. Rough guideline:
    * 90–100: nearly all must-haves clearly met
    * 70–89: most must-haves met, a few minor gaps
    * 40–69: mixed; several important gaps
    * 10–39: weak fit; only a few must-haves met
    * 0–9: almost no relevant must-haves met

Return ONLY JSON. Do not include any explanation outside the JSON.
"""

    data = _llm_json(prompt) or {}

    # ---- Normalize arrays ----
    def _as_str_list(x) -> list[str]:
        if isinstance(x, str):
            return [x.strip()] if x.strip() else []
        if isinstance(x, (list, tuple, set)):
            out = []
            for item in x:
                s = str(item).strip()
                if s:
                    out.append(s)
            return out
        return []

    cand_skills = _as_str_list(data.get("candidate_skills"))
    matched_raw = _as_str_list(data.get("matched_must_have_skills"))
    missing_raw = _as_str_list(data.get("missing_must_have_skills"))
    strengths = _as_str_list(data.get("strengths"))
    gaps = _as_str_list(data.get("gaps"))

    # ---- Compute skill_match_percent if missing or obviously bad ----
    skill_match = data.get("skill_match_percent")
    valid_skill_match = isinstance(skill_match, (int, float)) and 0 <= skill_match <= 100

    if not valid_skill_match:
        # Derive % from matched vs total must-have skills
        total_must = len(proj_must)
        if total_must > 0:
            # We trust the LLM's classification here
            n_matched = len(matched_raw)
            derived = int(round(100.0 * n_matched / total_must))
            skill_match = derived
        else:
            skill_match = 0

    # ---- Candidate summary text ----
    candidate_summary = (data.get("candidate_summary") or "").strip()

    profile: Dict[str, Any] = {
        "candidate_skills": cand_skills,
        "matched_must_have_skills": matched_raw,
        "missing_must_have_skills": missing_raw,
        "strengths": strengths,
        "gaps": gaps,
        "candidate_summary": candidate_summary,
        "skill_match_percent": skill_match,
    }
    return profile

