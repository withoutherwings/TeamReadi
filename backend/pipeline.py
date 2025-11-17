"""TeamReadi backend pipeline.

LLM-powered helpers:

- build_project_profile(project_text)
- build_candidate_profile(resume_text, project_profile)
- compute_skill_match(must_have_skills, candidate_skills, matched_must_have_skills=None)
"""

import os
import json
from typing import Dict, List, Any, Set, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

USE_LLM = True  # flip to False if you want to disable live LLM calls

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

client = OpenAI(api_key=API_KEY) if API_KEY else None


def _llm_json(prompt: str) -> Dict[str, Any]:
    """
    Call the chat model and parse JSON from the response.

    If USE_LLM is False or the client is missing, return {} so callers can
    fall back to safe defaults.
    """
    if (not USE_LLM) or (client is None):
        return {}

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
        )
        text = resp.choices[0].message.content.strip()

        # Strip possible markdown code fences
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].lstrip().startswith("```"):
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
# Heuristics: company-level vs person-level requirements
# ---------------------------------------------------------------------------

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
# Project profile
# ---------------------------------------------------------------------------

def build_project_profile(job_text: str) -> Dict[str, Any]:
    """
    Use the LLM to summarize the RFP and extract structured fields.

    We intentionally keep *person-level* skill requirements separate from
    company-level business/certification requirements so that candidates
    are not penalized for SDVOSB status, FAR compliance, etc.

    We also ask for three summary paragraphs:
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

    # Person-level vs company-level skills
    raw_must = data.get("must_have_skills") or []
    raw_nice = data.get("nice_to_have_skills") or []

    must_person, must_company = _split_person_vs_company(raw_must)
    nice_person, nice_company = _split_person_vs_company(raw_nice)

    explicit_company = data.get("company_requirements") or []

    # Clean + de-duplicate company-level requirements
    company_reqs: List[str] = []
    for item in list(must_company + nice_company + explicit_company):
        s = str(item).strip()
        if s and s not in company_reqs:
            company_reqs.append(s)

    # Build final multi-paragraph summary
    p1 = str(data.get("project_summary_p1") or "").strip()
    p2 = str(data.get("project_summary_p2") or "").strip()
    p3 = str(data.get("project_summary_p3") or "").strip()

    if any([p1, p2, p3]):
        pieces = [p for p in (p1, p2, p3) if p]
        project_summary = "\n\n".join(pieces)
    else:
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
        # Only person-level items feed into skill matching and highlights
        "must_have_skills": must_person,
        "nice_to_have_skills": nice_person,
        # Company-level requirements kept for the project summary page
        "company_requirements": company_reqs,
        "role_mix_by_bucket": data.get("role_mix_by_bucket") or {},
    }
    return profile

# ---------------------------------------------------------------------------
# Candidate profiling & skill matching
# ---------------------------------------------------------------------------

def extract_resume_skills(resume_text: str) -> Dict[str, Any]:
    """
    FIRST STAGE: look ONLY at the resume and pull out a grounded skill list.

    - No project/RFP context is shown here.
    - The model is explicitly told NOT to invent skills that are not clearly stated.
    """
    if not resume_text or not resume_text.strip():
        return {"candidate_summary": "", "raw_skills": []}

    prompt = f"""
You are reading ONE anonymous construction/engineering RESUME.

Your job is to stay 100% grounded in the resume and extract only the skills
and experiences that are clearly supported by the text. DO NOT guess or infer
skills that are not mentioned.

RESUME TEXT:
---
{resume_text[:12000]}
---

Return a JSON object with:

1. "candidate_summary": 2–3 sentences summarizing the person's background
   and strengths for construction / infrastructure work. Mention sectors
   (healthcare, transportation, etc.) and typical scope, but do NOT mention
   any specific RFP or project from outside this resume.

2. "raw_skills": a list of SHORT skill/experience phrases (5–12 words each).
   Each item MUST be directly supported by the resume text.

Rules for "raw_skills":
- If the resume does NOT say "EHRM", "EHR", "electronic health record(s)",
  or similar phrasing, you MUST NOT claim that as a skill.
- Do NOT infer ownership status (SDVOSB, WOSB, 8(a), HUBZone, etc.).
- Do NOT invent experience with specific DOTs, agencies, or hospitals unless
  those names appear in the resume.
- Do NOT include generic legal/contract trivia like
  "experience handling liquidated damages" or "knowledge of North Carolina
  state construction regulations" unless those phrases (or very close wording)
  appear explicitly in the resume.

Return ONLY a JSON object with keys:
  "candidate_summary": string
  "raw_skills": [string, ...]
"""

    data = _llm_json(prompt) or {}
    summary = str(data.get("candidate_summary") or "").strip()
    skills = [
        str(s).strip()
        for s in data.get("raw_skills") or []
        if str(s).strip()
    ]
    return {"candidate_summary": summary, "raw_skills": skills}


def compute_skill_match(
    must_have_skills: List[str],
    candidate_skills: List[str],
    matched_must_have_skills: Optional[List[str]] = None,
) -> float:
    """
    Compute a conservative skill-match percentage (0–100).

    Primary logic:
      - We only score against project *must-have* skills.
      - If we have an explicit matched_must_have_skills list, we trust that
        and use it directly.
      - Otherwise we fall back to simple phrase overlap.
    """
    must = [s.strip().lower() for s in must_have_skills if s and s.strip()]
    if not must:
        return 0.0

    if matched_must_have_skills is not None:
        matched_norm = {
            s.strip().lower() for s in matched_must_have_skills if s and s.strip()
        }
        hits = sum(1 for m in must if m in matched_norm)
    else:
        cand_norm = {
            s.strip().lower() for s in candidate_skills if s and s.strip()
        }
        hits = sum(1 for m in must if m in cand_norm)

    return 100.0 * float(hits) / float(len(must))


def build_candidate_profile(
    resume_text: str,
    project_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    SECOND STAGE: compare resume skills to the project profile.

    Pipeline:
      1) Call extract_resume_skills() to get a grounded list of skills.
      2) Show ONLY those skills + the project must/nice-to-have lists to the LLM.
      3) Ask it to decide which project must-haves are met vs missing, and to
         write strengths/gaps WITHOUT inventing new skills.
    """
    base = extract_resume_skills(resume_text)
    candidate_summary = base["candidate_summary"]
    raw_skills = base["raw_skills"]

    must = project_profile.get("must_have_skills") or []
    nice = project_profile.get("nice_to_have_skills") or []

    prompt = f"""
You are aligning ONE candidate's resume skills to ONE construction project.

PROJECT REQUIREMENTS (from the RFP):
- Must-have skills (individual person-level, not company status):
{json.dumps(must, indent=2, ensure_ascii=False)}
- Nice-to-have skills (individual person-level):
{json.dumps(nice, indent=2, ensure_ascii=False)}

CANDIDATE SKILLS (derived directly from the resume and MUST NOT be changed):
{json.dumps(raw_skills, indent=2, ensure_ascii=False)}

Your tasks:

1. Decide which project must-have skills are clearly supported by the
   candidate skills. Treat a must-have as "matched" only if there is a direct
   or very close semantic match in CANDIDATE SKILLS.

2. Build two lists:
   - "matched_must_have_skills": project must-have phrases that ARE supported.
   - "missing_must_have_skills": project must-have phrases that are NOT clearly
     supported.

3. Write:
   - "strengths": 3–6 short bullet phrases (max ~15 words each) describing
     this candidate's BEST strengths for THIS project.
   - "gaps": 3–6 short bullet phrases describing the most important weaknesses
     or missing requirements vs THIS project.

Important rules:
- You MAY NOT invent new candidate skills that are not in CANDIDATE SKILLS.
- You MAY NOT upgrade generic healthcare or transportation experience into
  highly specific systems (e.g. "extensive EHRM experience") unless that
  system is explicitly mentioned in CANDIDATE SKILLS.
- Focus "gaps" on meaningful person-level skills and experience patterns
  (e.g. no large healthcare projects, limited PM responsibility).
- Avoid trivia and ultra-specific legal/contract points as gaps, such as:
    * liquidated damages clauses
    * knowledge of a particular state's procurement laws or regulations
    * having worked at the exact named facility
  unless those are spelled out as strict must-haves AND clearly absent.

Return ONLY a JSON object with keys:
  "matched_must_have_skills"
  "missing_must_have_skills"
  "strengths"
  "gaps"
"""

    data = _llm_json(prompt) or {}

    matched = [
        str(s).strip()
        for s in data.get("matched_must_have_skills") or []
        if str(s).strip()
    ]
    missing = [
        str(s).strip()
        for s in data.get("missing_must_have_skills") or []
        if str(s).strip()
    ]
    strengths = [
        str(s).strip()
        for s in data.get("strengths") or []
        if str(s).strip()
    ]
    gaps = [
        str(s).strip()
        for s in data.get("gaps") or []
        if str(s).strip()
    ]

    # Deterministic skill-match percentage based on must-have list
    skill_match_percent = compute_skill_match(
        project_profile.get("must_have_skills", []),
        raw_skills,
        matched_must_have_skills=matched,
    )

    profile: Dict[str, Any] = {
        "candidate_summary": candidate_summary,
        "candidate_skills": raw_skills,
        "matched_must_have_skills": matched,
        "missing_must_have_skills": missing,
        "strengths": strengths,
        "gaps": gaps,
        "skill_match_percent": skill_match_percent,
    }
    return profile
