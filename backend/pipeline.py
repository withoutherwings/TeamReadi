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

from typing import Dict, Any  # keep this near the top of the file too


def build_project_profile(job_text: str) -> Dict[str, Any]:
    """
    Use the LLM (or a stub when USE_LLM is False) to summarize the RFP and
    extract structured fields for the rest of the pipeline.

    We intentionally keep *person-level* skill requirements separate from
    company-level business/certification requirements so that candidates
    are not penalized for SDVOSB status, FAR compliance, etc.

    We also ask for three clearly separated summary paragraphs:
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
            "trainable_requirements": [],
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

10. "trainable_requirements": an array of SHORT items describing regulatory,
    procedural, or contract-knowledge expectations that are important but
    realistically trainable for competent staff (e.g. NPDES reporting,
    Buy American Act compliance, specific state procurement rules,
    facility-specific IT or scheduling portals).
    These are NOT treated as hard disqualifying must-have skills.

11. "role_mix_by_bucket": an object with integer counts estimating how many
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
"company_requirements", "trainable_requirements",
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

    # Trainable requirements (keep as-is, just clean)
    trainable_raw = data.get("trainable_requirements") or []
    trainable: list[str] = []
    for item in trainable_raw:
        s = str(item).strip()
        if s and s not in trainable:
            trainable.append(s)

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
        # New: things like NPDES, specific regs, etc., treated as teachable
        "trainable_requirements": trainable,
        "role_mix_by_bucket": data.get("role_mix_by_bucket") or {},
    }
    return profile

# ---------------------------------------------------------------------------
# Minimal resume skill extractor (no external backend.skills_backend needed)
# ---------------------------------------------------------------------------

def extract_resume_skills(resume_text: str) -> Dict[str, Any]:
    """
    Lightweight, non-LLM skill extractor used by the TeamReadi pipeline.

    Returns a dict with:
      - candidate_summary: short plain-text blurb
      - raw_skills: list of text snippets we treat as 'skills'
    """
    text = (resume_text or "").strip()
    if not text:
        return {"candidate_summary": "", "raw_skills": []}

    # Simple summary: first ~400 characters with whitespace collapsed
    cleaned = " ".join(text.split())
    candidate_summary = cleaned[:400]

    # Very crude "skills": non-empty lines / bullet points
    raw_skills: List[str] = []
    for line in resume_text.splitlines():
        line = line.strip(" \t•-*–")
        if not line:
            continue
        if len(line) < 3:
            continue
        raw_skills.append(line)

    if not raw_skills:
        raw_skills = [cleaned]

    return {"candidate_summary": candidate_summary, "raw_skills": raw_skills}

# ---------------------------------------------------------------------------
# Candidate profiling & skill matching
# ---------------------------------------------------------------------------

from typing import Dict, Any, List, Optional
import json


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

    NEW:
      - project_profile['trainable_requirements'] is passed separately.
      - Trainable items may be mentioned as "will need to learn X" style gaps,
        but they do NOT count as missing must-have skills and they do NOT
        reduce the skill_match_percent.
    """
    # Extract structured skills from the raw resume text
    base = extract_resume_skills(resume_text)
    candidate_summary = base["candidate_summary"]
    raw_skills = base["raw_skills"]

    must = project_profile.get("must_have_skills") or []
    nice = project_profile.get("nice_to_have_skills") or []
    trainable = project_profile.get("trainable_requirements") or []

    prompt = f"""
You are aligning ONE candidate's resume skills to ONE construction project.

PROJECT REQUIREMENTS (from the RFP):
- Must-have skills (individual person-level, not company status):
{json.dumps(must, indent=2, ensure_ascii=False)}
- Nice-to-have skills (individual person-level):
{json.dumps(nice, indent=2, ensure_ascii=False)}

TRAINABLE KNOWLEDGE / REQUIREMENTS
(important but realistically teachable for competent staff):
{json.dumps(trainable, indent=2, ensure_ascii=False)}

CANDIDATE SKILLS (derived directly from the resume and MUST NOT be changed):
{json.dumps(raw_skills, indent=2, ensure_ascii=False)}

Your tasks:

1. Decide which project MUST-HAVE skills are clearly supported by the
   candidate skills. Treat a must-have as "matched" only if there is a direct
   or very close semantic match in CANDIDATE SKILLS.

2. Build two lists ONLY for the strict must-have skills (NOT for trainable items):
   - "matched_must_have_skills": project must-have phrases that ARE supported.
   - "missing_must_have_skills": project must-have phrases that are NOT clearly
     supported.

   Do NOT include items from the TRAINABLE list in either of these lists.

3. Write:
   - "strengths": 3–6 short bullet phrases (max ~15 words each) describing
     this candidate's BEST strengths for THIS project.
   - "gaps": 3–6 short bullet phrases describing the most important weaknesses
     or missing requirements vs THIS project, including:
       * true missing must-have skills, and
       * any TRAINABLE items that are absent, phrased softly as
         "Will need to learn/receive support on X" instead of hard disqualifiers.

Important rules:
- You MAY NOT invent new candidate skills that are not in CANDIDATE SKILLS.
- You MAY NOT upgrade generic healthcare or transportation experience into
  highly specific systems (e.g. "extensive EHRM experience") unless that
  system is explicitly mentioned in CANDIDATE SKILLS.
- Avoid trivia and ultra-specific legal/contract points as hard gaps, such as:
    * liquidated damages clauses
    * knowledge of a particular state's procurement laws or regulations
    * having worked at the exact named facility
  unless those are spelled out as strict must-haves AND clearly absent.
- TRAINABLE items should be treated as "nice to have but teachable", never as
  automatic disqualifiers.

Return ONLY a JSON object with keys:
  "matched_must_have_skills"
  "missing_must_have_skills"
  "strengths"
  "gaps"
"""

    data = _llm_json(prompt) or {}

    def _norm_list(key: str) -> List[str]:
        raw = data.get(key) or []
        if isinstance(raw, str):
            raw = [raw]
        out: List[str] = []
        for item in raw:
            s = str(item).strip()
            if s:
                out.append(s)
        return out

    matched = _norm_list("matched_must_have_skills")
    missing = _norm_list("missing_must_have_skills")
    strengths = _norm_list("strengths")
    gaps = _norm_list("gaps")

    # Deterministic skill-match percentage based only on strict must-have list.
    # We let compute_skill_match use the *full* candidate_skills overlap,
    # not just the LLM's matched_must_have_skills bucket, so the percentage
    # reflects overall resume content instead of only 2–3 highlighted items.
    strict_must = [s for s in must if s not in trainable]
    skill_match_percent = compute_skill_match(
        strict_must,
        raw_skills,
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


# ---------------------------------------------------------------------------
# Low-level skill matching helper
# ---------------------------------------------------------------------------

def compute_skill_match(
    must_have_skills: List[str],
    candidate_skills: List[str],
    matched_must_have_skills: Optional[List[str]] = None,
) -> float:
    """
    Return a 0–100 percentage of how many must_have_skills are clearly met.

    If matched_must_have_skills is provided (from the LLM), we trust that list
    and compute the score from it. Otherwise we fall back to a simple overlap
    between normalized text in must_have_skills and candidate_skills.
    """
    # Normalize must-have list
    must = [str(s).strip().lower() for s in (must_have_skills or []) if str(s).strip()]
    if not must:
        return 0.0

    # Case 1: LLM gave us an explicit matched_must_have_skills bucket
    if matched_must_have_skills is not None:
        matched_clean = [
            str(s).strip().lower()
            for s in (matched_must_have_skills or [])
            if str(s).strip()
        ]
        return 100.0 * len(matched_clean) / len(must)

    # Case 2: simple overlap fallback based on all candidate skills
    cand = {str(s).strip().lower() for s in (candidate_skills or []) if str(s).strip()}
    matched_count = sum(1 for m in must if m in cand)
    return 100.0 * matched_count / len(must)
