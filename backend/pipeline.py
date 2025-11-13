"""
pipeline.py
High-level TeamReadi pipeline:
- Project text + resumes -> skill_match results
- Calendar -> availability
- Combine into ReadiScore and sorted ranking
"""

import datetime as dt
from typing import Dict

from backend.skills_backend import (
    extract_project_requirements,
    score_resume_against_requirements,
)
from backend.calendar_backend import (
    fetch_calendar_hours_by_employee,
    compute_metrics_for_employee,
)


def build_skill_results(project_text: str, resumes: Dict[str, str]) -> Dict[str, dict]:
    """
    project_text: full text of project RFP/spec
    resumes: {"Employee_001": "full resume text", ...}

    Returns dict:
    {
        "Employee_001": {
            "skill_match_pct": float,
            "highlights": [{"skill": str, "met": bool}, ...],
        },
        ...
    }
    """
    requirements = extract_project_requirements(project_text)

    results = {}
    for emp_id, resume_text in resumes.items():
        scored = score_resume_against_requirements(requirements, resume_text)
        per_skill = scored["per_skill"]
        skill_match_pct = float(scored["skill_match_pct"])

        highlights = []
        for s in per_skill:
            status = s.get("match_status", "")
            met = status == "strong_match"
            label = s.get("label", "")
            highlights.append({"skill": label, "met": met})

        results[emp_id] = {
            "skill_match_pct": skill_match_pct,
            "highlights": highlights,
            "per_skill_raw": per_skill,  # keep if needed for full report
        }

    return results


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
    - derive skill match from project + resumes
    - derive availability from calendar
    - compute ReadiScore
    - return ranked list of employees
    """
    # 1) Skill side
    skill_results = build_skill_results(project_text, resumes)
    employee_ids = list(skill_results.keys())

    # 2) Calendar side
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
        skill_match_pct = skill_results[emp_id]["skill_match_pct"]
        highlights = skill_results[emp_id]["highlights"]

        emp_data = calendar_data.get(
            emp_id,
            {"booked_hours": 0.0, "capacity_hours": 0.0, "events": []},
        )

        metrics = compute_metrics_for_employee(emp_data, skill_match_pct)

        rows.append(
            {
                "employee_id": emp_id,
                "metrics": metrics,
                "highlights": highlights,
            }
        )

    # 4) Sort by ReadiScore descending
    rows_sorted = sorted(
        rows, key=lambda r: r["metrics"]["readiscore"], reverse=True
    )

    return rows_sorted
