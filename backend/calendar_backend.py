"""
calendar_backend.py
Backend logic for TeamReadi:
- Fetch availability from public ICS calendar
- Extract booked hours per Employee_XXX
- Compute availability and ReadiScore
- Use OpenAI to generate narrative explanation
"""

import os
import re
import requests
import datetime as dt
from typing import Optional, Dict, Any, List, Set

from dateutil import tz
from icalendar import Calendar
from openai import OpenAI

# =======================
# Configuration constants
# =======================

SKILL_WEIGHT = 0.70          # Skill contribution to ReadiScore (70%)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =======================
# Helper functions
# =======================

def _canonical_emp_id(raw: str) -> Optional[str]:
    """
    Normalize 'Employee 1', 'employee_01', 'EMPLOYEE-001', 'John - employee 7'
    to 'Employee_007'. If no 'employee + number' pattern is found, return None.
    """
    if not raw:
        return None
    m = re.search(r"employee[\s_\-]*([0-9]+)", raw, re.IGNORECASE)
    if not m:
        return None
    n = int(m.group(1))
    return f"Employee_{n:03d}"


def _parse_employee_id(summary: str, description: str) -> Optional[str]:
    """
    Extract canonical Employee_XXX ID from summary or description.

    Formats supported (examples):
        'Employee 17 - Site Visit'
        'employee_017 | Project Meeting'
        'EMPLOYEE-7: Coordination Call'
        'EmployeeID: Employee_017 | Notes: ...'
    """
    summary = summary or ""
    description = description or ""

    # Try matching directly in the summary text
    eid = _canonical_emp_id(summary)
    if eid:
        return eid

    # Try a more explicit 'EmployeeID:' marker in the description
    marker = "EmployeeID:"
    if marker in description:
        after = description.split(marker, 1)[1].strip()
        eid = _canonical_emp_id(after)
        if eid:
            return eid

    # Fallback: also try a direct scan of the description
    eid = _canonical_emp_id(description)
    if eid:
        return eid

    return None


def _clip_to_window(
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    window_start: dt.datetime,
    window_end: dt.datetime,
) -> (Optional[dt.datetime], Optional[dt.datetime]):
    """
    Clip event [start_dt, end_dt] to the analysis window.
    Return (s,e) or (None,None) if no overlap.
    """
    s = max(start_dt, window_start)
    e = min(end_dt, window_end)
    if e <= s:
        return None, None
    return s, e


def compute_capacity_hours(
    start_date: dt.date,
    end_date: dt.date,
    working_days: Set[int],
    hours_per_day: float,
) -> float:
    """
    Compute capacity hours between start_date and end_date
    given selected working_days (0=Mon .. 6=Sun) and hours_per_day.
    """
    if not working_days:
        working_days = {0, 1, 2, 3, 4}  # fallback Mon–Fri

    cur = start_date
    total_days = 0
    while cur <= end_date:
        if cur.weekday() in working_days:
            total_days += 1
        cur += dt.timedelta(days=1)
    return total_days * hours_per_day


# =======================
# Calendar Parsing
# =======================

def fetch_calendar_hours_by_employee(
    ics_url: str,
    employee_ids: List[str],
    window_start: dt.datetime,
    window_end: dt.datetime,
    working_days: Set[int],
    hours_per_day: float,
    tz_name: str = "America/New_York",
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch a public .ics feed and sum booked hours for each employee in employee_ids.

    working_days: set of weekday ints (0=Mon..6=Sun) chosen by user
    hours_per_day: max working hours per chosen day (user slider/input)

    Returns structured dictionary:

    {
        "Employee_003": {
            "booked_hours": 28.0,
            "capacity_hours": 160.0,
            "events": [
                {"summary": "...", "start": "...", "end": "...", "duration_hours": 4.0}
            ]
        }
    }
    """
    tzinfo = tz.gettz(tz_name)
    window_start = window_start.replace(tzinfo=tzinfo)
    window_end = window_end.replace(tzinfo=tzinfo)

    resp = requests.get(ics_url, timeout=30)
    resp.raise_for_status()
    cal = Calendar.from_ical(resp.content)

    # Canonical employee IDs (we'll still only return metrics for the requested IDs)
    employee_ids = list(set(employee_ids))
    capacity_hours = compute_capacity_hours(
        start_date=window_start.date(),
        end_date=window_end.date(),
        working_days=working_days,
        hours_per_day=hours_per_day,
    )

    data: Dict[str, Dict[str, Any]] = {
        emp: {
            "booked_hours": 0.0,
            "capacity_hours": capacity_hours,
            "events": [],
        }
        for emp in employee_ids
    }

    # Walk all VEVENT blocks
    for comp in cal.walk("vevent"):
        summary = str(comp.get("summary", "") or "")
        description = str(comp.get("description", "") or "")

        emp_id = _parse_employee_id(summary, description)
        if not emp_id or emp_id not in data:
            continue

        dtstart = comp.decoded("dtstart")
        dtend = comp.decoded("dtend")

        # Normalize datetimes
        if isinstance(dtstart, dt.date) and not isinstance(dtstart, dt.datetime):
            # Convert all-day events to a generic working block (8–16)
            dtstart = dt.datetime.combine(dtstart, dt.time(8, 0), tzinfo=tzinfo)
            dtend = dt.datetime.combine(dtend, dt.time(16, 0), tzinfo=tzinfo)
        else:
            if dtstart.tzinfo is None:
                dtstart = dtstart.replace(tzinfo=tzinfo)
            if dtend.tzinfo is None:
                dtend = dtend.replace(tzinfo=tzinfo)

        # Respect user-selected working days
        if dtstart.date().weekday() not in working_days:
            continue

        s, e = _clip_to_window(dtstart, dtend, window_start, window_end)
        if s is None:
            continue

        duration_hours = (e - s).total_seconds() / 3600.0
        if duration_hours <= 0:
            continue

        data[emp_id]["booked_hours"] += duration_hours
        data[emp_id]["events"].append(
            {
                "summary": summary,
                "start": s.isoformat(),
                "end": e.isoformat(),
                "duration_hours": duration_hours,
            }
        )

    return data


# =======================
# ReadiScore Calculation
# =======================

def compute_metrics_for_employee(emp_data: Dict[str, Any], skill_match_pct: float) -> Dict[str, Any]:
    """
    Compute availability, ReadiScore, and expose basic metrics.
    """
    booked = float(emp_data.get("booked_hours", 0.0))
    capacity = float(emp_data.get("capacity_hours", 0.0))

    if capacity <= 0:
        availability_hours = 0.0
        availability_pct = 0.0
    else:
        availability_hours = max(capacity - booked, 0.0)
        availability_pct = max(0.0, min(100.0, (availability_hours / capacity) * 100.0))

    skill_match_pct = max(0.0, min(100.0, float(skill_match_pct)))

    readiscore = SKILL_WEIGHT * skill_match_pct + (1.0 - SKILL_WEIGHT) * availability_pct

    return {
        "booked_hours": booked,
        "capacity_hours": capacity,
        "availability_hours": availability_hours,
        "availability_pct": availability_pct,
        "skill_match_pct": skill_match_pct,
        "readiscore": readiscore,
    }


# =======================
# LLM Explanation Generation
# =======================

def llm_explain_employee(
    employee_id: str,
    metrics: Dict[str, Any],
    project_name: Optional[str] = None,
) -> str:
    """
    Generate a short human-readable explanation using OpenAI Responses API.
    """
    proj = project_name or "this project"

    prompt = f"""
You are generating a concise explanation for a workforce readiness dashboard.

Employee: {employee_id}
Project: {proj}

Metrics:
- Skill match: {metrics['skill_match_pct']:.1f}%
- Booked hours: {metrics['booked_hours']:.1f}
- Capacity: {metrics['capacity_hours']:.1f}
- Availability: {metrics['availability_pct']:.1f}%
- ReadiScore: {metrics['readiscore']:.1f}

Write 3–5 sentences:
- One-line summary of readiness
- Comment on utilization level (over- or under-loaded)
- Note any tradeoff between skill and availability
- Clear, factual, professional tone
"""

resp = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "You generate professional staffing analytics summaries."},
        {"role": "user", "content": prompt},
    ],
    temperature=0,
    max_tokens=220,
)

return resp.choices[0].message.content

