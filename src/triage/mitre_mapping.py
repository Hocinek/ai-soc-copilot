"""
Deterministic mapping from triggered detection features to MITRE ATT&CK
techniques. Kept separate from the LLM so technique IDs are never
hallucinated -- the LLM explains and prioritizes, this module decides
*which* technique applies.
"""
from __future__ import annotations

THRESHOLDS = {
    "brute_force": 10,       # failed_attempts_10min_by_ip
    "lateral_movement": 7,   # distinct_hosts_60min_by_user
    "impossible_travel": 2,  # distinct_countries_180min_by_user
}


def map_to_mitre(event: dict) -> list[dict]:
    """
    event: a dict/Series with the engineered feature columns.
    returns: list of {technique_id, technique_name, tactic, reason} --
             an event can match more than one technique.
    """
    matches = []

    if event.get("failed_attempts_10min_by_ip", 0) >= THRESHOLDS["brute_force"]:
        matches.append({
            "technique_id": "T1110",
            "technique_name": "Brute Force",
            "tactic": "Credential Access",
            "reason": f"{int(event['failed_attempts_10min_by_ip'])} failed attempts from "
                      f"{event.get('src_ip')} within 10 minutes",
        })

    if event.get("distinct_countries_180min_by_user", 0) >= THRESHOLDS["impossible_travel"]:
        matches.append({
            "technique_id": "T1078",
            "technique_name": "Valid Accounts",
            "tactic": "Defense Evasion / Initial Access",
            "reason": f"{event.get('src_user')} authenticated from "
                      f"{int(event['distinct_countries_180min_by_user'])} distinct countries "
                      f"within 3 hours -- physically implausible",
        })

    if event.get("distinct_hosts_60min_by_user", 0) >= THRESHOLDS["lateral_movement"]:
        matches.append({
            "technique_id": "T1021",
            "technique_name": "Remote Services",
            "tactic": "Lateral Movement",
            "reason": f"{event.get('src_user')} authenticated to "
                      f"{int(event['distinct_hosts_60min_by_user'])} distinct hosts within 60 minutes",
        })

    if event.get("is_off_hours") and event.get("unusual_host_for_user"):
        matches.append({
            "technique_id": "T1078.002",
            "technique_name": "Valid Accounts: Domain Accounts",
            "tactic": "Initial Access / Persistence",
            "reason": f"off-hours success by {event.get('src_user')} to a host they don't "
                      f"normally access ({event.get('dst_host')})",
        })

    if not matches:
        matches.append({
            "technique_id": None,
            "technique_name": "Unmapped anomaly",
            "tactic": "Unknown",
            "reason": "flagged by the anomaly model but did not cross any rule threshold "
                      "-- review manually",
        })

    return matches
