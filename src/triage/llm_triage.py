"""
LLM triage layer.

Takes a flagged event + its deterministic MITRE ATT&CK matches (see
mitre_mapping.py) and produces an analyst-readable triage note: severity,
plain-language summary, and a recommended next action.

Design choice: the LLM is only asked to *explain and prioritize*, never to
pick the MITRE technique itself -- that's decided upstream by
mitre_mapping.py so technique IDs can't be hallucinated. This keeps the
LLM's job to what LLMs are actually good at (turning structured signal into
readable analyst language) instead of using it as an unverified oracle.

Falls back to a deterministic template if ANTHROPIC_API_KEY isn't set, so
the pipeline is fully runnable/demoable without an API key.
"""
from __future__ import annotations

import json
import os

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

SYSTEM_PROMPT = """You are a SOC triage assistant. You will be given one flagged \
authentication event, the behavioral features that triggered it, and MITRE ATT&CK \
techniques already matched by a deterministic rules engine (you must not invent or \
change these technique IDs -- only explain and prioritize using them).

Respond with ONLY a JSON object, no prose before or after, matching this schema:
{
  "severity": "low" | "medium" | "high" | "critical",
  "summary": "2-3 sentences, plain language, written for an analyst triaging a queue of alerts",
  "recommended_action": "1-2 sentences, concrete next step"
}"""


def _fallback_triage(event: dict, mitre_matches: list[dict]) -> dict:
    """Deterministic, no-API-key triage so the pipeline works out of the box."""
    top = mitre_matches[0]
    severity_map = {
        "T1110": "high",
        "T1078": "critical",
        "T1078.002": "medium",
        "T1021": "high",
    }
    severity = severity_map.get(top["technique_id"], "medium")
    summary = (
        f"Event on {event.get('dst_host')} involving {event.get('src_user')}: {top['reason']}. "
        f"Matched {top['technique_name']} ({top['technique_id'] or 'unmapped'}, "
        f"tactic: {top['tactic']})."
    )
    action = {
        "T1110": "Lock the targeted account and block the source IP; check for a successful "
                 "login at the end of the attempt sequence.",
        "T1078": "Force re-authentication with MFA and verify both locations with the user "
                 "directly before taking further action.",
        "T1021": "Confirm with the user whether this activity was expected (e.g. scripted "
                 "admin task); if not, isolate the account's active sessions.",
        "T1078.002": "Verify the access was intended; if the user cannot confirm it, treat as "
                     "a possible compromised-credential case.",
    }.get(top["technique_id"], "Review manually; insufficient signal for an automated recommendation.")
    return {"severity": severity, "summary": summary, "recommended_action": action}


def generate_triage(event: dict, mitre_matches: list[dict]) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        result = _fallback_triage(event, mitre_matches)
        result["source"] = "offline_fallback"
        return result

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        user_content = json.dumps({
            "event": {k: (str(v) if not isinstance(v, (int, float, str)) else v)
                      for k, v in event.items()},
            "mitre_matches": mitre_matches,
        }, default=str)

        response = client.messages.create(
            model=MODEL,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        result = json.loads(text)
        result["source"] = f"llm:{MODEL}"
        return result
    except Exception as exc:  # noqa: BLE001 -- degrade gracefully in a demo pipeline
        result = _fallback_triage(event, mitre_matches)
        result["source"] = f"offline_fallback (LLM call failed: {exc.__class__.__name__})"
        return result
