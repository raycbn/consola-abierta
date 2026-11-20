#!/usr/bin/env python3
"""Offline scorecard for manually recorded IT disaster-recovery exercises.

This tool validates timestamps and evidence completeness. It does not perform,
trigger, or verify a backup restore, replication, failover, or application test.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ScorecardError(ValueError):
    """Raised when the exercise input is incomplete or inconsistent."""


def parse_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ScorecardError(f"{field}: timestamp is required")
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ScorecardError(f"{field}: invalid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ScorecardError(f"{field}: timezone offset is required (for example Z or +01:00)")
    return parsed.astimezone(timezone.utc)


def seconds_between(later: datetime, earlier: datetime, field: str) -> int:
    seconds = int((later - earlier).total_seconds())
    if seconds < 0:
        raise ScorecardError(f"{field}: timestamp order is invalid")
    return seconds


def format_duration(seconds: int) -> str:
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours} h {minutes:02d} min {secs:02d} s"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def require_positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ScorecardError(f"{field}: must be a positive number")
    return float(value)


def analyse(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ScorecardError("Input root must be a JSON object")
    case_id = data.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ScorecardError("case_id: non-empty identifier is required")
    scenario = data.get("scenario", "")
    if not isinstance(scenario, str):
        raise ScorecardError("scenario: must be text")

    objectives = data.get("objectives")
    timestamps = data.get("timestamps")
    if not isinstance(objectives, dict) or not isinstance(timestamps, dict):
        raise ScorecardError("objectives and timestamps must be JSON objects")

    target_rto_seconds = int(require_positive_number(objectives.get("rto_minutes"), "objectives.rto_minutes") * 60)
    target_rpo_seconds = int(require_positive_number(objectives.get("rpo_minutes"), "objectives.rpo_minutes") * 60)

    disruption = parse_timestamp(timestamps.get("disruption_started_at"), "timestamps.disruption_started_at")
    recoverable_point = parse_timestamp(timestamps.get("last_recoverable_point_at"), "timestamps.last_recoverable_point_at")
    service_restored = parse_timestamp(timestamps.get("service_restored_at"), "timestamps.service_restored_at")
    business_accepted_raw = timestamps.get("business_accepted_at")
    business_accepted = parse_timestamp(business_accepted_raw, "timestamps.business_accepted_at") if business_accepted_raw else None

    rto_seconds = seconds_between(service_restored, disruption, "service_restored_at must not precede disruption_started_at")
    rpo_seconds = seconds_between(disruption, recoverable_point, "last_recoverable_point_at must not be after disruption_started_at")
    acceptance_lag_seconds = seconds_between(business_accepted, service_restored, "business_accepted_at must not precede service_restored_at") if business_accepted else None

    milestones = data.get("milestones", [])
    if not isinstance(milestones, list):
        raise ScorecardError("milestones must be an array")
    if len(milestones) < 3:
        raise ScorecardError("milestones: at least three recorded milestones are required")

    parsed_milestones: list[dict[str, Any]] = []
    evidence_issues: list[str] = []
    previous_time: datetime | None = None
    for index, item in enumerate(milestones, start=1):
        if not isinstance(item, dict):
            raise ScorecardError(f"milestones[{index}]: each milestone must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ScorecardError(f"milestones[{index}].name: non-empty name is required")
        at = parse_timestamp(item.get("at"), f"milestones[{index}].at")
        if at < disruption:
            raise ScorecardError(f"milestones[{index}].at: milestone precedes the disruption start")
        if previous_time and at < previous_time:
            raise ScorecardError("milestones: timestamps must be in chronological order")
        previous_time = at
        owner = item.get("owner")
        evidence = item.get("evidence")
        if not isinstance(owner, str) or not owner.strip():
            evidence_issues.append(f"Milestone '{name}': missing accountable owner")
        if not isinstance(evidence, str) or not evidence.strip():
            evidence_issues.append(f"Milestone '{name}': missing evidence reference")
        parsed_milestones.append({
            "name": name.strip(),
            "at": at.isoformat().replace("+00:00", "Z"),
            "owner": owner.strip() if isinstance(owner, str) else "",
            "evidence": evidence.strip() if isinstance(evidence, str) else "",
        })

    required_signals = {"service available", "technical validation", "business acceptance"}
    milestone_names = {m["name"].strip().lower() for m in parsed_milestones}
    missing_signals = sorted(required_signals - milestone_names)
    if missing_signals:
        evidence_issues.extend("Missing milestone: " + name for name in missing_signals)

    rto_pass = rto_seconds <= target_rto_seconds
    rpo_pass = rpo_seconds <= target_rpo_seconds
    evidence_pass = not evidence_issues
    overall_pass = rto_pass and rpo_pass and evidence_pass

    return {
        "case_id": case_id.strip(),
        "scenario": scenario,
        "measurement_basis": {
            "rto": "service_restored_at minus disruption_started_at",
            "rpo": "disruption_started_at minus last_recoverable_point_at",
            "acceptance_lag": "business_accepted_at minus service_restored_at; informational, not included in RTO",
        },
        "objectives": {
            "rto_seconds": target_rto_seconds,
            "rto_display": format_duration(target_rto_seconds),
            "rpo_seconds": target_rpo_seconds,
            "rpo_display": format_duration(target_rpo_seconds),
        },
        "observed": {
            "rto_seconds": rto_seconds,
            "rto_display": format_duration(rto_seconds),
            "rpo_seconds": rpo_seconds,
            "rpo_display": format_duration(rpo_seconds),
            "business_acceptance_lag_seconds": acceptance_lag_seconds,
            "business_acceptance_lag_display": format_duration(acceptance_lag_seconds) if acceptance_lag_seconds is not None else None,
        },
        "checks": {
            "rto": "PASS" if rto_pass else "FAIL",
            "rpo": "PASS" if rpo_pass else "FAIL",
            "evidence_completeness": "PASS" if evidence_pass else "FAIL",
            "overall": "PASS" if overall_pass else "FAIL",
        },
        "evidence_issues": evidence_issues,
        "milestones": parsed_milestones,
        "limitations": [
            "This scorecard checks entered timestamps and evidence references only.",
            "It does not execute or verify restores, replication, failover, data integrity, security controls, or application behaviour.",
            "The last recoverable point must be confirmed as usable by a separate restore or application-consistency test.",
            "RTO and RPO boundaries must be approved by the service owner before the exercise.",
        ],
    }


def markdown_report(report: dict[str, Any]) -> str:
    observed = report["observed"]
    objectives = report["objectives"]
    checks = report["checks"]
    lines = [
        f"# Recovery test scorecard — {report['case_id']}",
        "",
        report.get("scenario", ""),
        "",
        "## Result",
        "",
        f"- **Overall:** {checks['overall']}",
        f"- **RTO:** {checks['rto']} — observed {observed['rto_display']} / target {objectives['rto_display']}",
        f"- **RPO:** {checks['rpo']} — observed {observed['rpo_display']} / target {objectives['rpo_display']}",
        f"- **Evidence completeness:** {checks['evidence_completeness']}",
        f"- **Business acceptance lag:** {observed['business_acceptance_lag_display'] or 'not recorded'} (informational)",
        "",
        "## Evidence issues",
        "",
    ]
    lines.extend([f"- {issue}" for issue in report["evidence_issues"]] or ["- None recorded by this scorecard."])
    lines += ["", "## Milestones", "", "| Milestone | UTC timestamp | Owner | Evidence reference |", "|---|---|---|---|"]
    for item in report["milestones"]:
        lines.append(f"| {item['name']} | {item['at']} | {item['owner']} | {item['evidence']} |")
    lines += ["", "## Measurement basis", "", f"- RTO: {report['measurement_basis']['rto']}.", f"- RPO: {report['measurement_basis']['rpo']}.", f"- Acceptance lag: {report['measurement_basis']['acceptance_lag']}.", "", "## Limitations", ""]
    lines.extend([f"- {item}" for item in report["limitations"]])
    lines += ["", "Generated offline by Consola Abierta Recovery Test Scorecard.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse an offline DR exercise scorecard JSON file.")
    parser.add_argument("input", type=Path, help="JSON file containing manually recorded exercise data")
    parser.add_argument("--json-out", type=Path, help="Optional path for a JSON report")
    parser.add_argument("--markdown-out", type=Path, help="Optional path for a Markdown report")
    args = parser.parse_args(argv)

    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
        report = analyse(data)
    except (OSError, json.JSONDecodeError, ScorecardError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.write_text(markdown_report(report), encoding="utf-8")
    return 0 if report["checks"]["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
