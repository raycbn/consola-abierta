#!/usr/bin/env python3
"""Offline audit helper for DR programme coverage and action follow-through."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

SERVICE_FIELDS = {
    "service_id", "service_name", "criticality", "in_scope",
    "target_cadence_days", "dependency_review_date",
}
EXERCISE_FIELDS = {
    "exercise_id", "service_id", "exercise_date", "exercise_type", "scope",
    "qualifying_for_recovery_coverage", "objective_result", "target_rto_minutes",
    "observed_rto_minutes", "target_rpo_minutes", "observed_rpo_minutes",
    "evidence_reference",
}
ACTION_FIELDS = {
    "action_id", "service_id", "finding_id", "owner", "due_date", "status",
    "verification_date", "verification_evidence",
}
EXCEPTION_FIELDS = {
    "exception_id", "service_id", "risk_owner", "rationale",
    "compensating_controls", "approver", "approval_date", "expiry_date", "status",
}
BOOLS = {"true": True, "false": False}
OBJECTIVE_RESULTS = {"met", "partially_met", "not_met", "not_tested"}
ACTION_STATES = {"OPEN", "IN_PROGRESS", "PENDING_VALIDATION", "CLOSED", "ACCEPTED_RISK"}
EXCEPTION_STATES = {"APPROVED", "DRAFT", "REJECTED", "REVOKED"}


class InputError(ValueError):
    pass


def read_csv(path: Path, required: set[str], label: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = set(reader.fieldnames or [])
            missing = required - headers
            if missing:
                raise InputError(f"{label}: missing columns: {', '.join(sorted(missing))}")
            rows = []
            for number, row in enumerate(reader, start=2):
                clean = {key: (value or "").strip() for key, value in row.items() if key}
                if not any(clean.values()):
                    continue
                clean["_row"] = str(number)
                rows.append(clean)
            return rows
    except OSError as exc:
        raise InputError(f"Cannot read {label} file {path}: {exc}") from exc


def parse_date(value: str, field: str, row: str, optional: bool = False) -> date | None:
    if optional and not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise InputError(f"row {row}: {field} must use YYYY-MM-DD, got {value!r}") from exc


def parse_bool(value: str, field: str, row: str) -> bool:
    if value.lower() not in BOOLS:
        raise InputError(f"row {row}: {field} must be true or false")
    return BOOLS[value.lower()]


def parse_number(value: str, field: str, row: str, optional: bool = False) -> float | None:
    if optional and value == "":
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise InputError(f"row {row}: {field} must be numeric, got {value!r}") from exc
    if number < 0:
        raise InputError(f"row {row}: {field} cannot be negative")
    return number


def unique(rows: list[dict[str, str]], field: str, label: str) -> None:
    seen: set[str] = set()
    for row in rows:
        value = row.get(field, "")
        if not value:
            raise InputError(f"{label} row {row['_row']}: {field} is required")
        if value in seen:
            raise InputError(f"{label} row {row['_row']}: duplicate {field} {value}")
        seen.add(value)


def validate_and_normalise(
    service_rows: list[dict[str, str]],
    exercise_rows: list[dict[str, str]],
    action_rows: list[dict[str, str]],
    exception_rows: list[dict[str, str]],
    as_of: date,
    window_days: int,
    dependency_freshness_days: int,
    qualification_policy_id: str | None = None,
) -> dict[str, Any]:
    unique(service_rows, "service_id", "services")
    unique(exercise_rows, "exercise_id", "exercises")
    unique(action_rows, "action_id", "actions")
    unique(exception_rows, "exception_id", "exceptions")

    services: dict[str, dict[str, Any]] = {}
    for row in service_rows:
        sid = row["service_id"]
        services[sid] = {
            **row,
            "in_scope_bool": parse_bool(row["in_scope"], "in_scope", row["_row"]),
            "cadence": int(row["target_cadence_days"]),
            "dependency_review": parse_date(row["dependency_review_date"], "dependency_review_date", row["_row"], True),
        }
        if services[sid]["cadence"] <= 0:
            raise InputError(f"services row {row['_row']}: target_cadence_days must be greater than zero")

    exercises: list[dict[str, Any]] = []
    for row in exercise_rows:
        sid = row["service_id"]
        if sid not in services:
            raise InputError(f"exercises row {row['_row']}: unknown service_id {sid}")
        exercise_date = parse_date(row["exercise_date"], "exercise_date", row["_row"])
        if exercise_date and exercise_date > as_of:
            raise InputError(f"exercises row {row['_row']}: exercise_date is after --as-of")
        result = row["objective_result"].lower()
        if result not in OBJECTIVE_RESULTS:
            raise InputError(f"exercises row {row['_row']}: invalid objective_result {result!r}")
        exercises.append({
            **row,
            "date": exercise_date,
            "qualifying": parse_bool(row["qualifying_for_recovery_coverage"], "qualifying_for_recovery_coverage", row["_row"]),
            "result": result,
            "target_rto": parse_number(row["target_rto_minutes"], "target_rto_minutes", row["_row"], True),
            "observed_rto": parse_number(row["observed_rto_minutes"], "observed_rto_minutes", row["_row"], True),
            "target_rpo": parse_number(row["target_rpo_minutes"], "target_rpo_minutes", row["_row"], True),
            "observed_rpo": parse_number(row["observed_rpo_minutes"], "observed_rpo_minutes", row["_row"], True),
        })
        if not row["exercise_type"] or not row["scope"] or not row["evidence_reference"]:
            raise InputError(f"exercises row {row['_row']}: exercise_type, scope and evidence_reference are required")

    actions: list[dict[str, Any]] = []
    for row in action_rows:
        if row["service_id"] not in services:
            raise InputError(f"actions row {row['_row']}: unknown service_id {row['service_id']}")
        status = row["status"].upper()
        if status not in ACTION_STATES:
            raise InputError(f"actions row {row['_row']}: invalid status {status!r}")
        actions.append({
            **row,
            "status_norm": status,
            "due": parse_date(row["due_date"], "due_date", row["_row"]),
            "verified_at": parse_date(row["verification_date"], "verification_date", row["_row"], True),
        })
        if not row["owner"] or not row["finding_id"]:
            raise InputError(f"actions row {row['_row']}: owner and finding_id are required")
        if status == "CLOSED" and not row["verification_evidence"]:
            # Kept as a finding, not silently accepted as verified closure.
            pass

    exceptions: list[dict[str, Any]] = []
    for row in exception_rows:
        if row["service_id"] not in services:
            raise InputError(f"exceptions row {row['_row']}: unknown service_id {row['service_id']}")
        approval = parse_date(row["approval_date"], "approval_date", row["_row"])
        expiry = parse_date(row["expiry_date"], "expiry_date", row["_row"])
        status = row["status"].upper()
        if status not in EXCEPTION_STATES:
            raise InputError(f"exceptions row {row['_row']}: invalid status {status!r}")
        if expiry < approval:
            raise InputError(f"exceptions row {row['_row']}: expiry_date precedes approval_date")
        required_text = ("risk_owner", "rationale", "compensating_controls", "approver")
        if any(not row[key] for key in required_text):
            raise InputError(f"exceptions row {row['_row']}: owner, rationale, controls and approver are required")
        exceptions.append({**row, "approval": approval, "expiry": expiry, "status_norm": status})

    start = as_of - timedelta(days=window_days - 1)
    return {
        "services": services,
        "exercises": exercises,
        "actions": actions,
        "exceptions": exceptions,
        "window_start": start,
        "as_of": as_of,
        "dependency_freshness_days": dependency_freshness_days,
        "qualification_policy_id": qualification_policy_id.strip() if qualification_policy_id and qualification_policy_id.strip() else None,
    }


def audit(data: dict[str, Any]) -> dict[str, Any]:
    services = data["services"]
    in_scope = [s for s in services.values() if s["in_scope_bool"]]
    by_service: dict[str, list[dict[str, Any]]] = {sid: [] for sid in services}
    for ex in data["exercises"]:
        by_service[ex["service_id"]].append(ex)

    exceptions_by_service: dict[str, list[dict[str, Any]]] = {sid: [] for sid in services}
    for exception in data["exceptions"]:
        if exception["status_norm"] == "APPROVED" and exception["expiry"] >= data["as_of"]:
            exceptions_by_service[exception["service_id"]].append(exception)

    findings: list[dict[str, str]] = []
    latest_qualifying: dict[str, dict[str, Any] | None] = {}
    covered = overdue = stale_dependencies = valid_exceptions = 0
    results = Counter()
    rto_measured = rto_passed = rpo_measured = rpo_passed = 0
    for service in in_scope:
        sid = service["service_id"]
        valid_exceptions += bool(exceptions_by_service[sid])
        eligible = [
            ex for ex in by_service[sid]
            if ex["qualifying"] and data["window_start"] <= ex["date"] <= data["as_of"]
        ]
        latest = max(eligible, key=lambda item: item["date"]) if eligible else None
        latest_qualifying[sid] = latest
        if latest:
            covered += 1
            age = (data["as_of"] - latest["date"]).days
            due = age > service["cadence"]
            results[latest["result"]] += 1
            if latest["target_rto"] is not None and latest["observed_rto"] is not None:
                rto_measured += 1
                rto_passed += latest["observed_rto"] <= latest["target_rto"]
            if latest["target_rpo"] is not None and latest["observed_rpo"] is not None:
                rpo_measured += 1
                rpo_passed += latest["observed_rpo"] <= latest["target_rpo"]
        else:
            age = None
            due = True
        overdue += due
        if due:
            findings.append({
                "severity": "HIGH" if service["criticality"].upper() in {"TIER 1", "CRITICAL", "CRITICA"} else "MEDIUM",
                "finding": "EXERCISE_OVERDUE",
                "service_id": sid,
                "detail": "No qualifying technical recovery exercise in the reporting window." if latest is None else f"Latest qualifying exercise is {age} days old; target cadence is {service['cadence']} days.",
            })
        if not latest:
            findings.append({
                "severity": "INFO",
                "finding": "NO_QUALIFYING_COVERAGE",
                "service_id": sid,
                "detail": "No qualifying technical recovery exercise in the reporting window; tabletop and component exercises are not automatically equivalent.",
            })
        if exceptions_by_service[sid]:
            findings.append({
                "severity": "INFO",
                "finding": "VALID_EXCEPTION",
                "service_id": sid,
                "detail": "Approved, unexpired exception exists; service remains in the coverage denominator.",
            })
        review = service["dependency_review"]
        if review is None or (data["as_of"] - review).days > data["dependency_freshness_days"]:
            stale_dependencies += 1
            findings.append({
                "severity": "MEDIUM",
                "finding": "DEPENDENCY_REVIEW_STALE",
                "service_id": sid,
                "detail": "Dependency review date is missing or older than the configured freshness threshold.",
            })

    verified_closed = overdue_actions = unverified_closed = 0
    for action in data["actions"]:
        verified = action["status_norm"] == "CLOSED" and action["verified_at"] is not None and bool(action["verification_evidence"])
        if verified:
            verified_closed += 1
        elif action["status_norm"] == "CLOSED":
            unverified_closed += 1
            findings.append({
                "severity": "HIGH",
                "finding": "CLOSED_ACTION_WITHOUT_VERIFICATION",
                "service_id": action["service_id"],
                "detail": f"Action {action['action_id']} is marked CLOSED but lacks verification date or evidence.",
            })
        if action["due"] < data["as_of"] and not verified and action["status_norm"] != "ACCEPTED_RISK":
            overdue_actions += 1
            findings.append({
                "severity": "HIGH",
                "finding": "ACTION_OVERDUE",
                "service_id": action["service_id"],
                "detail": f"Action {action['action_id']} passed its due date ({action['due']}).",
            })

    for exception in data["exceptions"]:
        if exception["status_norm"] == "APPROVED" and exception["expiry"] < data["as_of"]:
            findings.append({
                "severity": "HIGH",
                "finding": "EXCEPTION_EXPIRED",
                "service_id": exception["service_id"],
                "detail": f"Exception {exception['exception_id']} expired on {exception['expiry']}.",
            })

    denominator = len(in_scope)
    actions_total = len(data["actions"])
    scope_ids = sorted(service["service_id"] for service in in_scope)
    scope_fingerprint = hashlib.sha256("\n".join(scope_ids).encode("utf-8")).hexdigest()
    rto_ids = sorted(
        sid for sid, exercise in latest_qualifying.items()
        if exercise and exercise["target_rto"] is not None and exercise["observed_rto"] is not None
    )
    rpo_ids = sorted(
        sid for sid, exercise in latest_qualifying.items()
        if exercise and exercise["target_rpo"] is not None and exercise["observed_rpo"] is not None
    )
    rto_scope_fingerprint = hashlib.sha256("\n".join(rto_ids).encode("utf-8")).hexdigest()
    rpo_scope_fingerprint = hashlib.sha256("\n".join(rpo_ids).encode("utf-8")).hexdigest()
    return {
        "as_of": data["as_of"].isoformat(),
        "scope": {
            "in_scope_services": denominator,
            "service_ids_sha256": scope_fingerprint,
            "rto_measured_services": rto_measured,
            "rto_service_ids_sha256": rto_scope_fingerprint,
            "rpo_measured_services": rpo_measured,
            "rpo_service_ids_sha256": rpo_scope_fingerprint,
        },
        "methodology": {
            "auditor_version": "1.1",
            "dependency_freshness_days": data["dependency_freshness_days"],
            "qualification_policy_id": data["qualification_policy_id"],
        },
        "reporting_window": {
            "start": data["window_start"].isoformat(),
            "end": data["as_of"].isoformat(),
            "days": (data["as_of"] - data["window_start"]).days + 1,
        },
        "services": {
            "in_scope": denominator,
            "covered_by_qualifying_exercise": covered,
            "coverage_percent": round(covered / denominator * 100, 1) if denominator else None,
            "uncovered": denominator - covered,
            "overdue_by_target_cadence": overdue,
            "valid_approved_exceptions": valid_exceptions,
            "stale_or_missing_dependency_reviews": stale_dependencies,
        },
        "latest_qualifying_objective_results": dict(results),
        "rto": {
            "measured_services": rto_measured,
            "met": rto_passed,
            "pass_percent": round(rto_passed / rto_measured * 100, 1) if rto_measured else None,
        },
        "rpo": {
            "measured_services": rpo_measured,
            "met": rpo_passed,
            "pass_percent": round(rpo_passed / rpo_measured * 100, 1) if rpo_measured else None,
        },
        "actions": {
            "total": actions_total,
            "verified_closed": verified_closed,
            "verified_closure_percent_of_all_actions": round(verified_closed / actions_total * 100, 1) if actions_total else None,
            "overdue_unverified": overdue_actions,
            "closed_without_verification": unverified_closed,
        },
        "limitations": [
            "Qualifying status is supplied by the programme owner and is not independently assessed.",
            "The full in-scope service population remains the coverage denominator; exceptions are reported separately.",
            "RTO/RPO rates exclude records without both numeric target and observed values.",
            "The report does not test infrastructure or certify business continuity capability.",
        ],
        "findings": findings,
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "programme-summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "service-findings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["severity", "finding", "service_id", "detail"])
        writer.writeheader()
        writer.writerows(report["findings"])
    services = report["services"]
    lines = [
        "# DR Programme — Management Review",
        "",
        f"**As of:** {report['as_of']}  ",
        f"**Window:** {report['reporting_window']['start']} to {report['reporting_window']['end']} ({report['reporting_window']['days']} days)",
        f"**Scope checksum (SHA-256):** {report['scope']['service_ids_sha256']} (sorted in-scope service IDs; checksum is not anonymisation)",
        f"**RTO measured cohort:** {report['scope']['rto_measured_services']} services · SHA-256 {report['scope']['rto_service_ids_sha256']}",
        f"**RPO measured cohort:** {report['scope']['rpo_measured_services']} services · SHA-256 {report['scope']['rpo_service_ids_sha256']}",
        f"**Methodology:** {report['methodology']['auditor_version']} · dependency freshness {report['methodology']['dependency_freshness_days']} days · qualification policy {report['methodology']['qualification_policy_id'] or 'UNVERSIONED'}",
        "",
        "## Portfolio coverage",
        "",
        f"- In-scope services: {services['in_scope']}",
        f"- Covered by a qualifying exercise: {services['covered_by_qualifying_exercise']}",
        f"- Coverage: {services['coverage_percent']}% (full in-scope denominator)",
        f"- Uncovered: {services['uncovered']}",
        f"- Overdue by service cadence: {services['overdue_by_target_cadence']}",
        f"- Valid approved exceptions: {services['valid_approved_exceptions']} (shown separately, not removed from denominator)",
        f"- Missing/stale dependency reviews: {services['stale_or_missing_dependency_reviews']}",
        "",
        "## Observed outcomes",
        "",
        f"- Latest objective outcomes: {json.dumps(report['latest_qualifying_objective_results'], ensure_ascii=False, sort_keys=True)}",
        f"- RTO: {report['rto']['met']}/{report['rto']['measured_services']} measured services met target; pass rate {report['rto']['pass_percent']}%.",
        f"- RPO: {report['rpo']['met']}/{report['rpo']['measured_services']} measured services met target; pass rate {report['rpo']['pass_percent']}%.",
        "",
        "## Improvement actions",
        "",
        f"- Verified closed: {report['actions']['verified_closed']}/{report['actions']['total']} total actions ({report['actions']['verified_closure_percent_of_all_actions']}%).",
        f"- Overdue and not verified: {report['actions']['overdue_unverified']}",
        f"- Marked closed without verification: {report['actions']['closed_without_verification']}",
        "",
        "## Review notes",
        "",
        "This is a portfolio data-quality check, not an assurance opinion. Confirm service scope, qualifying-exercise criteria, target cadence, exception approvals and evidence with accountable owners before reporting.",
        "A blank or unmeasured RTO/RPO is not a pass. A tabletop or component test is not automatically an end-to-end service recovery test. Review each finding and its supporting source.",
        "",
    ]
    (out_dir / "management-review.md").write_text("\n".join(lines), encoding="utf-8")


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    as_of = date.fromisoformat(args.as_of)
    raw = validate_and_normalise(
        read_csv(Path(args.services), SERVICE_FIELDS, "services"),
        read_csv(Path(args.exercises), EXERCISE_FIELDS, "exercises"),
        read_csv(Path(args.actions), ACTION_FIELDS, "actions"),
        read_csv(Path(args.exceptions), EXCEPTION_FIELDS, "exceptions"),
        as_of, args.window_days, args.dependency_freshness_days, args.qualification_policy_id,
    )
    report = audit(raw)
    write_outputs(report, Path(args.out))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Review annual DR programme coverage, freshness and actions.")
    parser.add_argument("--services", required=True, help="Service inventory CSV")
    parser.add_argument("--exercises", required=True, help="Exercise history CSV")
    parser.add_argument("--actions", required=True, help="Corrective-action register CSV")
    parser.add_argument("--exceptions", required=True, help="Risk exception register CSV")
    parser.add_argument("--as-of", default=date.today().isoformat(), help="Reporting date (YYYY-MM-DD)")
    parser.add_argument("--window-days", type=int, default=365)
    parser.add_argument("--dependency-freshness-days", type=int, default=365)
    parser.add_argument("--qualification-policy-id", default=None, help="Stable identifier for the programme owner's qualifying-exercise policy (recommended for period comparison)")
    parser.add_argument("--out", default="programme-audit-out")
    args = parser.parse_args()
    if args.window_days < 1 or args.dependency_freshness_days < 1:
        parser.error("--window-days and --dependency-freshness-days must be positive")
    try:
        report = run_audit(args)
    except (InputError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: value for key, value in report.items() if key != "findings"}, ensure_ascii=False, indent=2))
    print(f"\nWrote programme review to: {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
