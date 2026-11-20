import tempfile
import unittest
from datetime import date
from pathlib import Path

from programme_audit import (
    InputError, audit, read_csv, validate_and_normalise,
    SERVICE_FIELDS, EXERCISE_FIELDS, ACTION_FIELDS, EXCEPTION_FIELDS,
)


def rows():
    services = [
        {"service_id":"S1","service_name":"Payments","criticality":"CRITICAL","in_scope":"true","target_cadence_days":"90","dependency_review_date":"2026-10-01","_row":"2"},
        {"service_id":"S2","service_name":"Identity","criticality":"HIGH","in_scope":"true","target_cadence_days":"180","dependency_review_date":"2025-01-01","_row":"3"},
        {"service_id":"S3","service_name":"Archive","criticality":"LOW","in_scope":"false","target_cadence_days":"365","dependency_review_date":"","_row":"4"},
    ]
    exercises = [
        {"exercise_id":"E1","service_id":"S1","exercise_date":"2026-10-01","exercise_type":"isolated-service-recovery","scope":"application and data path","qualifying_for_recovery_coverage":"true","objective_result":"met","target_rto_minutes":"60","observed_rto_minutes":"55","target_rpo_minutes":"15","observed_rpo_minutes":"12","evidence_reference":"EV-01","_row":"2"},
        {"exercise_id":"E2","service_id":"S2","exercise_date":"2026-09-01","exercise_type":"tabletop","scope":"decision roles only","qualifying_for_recovery_coverage":"false","objective_result":"partially_met","target_rto_minutes":"","observed_rto_minutes":"","target_rpo_minutes":"","observed_rpo_minutes":"","evidence_reference":"EV-02","_row":"3"},
    ]
    actions = [
        {"action_id":"A1","service_id":"S1","finding_id":"F1","owner":"Team A","due_date":"2026-10-31","status":"CLOSED","verification_date":"2026-11-01","verification_evidence":"EV-CLOSE-1","_row":"2"},
        {"action_id":"A2","service_id":"S1","finding_id":"F2","owner":"Team B","due_date":"2026-10-31","status":"CLOSED","verification_date":"","verification_evidence":"","_row":"3"},
        {"action_id":"A3","service_id":"S2","finding_id":"F3","owner":"Team C","due_date":"2026-10-01","status":"OPEN","verification_date":"","verification_evidence":"","_row":"4"},
    ]
    exceptions = [
        {"exception_id":"X1","service_id":"S2","risk_owner":"Owner B","rationale":"Migration window","compensating_controls":"Daily backup review","approver":"Risk forum","approval_date":"2026-10-01","expiry_date":"2026-12-01","status":"APPROVED","_row":"2"},
    ]
    return services, exercises, actions, exceptions


class ProgrammeAuditTests(unittest.TestCase):
    def build(self, data=None, **kwargs):
        services, exercises, actions, exceptions = data or rows()
        defaults = dict(as_of=date(2026, 11, 20), window_days=365, dependency_freshness_days=365)
        defaults.update(kwargs)
        return validate_and_normalise(services, exercises, actions, exceptions, **defaults)

    def test_full_in_scope_denominator_includes_exception(self):
        report = audit(self.build())
        self.assertEqual(report["services"]["in_scope"], 2)
        self.assertEqual(report["services"]["covered_by_qualifying_exercise"], 1)
        self.assertEqual(report["services"]["coverage_percent"], 50.0)
        self.assertEqual(report["services"]["valid_approved_exceptions"], 1)

    def test_tabletop_does_not_automatically_count_as_technical_recovery(self):
        report = audit(self.build())
        self.assertEqual(report["services"]["uncovered"], 1)

    def test_excluded_service_not_in_denominator(self):
        report = audit(self.build())
        self.assertEqual(report["services"]["in_scope"], 2)

    def test_rto_and_rpo_only_count_measured_values(self):
        report = audit(self.build())
        self.assertEqual(report["rto"]["measured_services"], 1)
        self.assertEqual(report["rto"]["pass_percent"], 100.0)
        self.assertEqual(report["rpo"]["measured_services"], 1)

    def test_closed_action_requires_verification_evidence(self):
        report = audit(self.build())
        self.assertEqual(report["actions"]["verified_closed"], 1)
        self.assertEqual(report["actions"]["closed_without_verification"], 1)

    def test_overdue_open_action_is_reported(self):
        report = audit(self.build())
        self.assertEqual(report["actions"]["overdue_unverified"], 2)

    def test_old_dependency_review_is_reported(self):
        report = audit(self.build())
        self.assertEqual(report["services"]["stale_or_missing_dependency_reviews"], 1)

    def test_duplicate_service_id_rejected(self):
        data = rows()
        data[0].append(dict(data[0][0]))
        with self.assertRaises(InputError):
            self.build(data)

    def test_unknown_service_reference_rejected(self):
        data = rows()
        data[1][0]["service_id"] = "UNKNOWN"
        with self.assertRaises(InputError):
            self.build(data)

    def test_invalid_qualifying_boolean_rejected(self):
        data = rows()
        data[1][0]["qualifying_for_recovery_coverage"] = "sometimes"
        with self.assertRaises(InputError):
            self.build(data)

    def test_negative_recovery_target_rejected(self):
        data = rows()
        data[1][0]["target_rto_minutes"] = "-1"
        with self.assertRaises(InputError):
            self.build(data)

    def test_exercise_after_as_of_rejected(self):
        data = rows()
        data[1][0]["exercise_date"] = "2026-12-01"
        with self.assertRaises(InputError):
            self.build(data)

    def test_expired_exception_not_counted_as_valid(self):
        data = rows()
        data[3][0]["expiry_date"] = "2026-11-01"
        report = audit(self.build(data))
        self.assertEqual(report["services"]["valid_approved_exceptions"], 0)
        self.assertTrue(any(f["finding"] == "EXCEPTION_EXPIRED" for f in report["findings"]))

    def test_expired_exception_dates_must_be_ordered(self):
        data = rows()
        data[3][0]["approval_date"] = "2026-12-01"
        with self.assertRaises(InputError):
            self.build(data)

    def test_missing_columns_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("service_id,service_name\nS1,Payments\n", encoding="utf-8")
            with self.assertRaises(InputError):
                read_csv(path, SERVICE_FIELDS, "services")

    def test_outputs_include_limits_and_findings(self):
        report = audit(self.build())
        self.assertTrue(report["limitations"])
        self.assertTrue(any(f["finding"] == "NO_QUALIFYING_COVERAGE" for f in report["findings"]))


if __name__ == "__main__":
    unittest.main()
