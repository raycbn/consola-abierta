import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recovery_scorecard import ScorecardError, analyse, markdown_report  # noqa: E402


class RecoveryScorecardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = json.loads((ROOT / "sample-test.json").read_text(encoding="utf-8"))

    def test_sample_passes_rto_but_fails_rpo(self):
        result = analyse(self.sample)
        self.assertEqual(result["checks"]["rto"], "PASS")
        self.assertEqual(result["checks"]["rpo"], "FAIL")
        self.assertEqual(result["observed"]["rto_seconds"], 4980)
        self.assertEqual(result["observed"]["rpo_seconds"], 1080)
        self.assertEqual(result["checks"]["overall"], "FAIL")

    def test_acceptance_lag_is_separate_from_rto(self):
        result = analyse(self.sample)
        self.assertEqual(result["observed"]["business_acceptance_lag_seconds"], 480)
        self.assertEqual(result["observed"]["rto_seconds"], 4980)

    def test_missing_timezone_is_rejected(self):
        data = json.loads(json.dumps(self.sample))
        data["timestamps"]["disruption_started_at"] = "2026-11-20T08:00:00"
        with self.assertRaises(ScorecardError):
            analyse(data)

    def test_recovery_point_after_disruption_is_rejected(self):
        data = json.loads(json.dumps(self.sample))
        data["timestamps"]["last_recoverable_point_at"] = "2026-11-20T08:01:00Z"
        with self.assertRaises(ScorecardError):
            analyse(data)

    def test_milestones_must_be_chronological(self):
        data = json.loads(json.dumps(self.sample))
        data["milestones"][2]["at"] = "2026-11-20T08:30:00Z"
        with self.assertRaises(ScorecardError):
            analyse(data)

    def test_missing_evidence_fails_evidence_check(self):
        data = json.loads(json.dumps(self.sample))
        data["milestones"][1]["evidence"] = ""
        result = analyse(data)
        self.assertEqual(result["checks"]["evidence_completeness"], "FAIL")
        self.assertTrue(any("missing evidence" in x.lower() for x in result["evidence_issues"]))

    def test_markdown_report_contains_separate_objectives(self):
        report = markdown_report(analyse(self.sample))
        self.assertIn("**RTO:** PASS", report)
        self.assertIn("**RPO:** FAIL", report)
        self.assertIn("Acceptance lag", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
