import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runbook_preflight import validate_plan


def load_sample(name):
    return json.loads((ROOT / "samples" / name).read_text(encoding="utf-8"))


class RunbookPreflightTests(unittest.TestCase):
    def test_valid_test_failover_passes(self):
        result = validate_plan(load_sample("test-failover.json"))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["step_count"], 8)
        self.assertEqual(result["execution_order"][0], "authorize")
        self.assertEqual(result["execution_order"][-1], "cleanup")

    def test_valid_failback_passes(self):
        result = validate_plan(load_sample("failback.json"))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["execution_order"][-1], "protection")

    def test_missing_isolation_control_fails_test(self):
        plan = load_sample("test-failover.json")
        plan["controls"]["test_network_isolated"] = False
        self.assertTrue(any("test_network_isolated" in e for e in validate_plan(plan)["errors"]))

    def test_missing_required_phase_fails(self):
        plan = load_sample("test-failover.json")
        plan["steps"] = [s for s in plan["steps"] if s["phase"] != "identity"]
        self.assertTrue(any("Required phase is missing" in e for e in validate_plan(plan)["errors"]))

    def test_duplicate_step_id_fails(self):
        plan = load_sample("test-failover.json")
        plan["steps"].append(copy.deepcopy(plan["steps"][0]))
        self.assertTrue(any("Duplicate step id" in e for e in validate_plan(plan)["errors"]))

    def test_unknown_dependency_fails(self):
        plan = load_sample("test-failover.json")
        plan["steps"][1]["depends_on"].append("missing-step")
        self.assertTrue(any("unknown step" in e for e in validate_plan(plan)["errors"]))

    def test_dependency_cycle_fails(self):
        plan = load_sample("test-failover.json")
        plan["steps"][0]["depends_on"] = ["cleanup"]
        self.assertTrue(any("cycle detected" in e for e in validate_plan(plan)["errors"]))

    def test_dependency_must_precede_dependent(self):
        plan = load_sample("test-failover.json")
        plan["steps"][1]["order"] = 5
        self.assertTrue(any("Dependency order violation" in e for e in validate_plan(plan)["errors"]))

    def test_evidence_and_owner_are_required(self):
        plan = load_sample("test-failover.json")
        plan["steps"][0]["owner"] = ""
        plan["steps"][0]["evidence_ref"] = ""
        errors = " ".join(validate_plan(plan)["errors"])
        self.assertIn("owner", errors)
        self.assertIn("evidence_ref", errors)

    def test_three_stop_conditions_are_required(self):
        plan = load_sample("test-failover.json")
        plan["stop_conditions"] = ["One", ""]
        self.assertTrue(any("three explicit" in e for e in validate_plan(plan)["errors"]))

    def test_failback_requires_reverse_replication_and_data_controls(self):
        plan = load_sample("failback.json")
        plan["controls"]["reverse_replication_healthy"] = False
        plan["controls"]["data_reconciliation_plan_approved"] = False
        errors = " ".join(validate_plan(plan)["errors"])
        self.assertIn("reverse_replication_healthy", errors)
        self.assertIn("data_reconciliation_plan_approved", errors)

    def test_invalid_root_and_type_fail(self):
        self.assertEqual(validate_plan([])["status"], "FAIL")
        self.assertEqual(validate_plan({"exercise_type":"real-failover"})["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
