import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evidence_pack import build_pack, validate_exercise


class EvidencePackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.evidence_dir = self.root / "evidence"
        shutil.copytree(ROOT / "samples" / "evidence", self.evidence_dir)
        self.data = json.loads((ROOT / "samples" / "exercise.json").read_text(encoding="utf-8"))

    def tearDown(self):
        self.temp.cleanup()

    def test_synthetic_sample_passes(self):
        result = validate_exercise(self.data, self.evidence_dir)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["evidence_count"], 3)
        self.assertEqual(result["observation_count"], 2)
        self.assertEqual(result["action_count"], 1)

    def test_pack_generates_expected_outputs(self):
        input_path = self.root / "exercise.json"
        input_path.write_text(json.dumps(self.data), encoding="utf-8")
        output = self.root / "output"
        result = build_pack(input_path, self.evidence_dir, output)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue((output / "manifest.json").is_file())
        self.assertTrue((output / "manifest.sha256").is_file())
        self.assertTrue((output / "evidence-register.csv").is_file())
        self.assertTrue((output / "AAR-IP.md").is_file())
        self.assertTrue((output / "validation-report.json").is_file())

    def test_manifest_hash_matches_evidence(self):
        input_path = self.root / "exercise.json"
        input_path.write_text(json.dumps(self.data), encoding="utf-8")
        output = self.root / "output"
        build_pack(input_path, self.evidence_dir, output)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        record = next(item for item in manifest["evidence"] if item["id"] == "EV-001")
        actual = hashlib.sha256((self.evidence_dir / record["path"]).read_bytes()).hexdigest()
        self.assertEqual(record["sha256"], actual)

    def test_evidence_files_are_referenced_not_copied(self):
        input_path = self.root / "exercise.json"
        input_path.write_text(json.dumps(self.data), encoding="utf-8")
        output = self.root / "output"
        build_pack(input_path, self.evidence_dir, output)
        names = {p.name for p in output.iterdir()}
        self.assertNotIn("EV-001-chronology.csv", names)

    def test_missing_evidence_fails(self):
        (self.evidence_dir / "EV-003-service-acceptance.txt").unlink()
        result = validate_exercise(self.data, self.evidence_dir)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("missing" in error for error in result["errors"]))

    def test_absolute_path_is_rejected(self):
        self.data["evidence"][0]["path"] = str((self.evidence_dir / "EV-001-chronology.csv").resolve())
        result = validate_exercise(self.data, self.evidence_dir)
        self.assertTrue(any("unsafe path" in error for error in result["errors"]))

    def test_parent_traversal_is_rejected(self):
        self.data["evidence"][0]["path"] = "../outside.txt"
        result = validate_exercise(self.data, self.evidence_dir)
        self.assertTrue(any("unsafe path" in error for error in result["errors"]))

    def test_symlink_evidence_is_rejected(self):
        outside = self.root / "outside.txt"
        outside.write_text("not in evidence root", encoding="utf-8")
        link = self.evidence_dir / "linked.txt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks are not available on this platform")
        self.data["evidence"][0]["path"] = "linked.txt"
        result = validate_exercise(self.data, self.evidence_dir)
        self.assertTrue(any("symlink" in error for error in result["errors"]))

    def test_collection_timestamp_requires_timezone(self):
        self.data["evidence"][0]["collected_at"] = "2026-11-19T10:35:00"
        result = validate_exercise(self.data, self.evidence_dir)
        self.assertTrue(any("explicit timezone" in error for error in result["errors"]))

    def test_duplicate_evidence_id_fails(self):
        self.data["evidence"].append(copy.deepcopy(self.data["evidence"][0]))
        result = validate_exercise(self.data, self.evidence_dir)
        self.assertTrue(any("Duplicate evidence id" in error for error in result["errors"]))

    def test_unknown_objective_reference_fails(self):
        self.data["evidence"][0]["objective_ids"] = ["OBJ-UNKNOWN"]
        result = validate_exercise(self.data, self.evidence_dir)
        self.assertTrue(any("unknown objective" in error for error in result["errors"]))

    def test_observation_requires_evidence(self):
        self.data["observations"][0]["evidence_ids"] = []
        result = validate_exercise(self.data, self.evidence_dir)
        self.assertTrue(any("requires at least one evidence_id" in error for error in result["errors"]))

    def test_unknown_evidence_reference_fails(self):
        self.data["observations"][0]["evidence_ids"] = ["EV-DOES-NOT-EXIST"]
        result = validate_exercise(self.data, self.evidence_dir)
        self.assertTrue(any("references unknown evidence" in error for error in result["errors"]))

    def test_improvement_requires_linked_action(self):
        self.data["actions"] = []
        result = validate_exercise(self.data, self.evidence_dir)
        self.assertTrue(any("has no linked corrective action" in error for error in result["errors"]))

    def test_action_cannot_reference_strength(self):
        self.data["actions"][0]["observation_id"] = "OBS-001"
        result = validate_exercise(self.data, self.evidence_dir)
        self.assertTrue(any("not a strength" in error for error in result["errors"]))

    def test_closed_action_requires_closure_evidence(self):
        self.data["actions"][0]["status"] = "closed"
        self.data["actions"][0]["closure_evidence_ids"] = []
        result = validate_exercise(self.data, self.evidence_dir)
        self.assertTrue(any("requires closure evidence" in error for error in result["errors"]))

    def test_unknown_closure_evidence_fails(self):
        self.data["actions"][0]["status"] = "closed"
        self.data["actions"][0]["closure_evidence_ids"] = ["EV-NOT-THERE"]
        result = validate_exercise(self.data, self.evidence_dir)
        self.assertTrue(any("unknown closure evidence" in error for error in result["errors"]))

    def test_action_due_date_must_be_iso_date(self):
        self.data["actions"][0]["due_date"] = "04/12/2026"
        result = validate_exercise(self.data, self.evidence_dir)
        self.assertTrue(any("due_date must use YYYY-MM-DD" in error for error in result["errors"]))

    def test_invalid_objective_result_fails(self):
        self.data["objectives"][0]["result"] = "great"
        result = validate_exercise(self.data, self.evidence_dir)
        self.assertTrue(any("result must be one of" in error for error in result["errors"]))

    def test_start_after_end_fails(self):
        self.data["exercise"]["start_at"] = "2026-11-19T11:00:00+01:00"
        result = validate_exercise(self.data, self.evidence_dir)
        self.assertTrue(any("start_at must not be later" in error for error in result["errors"]))

    def test_empty_objectives_fail(self):
        self.data["objectives"] = []
        result = validate_exercise(self.data, self.evidence_dir)
        self.assertTrue(any("non-empty array" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
