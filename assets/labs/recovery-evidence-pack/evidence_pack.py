#!/usr/bin/env python3
"""Offline DR evidence register and AAR/IP draft builder."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

VALID_OBJECTIVE_RESULTS = {"met", "partially_met", "not_met", "not_tested"}
VALID_OBSERVATION_TYPES = {"strength", "improvement"}
VALID_ACTION_STATUSES = {"open", "in_progress", "blocked", "closed"}
VALID_PRIORITIES = {"P1", "P2", "P3"}

def is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())

def parse_aware(value: Any, label: str, errors: list[str]) -> datetime | None:
    if not is_text(value):
        errors.append(f"{label} must be an ISO-8601 timestamp with timezone.")
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{label} is not a valid ISO-8601 timestamp: {value!r}.")
        return None
    if result.tzinfo is None or result.utcoffset() is None:
        errors.append(f"{label} must include an explicit timezone offset.")
        return None
    return result

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def safe_path(root: Path, raw: Any, evidence_id: str, errors: list[str]) -> Path | None:
    if not is_text(raw):
        errors.append(f"Evidence '{evidence_id}' requires a non-empty relative path.")
        return None
    value = raw.strip()
    rel = PurePosixPath(value)
    if rel.is_absolute() or ".." in rel.parts or "\\" in value or not rel.parts:
        errors.append(f"Evidence '{evidence_id}' has an unsafe path: {value!r}.")
        return None
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            errors.append(f"Evidence '{evidence_id}' uses a symlink path, which is not accepted: {value!r}.")
            return None
    try:
        resolved = (root / Path(*rel.parts)).resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        errors.append(f"Evidence '{evidence_id}' is missing or resolves outside the evidence directory: {value!r}.")
        return None
    if not resolved.is_file():
        errors.append(f"Evidence '{evidence_id}' is not a regular file: {value!r}.")
        return None
    return resolved

def validate_exercise(data: Any, evidence_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(data, dict):
        return {"status":"FAIL","errors":["Input root must be a JSON object."],"warnings":[],"evidence":[]}
    exercise = data.get("exercise")
    if not isinstance(exercise, dict):
        errors.append("exercise must be a JSON object.")
        exercise = {}
    for field in ("id","title","date","scope","scenario","lead_role","sponsor_role"):
        if not is_text(exercise.get(field)):
            errors.append(f"exercise.{field} is required.")
    if is_text(exercise.get("date")):
        try:
            date.fromisoformat(exercise["date"])
        except ValueError:
            errors.append("exercise.date must use YYYY-MM-DD.")
    start = parse_aware(exercise.get("start_at"),"exercise.start_at",errors)
    end = parse_aware(exercise.get("end_at"),"exercise.end_at",errors)
    if start and end and start > end:
        errors.append("exercise.start_at must not be later than exercise.end_at.")

    objectives = data.get("objectives")
    if not isinstance(objectives,list) or not objectives:
        errors.append("objectives must be a non-empty array.")
        objectives = []
    objective_ids: set[str] = set()
    for i,item in enumerate(objectives,1):
        if not isinstance(item,dict):
            errors.append(f"Objective {i} must be an object.")
            continue
        oid = item.get("id")
        if not is_text(oid):
            errors.append(f"Objective {i} requires an id.")
            continue
        oid = oid.strip()
        if oid in objective_ids:
            errors.append(f"Duplicate objective id: {oid}.")
        objective_ids.add(oid)
        for field in ("statement","success_criteria"):
            if not is_text(item.get(field)):
                errors.append(f"Objective '{oid}' requires {field}.")
        if item.get("result") not in VALID_OBJECTIVE_RESULTS:
            errors.append(f"Objective '{oid}' result must be one of: {', '.join(sorted(VALID_OBJECTIVE_RESULTS))}.")

    raw_evidence = data.get("evidence")
    if not isinstance(raw_evidence,list) or not raw_evidence:
        errors.append("evidence must be a non-empty array.")
        raw_evidence = []
    evidence_ids: set[str] = set()
    evidence_records: list[dict[str,Any]] = []
    root = evidence_dir.resolve()
    for i,item in enumerate(raw_evidence,1):
        if not isinstance(item,dict):
            errors.append(f"Evidence record {i} must be an object.")
            continue
        eid = item.get("id")
        if not is_text(eid):
            errors.append(f"Evidence record {i} requires an id.")
            continue
        eid = eid.strip()
        if eid in evidence_ids:
            errors.append(f"Duplicate evidence id: {eid}.")
        evidence_ids.add(eid)
        for field in ("description","collected_by_role","source","classification"):
            if not is_text(item.get(field)):
                errors.append(f"Evidence '{eid}' requires {field}.")
        parse_aware(item.get("collected_at"),f"evidence '{eid}'.collected_at",errors)
        refs = item.get("objective_ids",[])
        if not isinstance(refs,list) or any(not isinstance(x,str) for x in refs):
            errors.append(f"Evidence '{eid}' objective_ids must be an array of strings.")
            refs = []
        for ref in refs:
            if ref not in objective_ids:
                errors.append(f"Evidence '{eid}' references unknown objective '{ref}'.")
        path = safe_path(root,item.get("path"),eid,errors)
        if path:
            evidence_records.append({
                "id":eid,"path":str(item["path"]).replace("\\","/"),
                "description":item.get("description",""),
                "collected_by_role":item.get("collected_by_role",""),
                "collected_at":item.get("collected_at",""),
                "source":item.get("source",""),"classification":item.get("classification",""),
                "objective_ids":refs,"size_bytes":path.stat().st_size,"sha256":sha256_file(path)
            })

    observations = data.get("observations")
    if not isinstance(observations,list):
        errors.append("observations must be an array.")
        observations = []
    observation_ids: set[str] = set()
    observation_map: dict[str,dict[str,Any]] = {}
    for i,item in enumerate(observations,1):
        if not isinstance(item,dict):
            errors.append(f"Observation {i} must be an object.")
            continue
        oid=item.get("id")
        if not is_text(oid):
            errors.append(f"Observation {i} requires an id.")
            continue
        oid=oid.strip()
        if oid in observation_ids:
            errors.append(f"Duplicate observation id: {oid}.")
        observation_ids.add(oid)
        observation_map[oid]=item
        if item.get("type") not in VALID_OBSERVATION_TYPES:
            errors.append(f"Observation '{oid}' type must be strength or improvement.")
        for field in ("objective_id","statement","impact","evaluator_role"):
            if not is_text(item.get(field)):
                errors.append(f"Observation '{oid}' requires {field}.")
        if item.get("objective_id") not in objective_ids:
            errors.append(f"Observation '{oid}' references an unknown objective.")
        refs=item.get("evidence_ids")
        if not isinstance(refs,list) or not refs:
            errors.append(f"Observation '{oid}' requires at least one evidence_id.")
            refs=[]
        for ref in refs:
            if ref not in evidence_ids:
                errors.append(f"Observation '{oid}' references unknown evidence '{ref}'.")
        if item.get("type")=="improvement" and not is_text(item.get("recommendation")):
            errors.append(f"Improvement observation '{oid}' requires a recommendation separate from the observation.")

    actions=data.get("actions")
    if not isinstance(actions,list):
        errors.append("actions must be an array.")
        actions=[]
    action_ids:set[str]=set()
    action_by_observation:dict[str,int]={}
    for i,item in enumerate(actions,1):
        if not isinstance(item,dict):
            errors.append(f"Action {i} must be an object.")
            continue
        aid=item.get("id")
        if not is_text(aid):
            errors.append(f"Action {i} requires an id.")
            continue
        aid=aid.strip()
        if aid in action_ids:
            errors.append(f"Duplicate action id: {aid}.")
        action_ids.add(aid)
        oid=item.get("observation_id")
        obs=observation_map.get(oid)
        if not obs:
            errors.append(f"Action '{aid}' references unknown observation '{oid}'.")
        elif obs.get("type")!="improvement":
            errors.append(f"Action '{aid}' must reference an improvement observation, not a strength.")
        action_by_observation[oid]=action_by_observation.get(oid,0)+1
        for field in ("action","owner_role","due_date","verification_method"):
            if not is_text(item.get(field)):
                errors.append(f"Action '{aid}' requires {field}.")
        if is_text(item.get("due_date")):
            try:
                date.fromisoformat(item["due_date"])
            except ValueError:
                errors.append(f"Action '{aid}' due_date must use YYYY-MM-DD.")
        if item.get("status") not in VALID_ACTION_STATUSES:
            errors.append(f"Action '{aid}' has invalid status.")
        if item.get("priority") not in VALID_PRIORITIES:
            errors.append(f"Action '{aid}' priority must be P1, P2 or P3.")
        closure=item.get("closure_evidence_ids",[])
        if not isinstance(closure,list):
            errors.append(f"Action '{aid}' closure_evidence_ids must be an array.")
            closure=[]
        if item.get("status")=="closed" and not closure:
            errors.append(f"Closed action '{aid}' requires closure evidence.")
        for ref in closure:
            if ref not in evidence_ids:
                errors.append(f"Action '{aid}' references unknown closure evidence '{ref}'.")
    for oid,obs in observation_map.items():
        if obs.get("type")=="improvement" and not action_by_observation.get(oid):
            errors.append(f"Improvement observation '{oid}' has no linked corrective action.")

    participants=data.get("participants",[])
    if not isinstance(participants,list):
        errors.append("participants must be an array.")
        participants=[]
    for i,person in enumerate(participants,1):
        if not isinstance(person,dict) or not is_text(person.get("role")) or not is_text(person.get("team")):
            errors.append(f"Participant {i} requires role and team; avoid adding unnecessary personal data.")
    if not observations:
        warnings.append("No observations are recorded; explain why the exercise produced none before issuing an AAR/IP.")
    if not actions:
        warnings.append("No corrective actions are recorded; ensure every improvement is dispositioned before approval.")
    if any(x.get("status")!="closed" for x in actions if isinstance(x,dict)):
        warnings.append("Open actions remain; the AAR/IP should not describe the improvement programme as complete.")
    warnings.append("SHA-256 fingerprints detect file drift after collection; they do not prove authorship, trusted time, approval or forensic chain of custody.")
    warnings.append("Collection timestamps and evidence descriptions are operator-supplied metadata and require independent review.")
    return {"status":"PASS" if not errors else "FAIL","exercise_id":exercise.get("id",""),
        "evidence_count":len(evidence_records),"observation_count":len(observations),"action_count":len(actions),
        "errors":errors,"warnings":warnings,"evidence":evidence_records}

def render_evidence_register(records: list[dict[str,Any]], output: Path) -> None:
    fields=["id","path","description","source","classification","collected_by_role","collected_at","objective_ids","size_bytes","sha256"]
    with output.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for record in records:
            row=dict(record)
            row["objective_ids"]=";".join(record["objective_ids"])
            writer.writerow({key:row.get(key,"") for key in fields})

def render_aar(data: dict[str,Any], result: dict[str,Any]) -> str:
    ex=data["exercise"]
    lines=[
        f"# After-Action Report / Improvement Plan — {ex['title']}","",
        "> Draft generated from structured exercise metadata. Review, approve and classify it before distribution.","",
        "## 1. Exercise overview","",
        f"- Exercise ID: {ex['id']}",f"- Date: {ex['date']}",f"- Window: {ex['start_at']} to {ex['end_at']}",
        f"- Scope: {ex['scope']}",f"- Scenario: {ex['scenario']}",f"- Lead role: {ex['lead_role']}",f"- Sponsor role: {ex['sponsor_role']}","",
        "## 2. Executive summary","",ex.get("executive_summary","[Complete the executive summary after evaluator review.]"),"",
        "## 3. Objectives and evaluation","","| Objective | Success criteria | Result |","|---|---|---|"
    ]
    for item in data["objectives"]:
        lines.append(f"| {item['id']} — {item['statement']} | {item['success_criteria']} | {item['result']} |")
    lines.extend(["","## 4. Evidence and chronology","","The detailed evidence register is in evidence-register.csv; the machine-readable record is in manifest.json.",""])
    for record in result["evidence"]:
        lines.append(f"- {record['id']} — {record['path']} — {record['description']} (collected by {record['collected_by_role']} at {record['collected_at']}; SHA-256 {record['sha256']}).")
    lines.extend(["","## 5. Findings and observations","","Keep observed condition separate from impact and recommendation.",""])
    for obs in data["observations"]:
        lines.extend([f"### {obs['id']} — {obs['type']}","",f"- Objective: {obs['objective_id']}",f"- Observation: {obs['statement']}",f"- Impact: {obs['impact']}",f"- Evaluator role: {obs['evaluator_role']}",f"- Evidence: {', '.join(obs['evidence_ids'])}"])
        if obs.get("recommendation"):
            lines.append(f"- Recommendation: {obs['recommendation']}")
        lines.append("")
    lines.extend(["## 6. Improvement plan","","| Action | Finding | Priority | Owner role | Due date | Status | Verification method | Closure evidence |","|---|---|---|---|---|---|---|---|"])
    for action in data["actions"]:
        lines.append("| {id} — {action} | {observation_id} | {priority} | {owner_role} | {due_date} | {status} | {verification_method} | {closure} |".format(**action,closure=", ".join(action.get("closure_evidence_ids",[])) or "Pending"))
    lines.extend(["","## 7. Decisions, residual risks and acceptance","",ex.get("decisions_and_risks","[Record decisions, accepted residual risks, approvers and review dates.]"),"","## 8. Approval and distribution","","- Operational reviewer: [Role / name in controlled copy]","- Service owner acceptance: [Pending / approved / rejected]","- Risk owner: [Role / name in controlled copy]","- Approval reference: [Change / meeting / record ID]","- Distribution and classification: [Need-to-know audience / classification]","","## 9. Limitations","","This draft uses operator-supplied metadata. It does not prove the exercise occurred as described, evidence authenticity or action effectiveness. Human review and formal approval are required.",""])
    return "\n".join(lines)

def build_pack(input_path: Path, evidence_dir: Path, output_dir: Path) -> dict[str,Any]:
    try:
        data=json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as exc:
        return {"status":"FAIL","errors":[f"Cannot read JSON input: {exc}"],"warnings":[],"evidence":[]}
    result=validate_exercise(data,evidence_dir)
    if result["status"]!="PASS":
        return result
    output_dir.mkdir(parents=True,exist_ok=True)
    generated=datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest={"schema_version":"1.0","generated_at_utc":generated,"exercise":data["exercise"],
        "objectives":data["objectives"],"evidence":result["evidence"],
        "limitations":["File fingerprints are not a digital signature or trusted timestamp.","The manifest references evidence files; it does not copy or preserve them.","Operator-supplied metadata is not independently authenticated."]}
    raw=(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n").encode("utf-8")
    (output_dir/"manifest.json").write_bytes(raw)
    (output_dir/"manifest.sha256").write_text(hashlib.sha256(raw).hexdigest()+"  manifest.json\n",encoding="utf-8")
    render_evidence_register(result["evidence"],output_dir/"evidence-register.csv")
    (output_dir/"AAR-IP.md").write_text(render_aar(data,result),encoding="utf-8")
    report={k:v for k,v in result.items() if k!="evidence"}
    report["generated_at_utc"]=generated
    (output_dir/"validation-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report

def main() -> int:
    parser=argparse.ArgumentParser(description="Build a DR evidence index and AAR/IP draft (offline).")
    parser.add_argument("input",type=Path)
    parser.add_argument("--evidence-dir",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=Path("evidence-pack-out"))
    args=parser.parse_args()
    report=build_pack(args.input,args.evidence_dir,args.out)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if report["status"]!="PASS":
        return 1
    print(f"\nEvidence pack index written to: {args.out.resolve()}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
