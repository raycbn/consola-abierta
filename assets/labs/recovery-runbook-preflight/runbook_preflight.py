#!/usr/bin/env python3
"""Offline structural and safety-gate validator for DR runbook plans."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_CONTROLS = {
    "test-failover": [
        "change_approved", "test_network_isolated", "production_routes_blocked",
        "primary_workload_remains_authoritative", "rollback_and_stop_conditions_defined",
        "monitoring_and_evidence_ready", "service_owner_available",
    ],
    "failback": [
        "change_approved", "source_target_roles_confirmed", "reverse_replication_healthy",
        "data_reconciliation_plan_approved", "traffic_cutover_plan_approved",
        "rollback_and_stop_conditions_defined", "monitoring_and_evidence_ready",
        "business_owner_available",
    ],
}
ALLOWED_PHASES = {
    "test-failover": {"prepare", "isolation", "identity", "platform", "data", "application", "acceptance", "cleanup"},
    "failback": {"prepare", "reprotect", "data-sync", "cutover", "validation", "resume-protection", "closure"},
}
REQUIRED_PHASES = {
    "test-failover": {"prepare", "isolation", "identity", "acceptance", "cleanup"},
    "failback": {"prepare", "reprotect", "data-sync", "cutover", "validation", "resume-protection"},
}

def validate_plan(plan: Any) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(plan, dict):
        return {"status":"FAIL","errors":["The plan root must be a JSON object."],"warnings":[],"step_count":0,"execution_order":[]}
    exercise_type = plan.get("exercise_type")
    if exercise_type not in REQUIRED_CONTROLS:
        return {"status":"FAIL","errors":["exercise_type must be 'test-failover' or 'failback'."],"warnings":[],"step_count":0,"execution_order":[]}
    controls = plan.get("controls")
    if not isinstance(controls, dict):
        errors.append("controls must be a JSON object.")
        controls = {}
    for key in REQUIRED_CONTROLS[exercise_type]:
        if controls.get(key) is not True:
            errors.append(f"Required safety control is not confirmed true: {key}.")
    stop_conditions = plan.get("stop_conditions")
    if not isinstance(stop_conditions, list) or len([x for x in stop_conditions if isinstance(x,str) and x.strip()]) < 3:
        errors.append("Define at least three explicit, non-empty stop_conditions.")
    raw_steps = plan.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        errors.append("steps must be a non-empty array.")
        raw_steps = []
    by_id: dict[str,dict[str,Any]] = {}
    orders: dict[int,str] = {}
    present_phases: set[str] = set()
    for index, step in enumerate(raw_steps,start=1):
        if not isinstance(step,dict):
            errors.append(f"Step {index} must be a JSON object.")
            continue
        step_id = step.get("id")
        if not isinstance(step_id,str) or not step_id.strip():
            errors.append(f"Step {index} has no valid id.")
            continue
        step_id = step_id.strip()
        if step_id in by_id:
            errors.append(f"Duplicate step id: {step_id}.")
            continue
        by_id[step_id] = step
        for field in ("name","owner","evidence_ref"):
            if not isinstance(step.get(field),str) or not step[field].strip():
                errors.append(f"Step '{step_id}' requires a non-empty {field}.")
        phase = step.get("phase")
        if phase not in ALLOWED_PHASES[exercise_type]:
            errors.append(f"Step '{step_id}' has invalid phase '{phase}'.")
        else:
            present_phases.add(phase)
        order = step.get("order")
        if not isinstance(order,int) or isinstance(order,bool) or order < 1:
            errors.append(f"Step '{step_id}' requires an integer order >= 1.")
        elif order in orders:
            errors.append(f"Duplicate order {order}: steps '{orders[order]}' and '{step_id}'.")
        else:
            orders[order] = step_id
        dependencies = step.get("depends_on",[])
        if not isinstance(dependencies,list) or any(not isinstance(x,str) for x in dependencies):
            errors.append(f"Step '{step_id}' depends_on must be an array of step ids.")
    for phase in sorted(REQUIRED_PHASES[exercise_type] - present_phases):
        errors.append(f"Required phase is missing for {exercise_type}: {phase}.")
    graph = {step_id:[] for step_id in by_id}
    indegree = {step_id:0 for step_id in by_id}
    for step_id,step in by_id.items():
        dependencies = step.get("depends_on",[])
        if not isinstance(dependencies,list):
            continue
        for dependency in dependencies:
            if dependency not in by_id:
                errors.append(f"Step '{step_id}' depends on unknown step '{dependency}'.")
                continue
            if dependency == step_id:
                errors.append(f"Step '{step_id}' cannot depend on itself.")
                continue
            graph[dependency].append(step_id)
            indegree[step_id] += 1
            dep_order = by_id[dependency].get("order")
            own_order = step.get("order")
            if isinstance(dep_order,int) and isinstance(own_order,int) and dep_order >= own_order:
                errors.append(f"Dependency order violation: '{dependency}' must precede '{step_id}'.")
    def sort_key(step_id: str) -> tuple[int,str]:
        value = by_id[step_id].get("order")
        return (value if isinstance(value,int) and not isinstance(value,bool) else 10**9,step_id)
    ready = sorted((step_id for step_id,degree in indegree.items() if degree == 0),key=sort_key)
    execution_order = []
    while ready:
        current = ready.pop(0)
        execution_order.append(current)
        for dependent in graph[current]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
                ready.sort(key=sort_key)
    if len(execution_order) != len(by_id):
        cyclic = sorted(step_id for step_id,degree in indegree.items() if degree > 0)
        errors.append("Dependency cycle detected involving: " + ", ".join(cyclic) + ".")
    if not isinstance(plan.get("plan_name"),str) or not plan["plan_name"].strip():
        errors.append("plan_name is required.")
    if not by_id:
        warnings.append("No usable steps were parsed.")
    return {"status":"PASS" if not errors else "FAIL","plan_name":plan.get("plan_name","Untitled plan"),
            "exercise_type":exercise_type,"step_count":len(by_id),"execution_order":execution_order,
            "errors":errors,"warnings":warnings}

def render_markdown(result: dict[str,Any]) -> str:
    lines = [f"# Runbook preflight report: {result.get('plan_name','Untitled plan')}","",
             f"- **Exercise type:** \`{result.get('exercise_type','unknown')}\`",
             f"- **Result:** **{result['status']}**",f"- **Steps parsed:** {result['step_count']}","",
             "## Dependency-aware execution order",""]
    order = result.get("execution_order",[])
    lines.extend([f"{number}. \`{step_id}\`" for number,step_id in enumerate(order,start=1)] or ["No execution order available."])
    lines.extend(["","## Errors",""])
    lines.extend([f"- {item}" for item in result["errors"]] or ["- None"])
    lines.extend(["","## Warnings",""])
    lines.extend([f"- {item}" for item in result["warnings"]] or ["- None"])
    lines.extend(["","> Offline structural check only. PASS does not authorize a change or prove that any recovery operation works.",""])
    return "\n".join(lines)

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DR runbook gates and dependency ordering (offline only).")
    parser.add_argument("plan",type=Path,help="JSON runbook plan")
    parser.add_argument("--json-out",type=Path,help="Write JSON report")
    parser.add_argument("--markdown-out",type=Path,help="Write Markdown report")
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as exc:
        print(f"Invalid input: {exc}",file=sys.stderr)
        return 2
    result = validate_plan(plan)
    rendered = json.dumps(result,ensure_ascii=False,indent=2) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered,encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.write_text(render_markdown(result),encoding="utf-8")
    print(render_markdown(result))
    return 0 if result["status"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
