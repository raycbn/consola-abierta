# Recovery Runbook Preflight — Lab 02

An offline validator for the structure and safety gates of a Disaster Recovery **test-failover** or **failback** runbook.

## What it checks
- Exercise type and mandatory approval, isolation, replication and data-reconciliation controls.
- At least three explicit stop conditions.
- Required operational phases for each exercise type.
- Unique step IDs and sequence numbers.
- Dependency references, cycles and explicit ordering.
- A named owner and evidence reference for every step.
- A dependency-aware execution order for review.

## Run it
Requires Python 3.10 or later. No third-party packages or network access are required.

\`\`\`bash
python runbook_preflight.py samples/test-failover.json --json-out test-failover-report.json --markdown-out test-failover-report.md
python runbook_preflight.py samples/failback.json --json-out failback-report.json --markdown-out failback-report.md
python -m unittest discover -s tests -v
\`\`\`

Exit codes: \`0\` = preflight PASS, \`1\` = plan has errors, \`2\` = invalid input or unreadable file.

## Scope and safety boundary
This is a **plan linter**, not a DR orchestration product. It reads JSON supplied by the operator and writes a report. It does not connect to Azure Site Recovery, AWS Elastic Disaster Recovery, VMware, hypervisors, storage, identity, DNS, firewalls, monitoring or production. It does not create or delete resources, start VMs, replicate data, route traffic, validate application behavior or prove data integrity.

A PASS means only that the submitted plan passed these structural checks. It is not a change approval, operational authorization, proof that a network is isolated, or evidence that a recovery will work. The controls are declarations in the JSON and must be independently verified and approved by responsible teams. Adapt the control set, phases, sequence and rollback procedure to the platform and service architecture. Do not use synthetic sample values as a production runbook.

## Sample plans
- \`samples/test-failover.json\`: isolated multi-tier test recovery.
- \`samples/failback.json\`: reverse protection, data synchronization/reconciliation, planned cutover, service acceptance and restored protection.

All names and evidence references are fictional.
