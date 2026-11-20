# DR Programme Coverage Auditor — Lab 04

Offline, standard-library Python tool to review a synthetic annual Disaster Recovery exercise programme.

## What it checks
- Service inventory, scope, criticality, risk-based target cadence and dependency-review date.
- Exercise history, including the operator-declared qualifying flag for technical recovery coverage.
- Latest qualifying exercise in a configurable reporting window.
- Service coverage and cadence freshness, reported against the full in-scope service denominator.
- RTO/RPO outcomes only where numeric target and observed values exist.
- Approved exceptions with owners, rationale, compensating controls and expiry dates.
- Corrective-action ageing and evidence-backed closure.

## Requirements
- Python 3.10+
- Standard library only
- Offline; no credentials, APIs, cloud access or infrastructure connectivity.

## Run
From this directory:

\`\`\`bash
python programme_audit.py --services samples/services.csv --exercises samples/exercises.csv --actions samples/actions.csv --exceptions samples/exceptions.csv --as-of 2026-11-20 --window-days 365 --out programme-audit-out
python -m unittest discover -s tests -v
\`\`\`

The tool writes:
- \`programme-summary.json\`
- \`service-findings.csv\`
- \`management-review.md\`

## Interpret the output carefully
- The coverage denominator is every in-scope service. Valid exceptions are shown separately and do **not** silently remove services from the denominator.
- \`qualifying_for_recovery_coverage\` is a programme-owner classification against an approved definition. The tool cannot decide whether an exercise was sufficiently realistic or independently validate its scope.
- A tabletop, walkthrough or component test should not be counted as an end-to-end technical service recovery unless the programme's documented criteria explicitly justify it.
- RTO/RPO pass rates use only the latest qualifying exercise with the relevant numeric target and observed value. Missing measurements are not passes and are excluded from that metric's denominator.
- An action is counted as verified closed only when status is CLOSED and both verification date and verification evidence are present.
- Exception expiry is checked against the supplied \`--as-of\` date. The tool does not approve risk or judge the adequacy of compensating controls.
- The reporting window is configurable. It is not a recommendation that all services must be exercised annually.

## Deliberate limitations
This is a data-quality and portfolio-review helper, not a DR orchestration tool, CMDB, BIA engine, risk engine, GRC platform, audit opinion or certification. It does not test infrastructure, backups, replication, application behaviour, restore integrity, real RTO/RPO or business acceptance. It does not infer criticality or prescribe cadence. Review and formally approve the inputs and conclusions.

All sample records are synthetic. Do not place production hostnames, IPs, customer data, credentials or confidential recovery details in the sample files.
