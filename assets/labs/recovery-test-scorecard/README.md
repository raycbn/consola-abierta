# Recovery Test Scorecard — Lab 01

An offline, dependency-free Python tool for checking manually recorded disaster-recovery exercise objectives, milestone chronology and evidence references.

## What it does

- Calculates observed RTO from \`service_restored_at - disruption_started_at\`.
- Calculates observed RPO from \`disruption_started_at - last_recoverable_point_at\`.
- Reports business-acceptance lag separately. It is not silently folded into RTO.
- Checks that milestones have chronological timestamps, an owner and an evidence reference.
- Produces JSON and Markdown reports with separate RTO, RPO and evidence outcomes.

## What it does not do

This is a scorecard, not a failover orchestrator or a restore validator. It does not connect to production, cloud APIs, hypervisors, backup systems, databases or monitoring platforms. It does not verify that a restore point is usable, that the data is consistent, that security controls work, or that a business process is functionally correct. Those checks must be performed and evidenced by the relevant technical and service owners.

## Requirements

- Python 3.10 or later.
- No third-party packages.
- Input timestamps must be ISO-8601 and include a timezone (\`Z\` or an explicit offset).

## Run

\`\`\`bash
python recovery_scorecard.py sample-test.json --json-out report.json --markdown-out report.md
\`\`\`

Exit codes: \`0\` means all configured objectives and evidence checks pass; \`1\` means a valid report was generated with one or more failed checks; \`2\` means the input could not be parsed or is inconsistent.

The sample is synthetic. It deliberately passes the 90-minute RTO target and fails the 15-minute RPO target: an exercise must not be called successful merely because one recovery objective passes.

## Input model

- \`objectives.rto_minutes\` and \`objectives.rpo_minutes\`: approved targets, not estimates invented during the test.
- \`timestamps.disruption_started_at\`: the agreed start of the service interruption.
- \`timestamps.last_recoverable_point_at\`: the latest point that has been confirmed recoverable and consistent, not merely the time a backup job reported success.
- \`timestamps.service_restored_at\`: the agreed minimum service has been restored and the defined technical checks have passed.
- \`timestamps.business_accepted_at\`: optional, recorded separately from technical service restoration.
- \`milestones\`: chronological events with a named owner and a reference to evidence stored in the organisation's approved repository. Do not place credentials, customer data or confidential material in this JSON file.

Agree the timing boundaries, recovery scope, safety rules, rollback/cleanup procedure and evidence location before running any real exercise.