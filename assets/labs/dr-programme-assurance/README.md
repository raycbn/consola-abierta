# DR Programme Coverage Auditor — Lab 04

Offline, standard-library Python tool and browser-only dashboard to review a Disaster Recovery exercise programme.

## What it checks
- Service inventory, scope, criticality, risk-based target cadence and dependency-review date.
- Exercise history, including the operator-declared qualifying flag for technical recovery coverage.
- Latest qualifying exercise in a configurable reporting window.
- Service coverage and cadence freshness, reported against the full in-scope service denominator.
- RTO/RPO outcomes only where numeric target and observed values exist.
- Approved exceptions with owners, rationale, compensating controls and expiry dates.
- Corrective-action ageing and evidence-backed closure.

## Requirements
- Python 3.10+ for the audit tool.
- A modern browser for the dashboard.
- Python standard library only; no external browser libraries.
- Offline; no credentials, APIs, cloud access or infrastructure connectivity.

## Run the audit
From this directory:

\`\`\`bash
python programme_audit.py --services samples/services.csv --exercises samples/exercises.csv --actions samples/actions.csv --exceptions samples/exceptions.csv --as-of 2026-11-20 --window-days 365 --qualification-policy-id DR-QUAL-v1 --out programme-audit-out
python -m unittest discover -s tests -v
\`\`\`

The auditor writes:
- \`programme-summary.json\`
- \`service-findings.csv\`
- \`management-review.md\`

## Open the dashboard
Open \`dashboard.html\` directly in a modern browser. It does not need a local web server.

1. Use **Import JSON** to select one or more \`programme-summary.json\` files exported by separate reporting runs.
2. Import all the reporting dates together to compare them. If the same cut-off date appears twice, the later selected file replaces the earlier one.
3. Use **Exportar evolución CSV** to download the trend table for further analysis.
4. Use **Generar informe para comité (.md)** to download an executive Markdown report for the latest cut-off and its comparison with the preceding one. It includes the metric movements, comparability checks, and current findings.

The initial four-quarter view is synthetic demonstration data. It is not derived from any real company or production service. Importing real report files replaces the demo data. The dashboard processes files in the browser only: it makes no network requests, stores no reports and sends no telemetry.

For fair comparisons, use the same service population, qualification criteria, reporting-window length and calculation rules in each quarterly snapshot. A changed denominator or methodology can make a percentage trend misleading; disclose scope and methodology changes alongside the result.

Newer reports include:
- A SHA-256 checksum of the sorted in-scope service IDs and the corresponding service count. This detects population changes even when the count remains the same.
- Separate SHA-256 checksums and counts for the service cohorts with measured RTO and RPO. The dashboard will not call an RTO/RPO rate movement a like-for-like improvement or deterioration unless the measured service IDs also match.
- Auditor version, dependency-review freshness setting and an optional `--qualification-policy-id` value. Keep this policy ID stable only while the documented qualifying-exercise criteria remain unchanged; increment it when those criteria change.
- The reporting-window length, already shown in the summary.

The dashboard calls portfolio snapshots comparable only when the in-scope checksum, methodology metadata (including a non-empty qualification policy ID) and reporting-window length match. RTO/RPO comparisons additionally require identical measured-service cohort checksums. Older JSON files without this metadata remain importable, but their comparisons are labelled **not verifiable**. The dashboard flags directional movements; these signals are not materiality thresholds or an automated risk opinion.

**Security note:** a SHA-256 checksum is a change-detection mechanism, not anonymisation. IDs with a predictable or small namespace may be guessed and hashed. Treat the JSON summaries, checksums, findings and executive report as operational information; share them only through approved channels.

## Interpret the output carefully
- The coverage denominator is every in-scope service. Valid exceptions are shown separately and do **not** silently remove services from the denominator.
- \`qualifying_for_recovery_coverage\` is a programme-owner classification against an approved definition. The tool cannot decide whether an exercise was sufficiently realistic or independently validate its scope.
- A tabletop, walkthrough or component test should not be counted as an end-to-end technical service recovery unless the programme's documented criteria explicitly justify it.
- RTO/RPO pass rates use only the latest qualifying exercise with the relevant numeric target and observed value. Missing measurements are not passes and are excluded from that metric's denominator.
- An action is counted as verified closed only when status is CLOSED and both verification date and verification evidence are present.
- The dashboard labels valid approved exceptions as **services with an active approved exception**, because the audit summary counts services with at least one such exception, not the raw number of exception records.
- Exception expiry is checked against the supplied \`--as-of\` date. The tool does not approve risk or judge the adequacy of compensating controls.
- The reporting window is configurable. It is not a recommendation that all services must be exercised annually.

## Deliberate limitations
This is a data-quality and portfolio-review helper, not a DR orchestration tool, CMDB, BIA engine, risk engine, GRC platform, audit opinion or certification. It does not test infrastructure, backups, replication, application behaviour, restore integrity, real RTO/RPO or business acceptance. It does not infer criticality or prescribe cadence. Review and formally approve the inputs and conclusions.

All sample records and dashboard demonstration values are synthetic. Do not place production hostnames, IPs, customer data, credentials or confidential recovery details in sample files or exported reports.
