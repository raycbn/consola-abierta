# Consola Abierta — Editorial Standard v1.0

## Editorial promise

Consola Abierta publishes substantial, useful technical material for mid-level, senior and specialist practitioners. We explain how systems are designed, operated, changed, broken, recovered and improved.

We do not publish basic filler, vendor announcements rewritten as articles, untested command dumps or generic AI summaries.

## Audience

- Mid-level and senior systems / infrastructure engineers
- IT operations, SRE and platform engineers
- Infrastructure, cloud and virtualization specialists
- Resilience, business continuity and recovery practitioners
- Automation engineers and technical architects

## Content formats

1. Technical deep dives
2. Troubleshooting and resolution reports
3. Operational procedures and runbooks
4. Architecture and engineering decisions
5. Labs, experiments and benchmarks
6. Technology radar, release notes and security advisories
7. Postmortems and lessons learned
8. Comparative evaluations and migration plans

## Evidence labels

- **Documentation reviewed:** claims checked against authoritative product documentation.
- **Lab verified:** commands and behaviour reproduced in a named, versioned test environment. State the exact environment.
- **Incident analysis:** analysis based on a published third-party or first-party postmortem. Attribute the original source and distinguish our analysis from the vendor's statements.
- **Field experience:** first-hand experience that can be shared legitimately, with customer and employer confidentiality protected.
- **Opinion:** reasoned editorial assessment, clearly separated from facts.

Never label a procedure lab verified unless it has been reproduced. Never present a hypothetical scenario as a real production incident.

## Required structure for procedures

- Purpose, scope and intended outcome
- Supported products, versions and platforms
- Preconditions and required privileges
- Dependencies and assumptions
- Risk, blast radius and service impact
- Pre-checks and baseline evidence
- Numbered execution steps
- Expected output / evidence after significant steps
- Troubleshooting and decision points
- Rollback or recovery procedure
- Post-checks and acceptance criteria
- References and review date

## Required structure for incident analysis

- What happened and when
- Observable symptoms and customer / service impact
- System topology and relevant dependencies
- Timeline
- Triggering event, root cause and contributing factors (do not conflate them)
- Detection gaps and why safeguards failed
- Recovery and data integrity considerations
- Corrective and preventive actions
- What is generalisable, and what is specific to the incident
- First-party source and publication date

## Source hierarchy

1. Primary vendor documentation, release notes, security advisories and support lifecycle pages
2. Standards bodies, project maintainers and upstream repositories
3. First-party postmortems and engineering blogs
4. Independent technical research and reproducible practitioner analysis
5. Community discussion only as supporting context, never as sole proof for critical claims

Record source URLs, publication / update dates and the date checked. For fast-moving topics, re-check sources immediately before publication.

## Technical quality gates

- Re-check product version, support status and dates.
- Verify commands syntactically and review side effects.
- Identify destructive actions explicitly.
- Never expose credentials, customer data, internal hostnames or confidential architecture.
- Explain privilege level and execution context.
- Include a rollback path where applicable.
- Distinguish assumptions, observed facts and hypotheses.
- Test code samples when marked lab verified.
- Review every external link and ensure it supports the adjacent statement.

## Editorial voice

Precise, direct, practitioner-to-practitioner. No hype, no fake certainty, no SEO padding, no "10 amazing tricks" framing. Explain trade-offs and failure modes. Use diagrams only when they clarify systems or decisions.

## Maintenance

Every version-sensitive article carries a reviewed date. Revisit content when vendor lifecycle, release, security, compatibility or behaviour changes materially. Keep a short revision history for substantial corrections.