# Consola Abierta

Independent technical publication about systems engineering, infrastructure, IT operations, resilience, automation, platform engineering and applied AI.

## Production architecture
- Static HTML/CSS/JavaScript; no framework, build step or application server.
- Spanish edition at / and English edition at /en/.
- Article archives at /es/articulos/ and /en/articles/.
- 30 article URLs: 15 Spanish articles and their 15 English editions.
- Language-aware catalogue search powered by /search-index.json and assets/site.js.
- Seven specialty filters on the archives, including Resilience & Recovery and the LLM Engineering series. Empty specialties show an explicit empty state, not placeholder links.
- RSS feeds: /feed.xml (Spanish) and /en/feed.xml (English), with an XSLT browser view and Atom self links.
- Sitemap and robots: /sitemap.xml and /robots.txt.
- Public corrections: /es/contacto/ and /en/contact/ via GitHub Issues (public); private contact: contacto@consolaabierta.com (published).
- Privacy routes now disclose Google AdSense, advertising technologies, cookies, consent management, providers and user rights in Spanish and English. They remain noindex until the owner confirms legal identity and address.
- ads.txt authorises Google as a direct seller using the verified publisher ID. Keep the file at the repository root and update it only from account-provided seller records.
- The public sitemap pages include a cookie/privacy preferences link wired to Google's `googlefc.showRevocationMessage` API. The link is shown only when the TCF reports that GDPR applies. The European message itself must still be published from AdSense > Privacy & messaging.
- Legal notice remains noindex until the owner confirms legal name, suitable address, NIF and any applicable registration/authorisation details.
- 13 downloadable labs, including four offline resilience tools: Recovery Test Scorecard (7 tests), Recovery Runbook Preflight (12 tests), Recovery Evidence Pack Builder (21 tests) and DR Programme Coverage Auditor (16 tests) with a browser-only quarterly dashboard. The tools analyse manually supplied data and file references; they do not run recovery operations, verify infrastructure controls, authenticate evidence sources or provide forensic chain of custody.
- GitHub Pages runs all four resilience lab test suites and builds their downloadable ZIPs from versioned source before running site validators and deploying on pushes to main, using .github/workflows/pages.yml.

## Local preview
Run a static server from the repository root:
```powershell
npx http-server . -p 4173
```
Then open http://127.0.0.1:4173/.

## Quality checks
- `python -m unittest discover -s assets/labs/recovery-test-scorecard/tests -v` checks the offline recovery scorecard (7 tests).
- `python assets/labs/recovery-test-scorecard/build_lab.py` regenerates its downloadable ZIP.
- `python -m unittest discover -s assets/labs/dr-programme-assurance/tests -v` checks the DR programme coverage auditor (16 tests).
- `python assets/labs/dr-programme-assurance/build_lab.py` builds the downloadable Lab 04 ZIP, including the offline dashboard.
- `node assets/labs/dr-programme-assurance/tests/check-dashboard.cjs` checks dashboard syntax, local-only processing and synthetic demo data.
- `node validate-text-encoding.cjs` checks text sources for common mojibake and known accent-replacement regressions.
- `node validate-structured-data.cjs` checks sitemap routes, canonicals, JSON-LD and visible breadcrumbs.
- `node validate-internal-links.cjs` checks contextual article links and article/lab/learning-path relationships.
- `node validate-privacy-implementation.cjs` checks AdSense identifiers, legal-page publication gates and consent-preference wiring.
- `node --check assets/site.js` checks client-side JavaScript syntax.
- Responsive/mobile QA is currently a manual check; there is no `mobile-check.cjs` in the repository.
- `git diff --check` before committing.
- After pushing, verify the Actions run and the public canonical URLs, feeds and OG image assets.

## Editorial integrity
No article is labelled lab-verified unless its commands and behaviour have been reproduced in a documented environment. Distinguish documentation reviewed, lab evidence, third-party incident analysis and first-hand experience.

## Contact and legal publication gate
The GitHub issue tracker accepts public corrections only; private correspondence and data-protection requests go to contacto@consolaabierta.com. Never publish personal or confidential data in public issues. Legal/privacy routes remain noindex until the owner verifies legal name, suitable business/professional address, NIF and any applicable register or authorisation details. The email address is confirmed and published; do not invent the remaining details.

## DNS
DNS was not changed during the November 2026 structural work. Keep DNS, nameservers and mail records unchanged unless the owner explicitly authorises a separately verified change.
