# GitHub Pages — production runbook

## Current production
- Repository: raycbn/consola-abierta
- Branch: main
- Hosting: GitHub Pages via GitHub Actions.
- Workflow: .github/workflows/pages.yml
- Deployment source: repository root (static HTML/CSS/JavaScript).
- Canonical host: https://consolaabierta.com/
- Custom domain source file: CNAME containing consolaabierta.com.
- No local build step or package installation is required for production.

## Release procedure
1. Fetch origin and verify the local main branch is aligned with origin/main before editing.
2. Run node validate-site.cjs.
3. Run node --check assets/site.js and the mobile checker where Chromium/Playwright are installed.
4. Check ES/EN translation pairs, canonical/hreflang, sitemap, both RSS feeds and their XSLT browser view, Open Graph images and git diff --check.
5. Stage only publication source files. Keep lab evidence and local browser previews outside this repository.
6. Commit and push to origin main only after all checks pass.
7. Verify the GitHub Actions run for the new commit reaches success.
8. Check live HTTP 200 for both homepages, both archives, contact pages, all articles, sitemap, robots, RSS ES/EN, CSS, JS and Open Graph PNG images.
9. Confirm http:// and www redirect to https://consolaabierta.com/.
10. Compare local and live pages after normalising newline conventions.

## Rollback
- Identify the last known-good commit and successful Pages workflow.
- Revert a faulty deployment with a new commit; do not force-push main.
- Wait for the Pages workflow to complete and verify the restored canonical URLs.
- Preserve the failing commit and its QA output for diagnosis.

## DNS and mail — protected
No DNS change was made during the 13 November 2026 structural close-out. Do not modify nameservers, apex A records, www CNAME, MX, SPF, DKIM, DMARC or other verification records as part of a routine content release. A future DNS change requires explicit approval, a complete zone export, verification of current records and a documented rollback plan. Mail delivery must be tested separately from website availability.

## Legal and contact publication gate
The legal notice and privacy routes are deliberately noindex drafts. Do not remove the robots restriction or add them to the sitemap until the site owner confirms the legal name, suitable business/professional address, NIF and any applicable public-register or authorisation details. The private contact email contacto@consolaabierta.com is published; GitHub Issues remain restricted to public corrections, not privacy-rights requests or confidential correspondence.
