# Consola Abierta

Independent technical publication about systems engineering, infrastructure, IT operations, resilience, automation, platform engineering and applied AI.

## Production architecture
- Static HTML/CSS/JavaScript; no framework, build step or application server.
- Spanish edition at / and English edition at /en/.
- Article archives at /es/articulos/ and /en/articles/.
- Four article URLs (two editorial topics translated into Spanish and English).
- Language-aware catalogue search powered by /search-index.json and assets/site.js.
- Six specialty filters on the archives. Empty specialties show an explicit empty state, not placeholder links.
- RSS feeds: /feed.xml (Spanish) and /en/feed.xml (English).
- Sitemap and robots: /sitemap.xml and /robots.txt.
- Public correction channel: /es/contacto/ and /en/contact/ (GitHub Issues; submissions are public).
- Legal/privacy routes remain noindex drafts until the owner confirms the required identity and private contact information.
- GitHub Pages deploys the repository root on pushes to main, using .github/workflows/pages.yml.

## Local preview
Run a static server from the repository root:
powershell
npx http-server . -p 4173
Then open http://127.0.0.1:4173/.

## Quality checks
- node validate-site.cjs checks core routes, metadata, internal links, sitemap and feeds.
- node mobile-check.cjs checks responsive layout (Chromium/Playwright dependencies required locally).
- node --check assets/site.js checks client-side JavaScript syntax.
- git diff --check before committing.
- After pushing, verify the Actions run and the public canonical URLs, feeds and OG image assets.

## Editorial integrity
No article is labelled lab-verified unless its commands and behaviour have been reproduced in a documented environment. Distinguish documentation reviewed, lab evidence, third-party incident analysis and first-hand experience.

## Contact and legal publication gate
The GitHub issue tracker accepts public corrections only. Never publish personal or confidential data there. Legal/privacy routes are marked noindex until the owner verifies legal name, suitable business/professional address, direct contact email and any applicable register or authorisation details. Do not invent these details.

## DNS
DNS was not changed during the November 2026 structural work. Keep DNS, nameservers and mail records unchanged unless the owner explicitly authorises a separately verified change.
