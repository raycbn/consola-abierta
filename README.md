# Consola Abierta

Independent technical publication about systems engineering, infrastructure, IT operations, resilience, automation, platform engineering and applied AI.

## Current implementation
- Static HTML/CSS/JavaScript. No client-side framework or build step.
- Spanish home page at /.
- English edition at /en/.
- Bilingual PowerShell lifecycle analysis.
- Responsive editorial layout.
- Canonical URLs, hreflang, Open Graph and WebSite/Article JSON-LD.
- Sitemap, robots.txt and 404 page.

## Local preview
Run a static server from this directory. For example:
```powershell
npx http-server . -p 4173
```
Then open http://localhost:4173.

## Deployment notes
1. Confirm the hosting provider and deployment workflow.
2. Preserve the existing Arsys email MX and SPF records.
3. Configure HTTPS and redirect HTTP to HTTPS.
4. Enforce one canonical hostname.
5. Validate routes, metadata, sitemap, redirects and mobile layouts.
6. Add Search Console after the public canonical URLs are stable.

## Editorial integrity
No article is marked lab-verified unless commands and behaviour have been reproduced in a documented environment. Separate official documentation, third-party incident analysis, lab results and first-hand experience.