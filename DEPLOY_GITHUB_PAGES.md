# Deployment — GitHub Pages + consolaabierta.com

Static HTML/CSS/JavaScript. No build system is required.

## Repository

Create a new **public** GitHub repository named `consola-abierta` under `raycbn`. Keep it separate from the existing `rayvega` portfolio and `Projects` repository.

Push the complete contents of this directory to the repository's `main` branch. The workflow at `.github/workflows/pages.yml` deploys the repository root to GitHub Pages on every push to `main`.

## GitHub Pages setup

1. In the new repository, open **Settings → Pages**.
2. Set Build and deployment to **GitHub Actions**.
3. Wait for the first successful deployment workflow.
4. Verify ownership of `consolaabierta.com` in GitHub Pages / account domain settings before attaching it.
5. In the repository's Pages settings, set the custom domain to `consolaabierta.com` and save.
6. Only after the custom domain is attached, update the web DNS records in Arsys.
7. Enable **Enforce HTTPS** after GitHub has issued the certificate.

For a GitHub Actions deployment, the custom domain is configured in Pages settings; a `CNAME` source file is not required.

## Current DNS baseline (checked from PCRAY, 11 Nov 2026)

- Nameservers: `dns97.servidoresdns.net` and `dns98.servidoresdns.net`
- A `@`: `217.76.128.47`
- A `www`: `217.76.128.47`
- MX `@`: `mx.serviciodecorreo.es`, priority `10`
- TXT SPF `@`: `v=spf1 include:_spf.serviciodecorreo.es ~all`

The current nameservers should remain in place. Do not move the domain to another DNS provider just to use GitHub Pages.

## Arsys DNS changes (only after GitHub Pages domain setup)

At the Arsys DNS zone editor:

Replace the apex `A @` record currently pointing to `217.76.128.47` with all four GitHub Pages IPv4 records:

- A `@` → `185.199.108.153`
- A `@` → `185.199.109.153`
- A `@` → `185.199.110.153`
- A `@` → `185.199.111.153`

For `www`, remove the current A record to `217.76.128.47` and create:

- CNAME `www` → `raycbn.github.io`

**Do not modify**:
- NS records (`dns97.servidoresdns.net`, `dns98.servidoresdns.net`)
- MX record (`mx.serviciodecorreo.es`, priority 10)
- SPF TXT record (`v=spf1 include:_spf.serviciodecorreo.es ~all`)
- Any DKIM, DMARC, verification or other mail-related TXT records

Export or screenshot the complete DNS zone before making the changes. Do not delete unrelated records.

## Verification after DNS propagation

Use PowerShell `Resolve-DnsName` to check the apex A records and the www CNAME. GitHub notes that DNS propagation can take up to 24 hours.

Validate:
- https://consolaabierta.com/
- https://consolaabierta.com/en/
- https://consolaabierta.com/es/analisis/powershell-7-6-migracion-empresarial/
- https://consolaabierta.com/en/analysis/powershell-7-6-enterprise-migration/
- https://consolaabierta.com/es/analisis/diagnostico-rendimiento-windows-server/
- https://consolaabierta.com/en/analysis/windows-server-performance-diagnostics/
- https://consolaabierta.com/es/sobre/
- https://consolaabierta.com/en/about/
- https://consolaabierta.com/sitemap.xml
- https://consolaabierta.com/robots.txt
- https://consolaabierta.com/feed.xml

Check certificate issuance, redirects, canonical URLs, hreflang, metadata, mobile layout and mail delivery. Keep NS/MX/SPF intact throughout.