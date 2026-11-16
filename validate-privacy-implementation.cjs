const fs = require('fs');
const path = require('path');
const vm = require('vm');
const failures = [];
const sitemap = fs.readFileSync('sitemap.xml', 'utf8');
const urls = [...sitemap.matchAll(/<loc>https:\/\/consolaabierta\.com([^<]*)<\/loc>/g)].map(m => m[1]);
const siteJs = fs.readFileSync('assets/site.js', 'utf8');
const adsTxt = fs.readFileSync('ads.txt', 'utf8');
const siteCss = fs.readFileSync('assets/site.css', 'utf8');
function simulateConsent(gdprApplies) {
  let revoked = 0;
  let tcfCallback;
  const handlers = {};
  const link = { hidden: true, addEventListener: (name, callback) => { handlers[name] = callback; } };
  const googlefc = { callbackQueue: [], showRevocationMessage: () => { revoked++; } };
  const document = { documentElement: { lang: 'es' }, querySelectorAll: () => [], getElementById: id => id === 'privacy-settings-link' ? link : null };
  const window = { googlefc, __tcfapi: (command, version, callback) => { if (command === 'addEventListener' && version === 2) tcfCallback = callback; } };
  vm.runInNewContext(siteJs, { document, window, location: { search: '' }, URLSearchParams, fetch: () => { throw new Error('Unexpected network request during consent test'); } });
  const ready = googlefc.callbackQueue.find(item => typeof item.CONSENT_API_READY === 'function');
  if (!ready) throw new Error('CONSENT_API_READY callback not registered');
  ready.CONSENT_API_READY();
  if (!tcfCallback) throw new Error('TCF v2 event listener not registered');
  tcfCallback({ gdprApplies }, true);
  const visible = !link.hidden;
  if (visible) handlers.click({ preventDefault: () => {} });
  return { visible, revoked };
}
const consentForEea = simulateConsent(true);
const consentOutsideEea = simulateConsent(false);
const publisherId = 'ca-pub-5250134414402941';
let adsPages = 0;
function check(ok, message) { if (!ok) failures.push(message); }

for (const url of urls) {
  const file = url === '/' ? 'index.html' : path.join(url.replace(/^\//, '').replace(/\/$/, ''), 'index.html');
  if (!fs.existsSync(file)) { failures.push(`${url}: local page missing`); continue; }
  const html = fs.readFileSync(file, 'utf8');
  const footer = (html.match(/<div class="footer-links">([\s\S]*?)<\/div>/) || [])[1] || '';
  const isEn = url.startsWith('/en/');
  const legalHref = isEn ? '/en/legal-notice/' : '/es/aviso-legal/';
  const privacyHref = isEn ? '/en/privacy/' : '/es/privacidad/';
  const consentText = isEn ? 'Cookie &amp; privacy settings' : 'Configurar cookies y privacidad';
  check(!!footer, `${url}: footer missing`);
  check(footer.split(`href="${legalHref}"`).length - 1 === 1, `${url}: legal link missing or duplicated`);
  check(footer.split(`href="${privacyHref}"`).length - 1 === 1, `${url}: privacy link missing or duplicated`);
  check((footer.match(/id="privacy-settings-link"/g) || []).length === 1, `${url}: consent settings link missing or duplicated`);
  check(footer.includes(consentText), `${url}: consent link label not localized`);
  check((html.match(/src="\/assets\/site\.js"/g) || []).length === 1, `${url}: site.js must load once`);
  check(!/<script\b[^>]*>[\s\S]*?window\.googlefc[\s\S]*?<\/script>/i.test(html), `${url}: legacy inline CMP code remains`);
  const ads = [...html.matchAll(/adsbygoogle\.js\?client=([^"&]+)/g)];
  if (ads.length) {
    adsPages++;
    check(ads.length === 1, `${url}: multiple AdSense loaders`);
    check(ads[0][1] === publisherId, `${url}: AdSense publisher ID mismatch`);
  }
}

let legalDrafts = 0;
for (const [url, file] of [['/es/aviso-legal/', 'es/aviso-legal/index.html'], ['/en/legal-notice/', 'en/legal-notice/index.html']]) {
  const html = fs.readFileSync(file, 'utf8');
  legalDrafts++;
  check(/<meta name="robots" content="noindex, nofollow">/i.test(html), `${url}: draft must remain noindex`);
  check(/pendiente|pending/i.test(html), `${url}: draft status marker missing`);
}
check(siteJs.includes('CONSOLA_ABIERTA_CONSENT_MANAGER'), 'central consent manager missing');
check(siteJs.includes('__tcfapi("addEventListener",2'), 'TCF v2 callback missing');
check(siteJs.includes('showRevocationMessage'), 'consent revocation method missing');
check(siteCss.includes('#privacy-settings-link[hidden]{display:none!important}'), 'hidden consent link must be forcibly hidden in CSS');
check(consentForEea.visible && consentForEea.revoked === 1, 'consent settings link must appear for EEA and reopen consent preferences');
check(!consentOutsideEea.visible && consentOutsideEea.revoked === 0, 'consent settings link must stay hidden where GDPR does not apply');
check(adsTxt.includes(publisherId.replace('ca-', '')), 'ads.txt publisher ID mismatch');
check(legalDrafts === 2, 'expected two legal notice drafts');
const report = { sitemapPages: urls.length, adsEnabledPages: adsPages, legalNoticeDraftsKeptNoindex: legalDrafts, consentLinks: urls.length, centralizedConsentManager: siteJs.includes('CONSOLA_ABIERTA_CONSENT_MANAGER'), consentRuntimeTests: { eea: consentForEea, outsideEea: consentOutsideEea }, failures: failures.length, results: failures.length ? failures : ['ALL PRIVACY IMPLEMENTATION CHECKS PASSED'] };
console.log(JSON.stringify(report, null, 2));
if (failures.length) process.exit(1);
