const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "dashboard.html"), "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(match => match[1]);

assert.equal(scripts.length, 1, "dashboard must use one inline script");
assert.doesNotThrow(() => new Function(scripts[0]), "dashboard JavaScript must parse");
assert.doesNotMatch(html, /<script[^>]+src=/i, "dashboard must not load remote scripts");
assert.doesNotMatch(html, /<link[^>]+href=["']https?:/i, "dashboard must not load remote stylesheets");
assert.doesNotMatch(scripts[0], /\bfetch\s*\(|XMLHttpRequest|sendBeacon/i, "dashboard must not send data over the network");

const demoMatch = html.match(/const DEMO_SNAPSHOTS = (\[[\s\S]*?\]);\s*let reports/);
assert.ok(demoMatch, "synthetic demo snapshots must be embedded");
const demo = JSON.parse(demoMatch[1]);
assert.equal(demo.length, 4, "dashboard demo must include four quarterly points");
assert.ok(demo.every(report => report.limitations?.some(item => item.includes("DEMO DATA ONLY"))), "all demo points must be explicitly labelled synthetic");
assert.ok(demo.every(report => report.services.covered_by_qualifying_exercise <= report.services.in_scope), "demo coverage counts must be coherent");
assert.ok(demo.every(report => report.rto.pass_percent === null || report.rto.measured_services > 0), "unmeasured RTO must not show a pass percentage");
assert.ok(demo.every(report => report.rpo.pass_percent === null || report.rpo.measured_services > 0), "unmeasured RPO must not show a pass percentage");

for (const required of ["programme-summary.json", "localmente", "coverage_percent", "valid_approved_exceptions", "overdue_unverified", "renderFindings", "downloadCsv", "service_ids_sha256", "qualification_policy_id", "compareMetadata", "renderComparison", "executiveMarkdown", "exportExecutive"]) {
  assert.ok(html.includes(required), "dashboard is missing expected capability: " + required);
}


assert.ok(demo.every(report => report.scope?.in_scope_services === report.services.in_scope), "demo scope counts must match the service denominator");
assert.ok(demo.every(report => /^[a-f0-9]{64}$/.test(report.scope?.service_ids_sha256 || "")), "demo scope checksums must be SHA-256 shaped");
assert.ok(demo.every(report => report.scope?.rto_measured_services === report.rto.measured_services && /^[a-f0-9]{64}$/.test(report.scope?.rto_service_ids_sha256 || "")), "demo RTO cohort metadata must match its measured count and checksum");
assert.ok(demo.every(report => report.scope?.rpo_measured_services === report.rpo.measured_services && /^[a-f0-9]{64}$/.test(report.scope?.rpo_service_ids_sha256 || "")), "demo RPO cohort metadata must match its measured count and checksum");
assert.ok(demo.every(report => report.methodology?.qualification_policy_id === "DEMO-QUAL-v1"), "demo methodology must carry a versioned qualification policy");
assert.notEqual(demo[0].scope.service_ids_sha256, demo[2].scope.service_ids_sha256, "demo must illustrate a changed service population");

const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    id, textContent: "", innerHTML: "", value: "", disabled: false, listeners: {},
    classList: { toggle() {}, add() {}, remove() {} },
    addEventListener(type, callback) { this.listeners[type] = callback; },
    click() { this.clicked = true; }
  });
  return elements.get(id);
}
const documentStub = {
  getElementById: element,
  createElement: tag => element("created-" + tag)
};
let downloadedBlob = null;
const urlStub = { createObjectURL(blob) { downloadedBlob = blob; return "blob:dashboard-test"; }, revokeObjectURL() {} };
class BlobStub {
  constructor(parts, options) { this.parts = parts; this.options = options; }
}
new Function("document", "Blob", "URL", "setTimeout", scripts[0])(documentStub, BlobStub, urlStub, callback => callback());
assert.ok(element("scopeStatus").innerHTML.includes("COINCIDE"), "latest demo periods should have the same service scope");
assert.ok(element("methodStatus").innerHTML.includes("COINCIDE"), "latest demo periods should have the same methodology");
assert.ok(element("windowStatus").innerHTML.includes("COINCIDE"), "latest demo periods should have matching reporting windows");
assert.ok(element("changeRows").innerHTML.includes("Mejora"), "dashboard should classify a favourable synthetic movement");
assert.ok(element("changeRows").innerHTML.includes("Cohorte medida distinta/no verificable"), "dashboard must not classify RTO/RPO as like-for-like when measured service IDs change");
assert.ok(element("scopeDetail").textContent.includes("Cohorte RTO: changed"), "dashboard should expose changes to the RTO measured cohort");
assert.equal(element("exportExecutive").disabled, false, "executive report export should be enabled when reports are loaded");
element("exportExecutive").listeners.click();
assert.ok(downloadedBlob, "executive export should create a downloadable Markdown blob");
assert.ok(String(downloadedBlob.parts[0]).includes("DR Programme Assurance — Informe ejecutivo"), "executive report should include its title");
assert.ok(String(downloadedBlob.parts[0]).includes("Comparabilidad con el cierre anterior"), "executive report should disclose comparability");
assert.ok(String(downloadedBlob.parts[0]).includes("Cohorte RTO: Cambio detectado"), "executive report should disclose measured-cohort changes");
assert.ok(String(downloadedBlob.parts[0]).includes("DATOS SINTÉTICOS"), "demo executive report must be labelled synthetic");

console.log("DR Programme Assurance dashboard checks passed (offline mode, syntax, four synthetic periods).");
