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

for (const required of ["programme-summary.json", "localmente", "coverage_percent", "valid_approved_exceptions", "overdue_unverified", "renderFindings", "downloadCsv"]) {
  assert.ok(html.includes(required), "dashboard is missing expected capability: " + required);
}

console.log("DR Programme Assurance dashboard checks passed (offline mode, syntax, four synthetic periods).");
