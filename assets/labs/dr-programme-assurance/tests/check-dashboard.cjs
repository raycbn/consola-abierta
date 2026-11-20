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

for (const required of ["programme-summary.json", "localmente", "coverage_percent", "valid_approved_exceptions", "overdue_unverified", "renderFindings", "downloadCsv", "service_ids_sha256", "qualification_policy_id", "compareMetadata", "renderComparison", "executiveMarkdown", "exportExecutive", "exportHistory", "dr-programme-history-v1", "unpackImportedPayload", "sustainedDeteriorationSignals", "sustainedTrend", "committeeWorkspace", "addDecision", "addCommitteeAction", "validateCommittee", "restaura también el borrador del comité", "committeeMarkdown", "exportCommitteeMinutes", "exportCommitteeCsv", "evidence", "DECISION_STATUSES", "ACTION_STATUSES", "BORRADOR — PENDIENTE DE VALIDACIÓN"]) {
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


async function importTestArchive(name, archive) {
  const content = JSON.stringify(archive);
  element("fileInput").listeners.change({
    target: { files: [{ name, size: content.length, text: async () => content }], value: "" }
  });
  await new Promise(resolve => setImmediate(resolve));
}

(async () => {
  element("exportHistory").listeners.click();
  const historyText = String(downloadedBlob.parts[0]);
  const exportedHistory = JSON.parse(historyText);
  assert.equal(exportedHistory.format, "dr-programme-history-v1", "history export must declare its format");
  assert.equal(exportedHistory.reports.length, 4, "history export must retain all four snapshots");
  assert.equal(exportedHistory.contains_synthetic_data, true, "history export must preserve the synthetic-data marker");
  await importTestArchive("dr-programme-history.json", exportedHistory);
  assert.ok(element("reportCount").textContent.includes("4 cierre(s) cargado(s)"), "history archive should restore every snapshot");
  assert.ok(element("reportCount").textContent.includes("DEMO SINTÉTICA"), "history import must preserve synthetic labelling");

  const sustained = demo.slice(0, 3).map(report => JSON.parse(JSON.stringify(report)));
  sustained.forEach((report, index) => {
    report.services.in_scope = 24;
    report.services.uncovered = 24 - report.services.covered_by_qualifying_exercise;
    report.services.coverage_percent = Math.round(report.services.covered_by_qualifying_exercise / 24 * 1000) / 10;
    report.scope.in_scope_services = 24;
    report.scope.service_ids_sha256 = "a".repeat(64);
    report.rto.measured_services = 8;
    report.rto.met = 5;
    report.rto.pass_percent = 62.5;
    report.scope.rto_measured_services = 8;
    report.scope.rto_service_ids_sha256 = "c".repeat(64);
    report.rpo.measured_services = 7;
    report.rpo.met = 4;
    report.rpo.pass_percent = 57.1;
    report.scope.rpo_measured_services = 7;
    report.scope.rpo_service_ids_sha256 = "d".repeat(64);
    report.actions.overdue_unverified = index + 1;
  });
  await importTestArchive("sustained-trend-test.json", {
    format: "dr-programme-history-v1",
    reports: sustained
  });
  assert.ok(element("sustainedTrend").innerHTML.includes("Deterioros sostenidos detectados"), "dashboard should detect deterioration over two comparable intervals");
  assert.ok(element("sustainedTrend").innerHTML.includes("Acciones vencidas sin verificar"), "sustained deterioration signal should identify the metric");

  element("meetingTitle").value = "Comité de continuidad";
  element("meetingDate").value = "2026-12-10";
  element("meetingReference").value = "DR-COM-2026-Q4";
  element("meetingChair").value = "Presidencia de prueba";
  element("meetingSecretary").value = "Secretaría de prueba";
  for (const [id, field] of [["meetingTitle","title"],["meetingDate","date"],["meetingReference","reference"],["meetingChair","chair"],["meetingSecretary","secretary"]]) {
    element("committeeWorkspace").listeners.input({target:{dataset:{meetingField:field},value:element(id).value}});
  }

  element("decisionTopic").value = "Cobertura Tier 1";
  element("decisionText").value = "Revisar y aprobar el calendario de ejercicios";
  element("decisionOwner").value = "Responsable DR";
  element("decisionDue").value = "2027-01-31";
  element("decisionStatus").value = "PENDING";
  element("addDecision").listeners.click();
  assert.ok(element("decisionRows").innerHTML.includes("DEC-001"), "decision form should add a traceable decision record");

  element("actionDecisionLink").value = "DEC-001";
  element("committeeActionText").value = "Publicar calendario de ejercicios";
  element("committeeActionOwner").value = "Equipo DR";
  element("committeeActionDue").value = "2027-01-15";
  element("committeeActionPriority").value = "HIGH";
  element("committeeActionStatus").value = "OPEN";
  element("addCommitteeAction").listeners.click();
  assert.ok(element("committeeActionRows").innerHTML.includes("ACT-001"), "action form should add a traceable action record");

  const invalidDueField = {dataset:{actionId:"ACT-001",actionField:"due_date"},value:""};
  element("committeeWorkspace").listeners.change({target:invalidDueField});
  assert.equal(invalidDueField.value,"2027-01-15","an empty action deadline must be rejected and the previous date restored");

  element("committeeWorkspace").listeners.change({target:{dataset:{actionId:"ACT-001",actionField:"status"},value:"CLOSED"}});
  assert.ok(element("committeeAlert").textContent.includes("referencia de evidencia"), "an action cannot close without an evidence reference");
  element("committeeWorkspace").listeners.input({target:{dataset:{actionId:"ACT-001",actionField:"evidence"},value:"EV-COM-2027-001"}});
  element("committeeWorkspace").listeners.change({target:{dataset:{actionId:"ACT-001",actionField:"status"},value:"CLOSED"}});
  assert.ok(element("committeeSummary").innerHTML.includes("Acciones cerradas con evidencia: <strong>1</strong>"), "closed action must be counted only with evidence");

  element("exportCommitteeMinutes").listeners.click();
  const minutes = String(downloadedBlob.parts[0]);
  assert.ok(minutes.includes("BORRADOR — PENDIENTE DE VALIDACIÓN"), "committee minutes must clearly be an unapproved draft");
  assert.ok(minutes.includes("DATOS SINTÉTICOS"), "committee minutes linked to demo data must carry a synthetic-data warning");
  assert.ok(minutes.includes("DR-COM-2026-Q4") && minutes.includes("DEC-001") && minutes.includes("ACT-001") && minutes.includes("EV-COM-2027-001"), "committee minutes must include session, decision, action and closure evidence");

  element("exportCommitteeCsv").listeners.click();
  const actionCsv = String(downloadedBlob.parts[0]);
  assert.ok(actionCsv.includes("action_id") && actionCsv.includes("decision_id") && actionCsv.includes("EV-COM-2027-001"), "committee action CSV must include ownership link and closure evidence");

  element("exportHistory").listeners.click();
  const committeeArchive = JSON.parse(String(downloadedBlob.parts[0]));
  assert.equal(committeeArchive.committee.decisions.length, 1, "history archive must preserve committee decisions");
  assert.equal(committeeArchive.committee.actions.length, 1, "history archive must preserve committee actions");
  assert.equal(committeeArchive.committee.actions[0].status, "CLOSED", "history archive must preserve action status");
  await importTestArchive("committee-history.json", committeeArchive);
  assert.ok(element("decisionRows").innerHTML.includes("DEC-001"), "history import must restore committee decisions");
  assert.ok(element("committeeActionRows").innerHTML.includes("EV-COM-2027-001"), "history import must restore action evidence");

  const invalidArchive = JSON.parse(JSON.stringify(committeeArchive));
  invalidArchive.committee.actions[0].evidence = "";
  await importTestArchive("invalid-committee-history.json", invalidArchive);
  assert.ok(element("status").textContent.includes("referencia de evidencia"), "invalid committee archives must be rejected when a closed action has no evidence");

  console.log("DR Programme Assurance dashboard checks passed (offline, comparisons, history, trends, committee decisions/actions, evidence-gated closure and minutes).");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});

