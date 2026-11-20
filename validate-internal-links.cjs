const fs = require("fs");
const path = require("path");
const root = __dirname;
const failures = [];
const stats = { articlePages: 0, contextualArticleLinks: 0, seriesLabLinks: 0, labPathLinks: 0, pathSteps: 0 };

const labByArticle = {
  "/es/analisis/anatomia-interna-transformer-llm/": "/assets/labs/transformer-internals/transformer-internals-01-lab.zip",
  "/en/analysis/inside-transformer-anatomy-llm/": "/assets/labs/transformer-internals/transformer-internals-01-lab.zip",
  "/es/analisis/self-attention-en-profundidad/": "/assets/labs/self-attention/self-attention-02-lab.zip",
  "/en/analysis/self-attention-deep-dive/": "/assets/labs/self-attention/self-attention-02-lab.zip",
  "/es/analisis/como-aprende-un-llm-next-token-cross-entropy-adamw/": "/assets/labs/training/training-optimisation-03-lab.zip",
  "/en/analysis/how-an-llm-learns-next-token-cross-entropy-adamw/": "/assets/labs/training/training-optimisation-03-lab.zip",
  "/es/analisis/inferencia-llm-prefill-decode-kv-cache/": "/assets/labs/inference/inference-engineering-04-lab.zip",
  "/en/analysis/llm-inference-prefill-decode-kv-cache/": "/assets/labs/inference/inference-engineering-04-lab.zip",
  "/es/analisis/vllm-pagedattention-continuous-batching/": "/assets/labs/serving/llm-serving-05-lab.zip",
  "/en/analysis/vllm-pagedattention-continuous-batching/": "/assets/labs/serving/llm-serving-05-lab.zip",
  "/es/analisis/cuantizacion-llm-fp16-bf16-fp8-int8-int4/": "/assets/labs/quantization/llm-quantization-06-lab.zip",
  "/en/analysis/llm-quantization-fp16-bf16-fp8-int8-int4/": "/assets/labs/quantization/llm-quantization-06-lab.zip",
  "/es/analisis/distributed-llm-inference-tensor-parallelism-pipeline-parallelism-nccl/": "/assets/labs/distributed-inference/llm-distributed-inference-07-lab.zip",
  "/en/analysis/distributed-llm-inference-tensor-parallelism-pipeline-parallelism-nccl/": "/assets/labs/distributed-inference/llm-distributed-inference-07-lab.zip",
  "/es/analisis/benchmarking-llm-ttft-tpot-throughput-metodologia/": "/assets/labs/benchmarking/llm-benchmarking-08-lab.zip",
  "/en/analysis/llm-benchmarking-ttft-tpot-throughput-methodology/": "/assets/labs/benchmarking/llm-benchmarking-08-lab.zip",
  "/es/analisis/sql-server-tempdb-crecimiento-log-transaccion-abierta/": "/assets/labs/tempdb-log-growth/tempdb-log-growth-lab.zip",
  "/en/analysis/sql-server-tempdb-log-growth-open-transaction/": "/assets/labs/tempdb-log-growth/tempdb-log-growth-lab.zip",
  "/es/analisis/disenar-prueba-disaster-recovery-medir-recuperacion-servicio/": "/assets/labs/recovery-test-scorecard/recovery-test-scorecard-01-lab.zip",
  "/en/analysis/disaster-recovery-test-design-measuring-service-recovery/": "/assets/labs/recovery-test-scorecard/recovery-test-scorecard-01-lab.zip",
  "/es/analisis/runbook-test-failover-failback-recuperacion-controlada/": "/assets/labs/recovery-runbook-preflight/recovery-runbook-preflight-02-lab.zip",
  "/en/analysis/disaster-recovery-test-failover-failback-runbook/": "/assets/labs/recovery-runbook-preflight/recovery-runbook-preflight-02-lab.zip",
  "/es/analisis/expediente-evidencias-disaster-recovery-aar-ip/": "/assets/labs/recovery-evidence-pack/recovery-evidence-pack-03-lab.zip",
  "/en/analysis/recovery-evidence-pack-after-action-report/": "/assets/labs/recovery-evidence-pack/recovery-evidence-pack-03-lab.zip",
  "/es/analisis/aseguramiento-programa-anual-disaster-recovery/": "/assets/labs/dr-programme-assurance/dr-programme-coverage-auditor-04-lab.zip",
  "/en/analysis/dr-programme-assurance-annual-exercise-coverage/": "/assets/labs/dr-programme-assurance/dr-programme-coverage-auditor-04-lab.zip"
};

function routeStageFor(articlePath) {
  if (articlePath.includes("aseguramiento-programa-anual-disaster-recovery") || articlePath.includes("dr-programme-assurance-annual-exercise-coverage")) return "resilience-04";
  if (articlePath.includes("expediente-evidencias-disaster-recovery") || articlePath.includes("recovery-evidence-pack-after-action-report")) return "resilience-03";
  if (articlePath.includes("runbook-test-failover-failback") || articlePath.includes("disaster-recovery-test-failover-failback")) return "resilience-02";
  if (articlePath.includes("disenar-prueba-disaster-recovery") || articlePath.includes("disaster-recovery-test-design")) return "resilience-01";
  if (articlePath.includes("tempdb")) return "systems-03";
  if (articlePath.includes("distributed-llm-inference")) return "llm-07";
  if (articlePath.includes("anatomia") || articlePath.includes("inside-transformer")) return "llm-01";
  if (articlePath.includes("self-attention")) return "llm-02";
  if (articlePath.includes("como-aprende") || articlePath.includes("how-an-llm-learns")) return "llm-03";
  if (articlePath.includes("inferencia-llm") || articlePath.includes("llm-inference")) return "llm-04";
  if (articlePath.includes("vllm-pagedattention")) return "llm-05";
  if (articlePath.includes("cuantizacion") || articlePath.includes("llm-quantization")) return "llm-06";
  return "llm-08";
}

function getFile(url) {
  const p = decodeURIComponent(new URL(url).pathname);
  return path.join(root, p === "/" ? "index.html" : p.replace(/^\/|\/$/g, "") + "/index.html");
}
function attrs(tag) {
  const out = {};
  for (const m of tag.matchAll(/([\w:-]+)\s*=\s*(["'])(.*?)\2/g)) out[m[1].toLowerCase()] = m[3];
  return out;
}
function anchors(fragment) {
  const out = [];
  for (const m of fragment.matchAll(/<a\b([^>]*)>([\s\S]*?)<\/a>/gi)) {
    const a = attrs("<a " + m[1] + ">");
    if (a.href) out.push({ href: a.href, text: m[2].replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim() });
  }
  return out;
}
function extractCards(html, className) {
  const tags = [...html.matchAll(/<\/?article\b[^>]*>/gi)];
  const result = [];
  for (let i = 0; i < tags.length; i++) {
    const opening = tags[i][0];
    if (opening.startsWith("</")) continue;
    const classMatch = opening.match(/class="([^"]*)"/i);
    if (!classMatch || !classMatch[1].split(/\s+/).includes(className)) continue;
    let depth = 1;
    let j = i + 1;
    for (; j < tags.length && depth > 0; j++) {
      if (tags[j][0].startsWith("</")) depth--;
      else depth++;
    }
    if (depth === 0) {
      const last = tags[j - 1];
      result.push(html.slice(tags[i].index, last.index + last[0].length));
    }
  }
  return result;
}
function articleBody(html) {
  const found = extractCards(html, "article-body");
  if (!found.length) return "";
  const card = found[0];
  const openEnd = card.indexOf(">") + 1;
  return card.slice(openEnd, card.lastIndexOf("</article>"));
}
function expect(ok, message) { if (!ok) failures.push(message); }

const sitemap = fs.readFileSync(path.join(root, "sitemap.xml"), "utf8");
const urls = [...sitemap.matchAll(/<loc>(.*?)<\/loc>/g)].map(m => m[1]);
const articleUrls = urls.filter(u => /^https:\/\/consolaabierta\.com\/(es\/analisis|en\/analysis)\//.test(u));

for (const url of articleUrls) {
  stats.articlePages++;
  const html = fs.readFileSync(getFile(url), "utf8");
  let content = articleBody(html);
  expect(!!content, url + ": missing article-body");
  if (!content) continue;
  const continuity = content.search(/<section\b[^>]*class=["'][^"']*\barticle-continuity\b/i);
  if (continuity >= 0) content = content.slice(0, continuity);
  const ownLang = url.includes("/es/") ? "es" : "en";
  const internal = anchors(content).filter(a => /^\/(es\/analisis|en\/analysis)\//.test(a.href));
  const targets = [...new Set(internal.map(a => a.href))];
  expect(targets.length >= 1, url + ": no contextual article links before the continuity panel");
  for (const target of targets) {
    expect(ownLang === "es" ? target.startsWith("/es/") : target.startsWith("/en/"),
      url + ": cross-language contextual link to " + target);
  }
  stats.contextualArticleLinks += targets.length;
}

for (const locale of ["es", "en"]) {
  const seriesFile = locale === "es" ? "es/series/llm-engineering/index.html" : "en/series/llm-engineering/index.html";
  const series = fs.readFileSync(path.join(root, seriesFile), "utf8");
  const cards = extractCards(series, "archive-card");
  expect(cards.length === 8, seriesFile + ": expected 8 chapter cards, got " + cards.length);
  let count = 0;
  for (const card of cards) {
    const h2 = card.match(/<h2\b[^>]*>\s*<a\b([^>]*)>/i);
    if (!h2) continue;
    const a = attrs("<a " + h2[1] + ">");
    const zip = labByArticle[a.href];
    if (!zip) continue;
    expect(card.includes(zip), seriesFile + ": missing direct lab ZIP for " + a.href);
    if (card.includes(zip)) count++;
  }
  expect(count === 8, seriesFile + ": expected 8 direct lab links, got " + count);
  stats.seriesLabLinks += count;

  const labFile = locale === "es" ? "es/laboratorios/index.html" : "en/labs/index.html";
  const labs = fs.readFileSync(path.join(root, labFile), "utf8");
  const labCards = extractCards(labs, "lab-card");
  expect(labCards.length === 13, labFile + ": expected 13 lab cards, got " + labCards.length);
  let pathCount = 0;
  for (const card of labCards) {
    const links = anchors(card);
    const article = links.find(a => /^\/(es\/analisis|en\/analysis)\//.test(a.href));
    if (!article) continue;
    const stage = routeStageFor(article.href);
    const expected = locale === "es" ? "/es/rutas/#" + stage : "/en/paths/#" + stage;
    expect(links.some(a => a.href === expected), labFile + ": lab for " + article.href + " does not link to " + expected);
    if (links.some(a => a.href === expected)) pathCount++;
  }
  expect(pathCount === 13, labFile + ": expected 13 lab-to-path links, got " + pathCount);
  stats.labPathLinks += pathCount;

  const pathFile = locale === "es" ? "es/rutas/index.html" : "en/paths/index.html";
  const paths = fs.readFileSync(path.join(root, pathFile), "utf8");
  for (let n = 1; n <= 8; n++) {
    const id = "llm-" + String(n).padStart(2, "0");
    expect(new RegExp("id=[\"']" + id + "[\"']").test(paths), pathFile + ": missing " + id);
    stats.pathSteps++;
  }
  expect(/id=["']systems-03["']/.test(paths), pathFile + ": missing systems-03");
  stats.pathSteps++;
  expect(/id=["']resilience-01["']/.test(paths), pathFile + ": missing resilience-01");
  stats.pathSteps++;
  expect(/id=["']resilience-02["']/.test(paths), pathFile + ": missing resilience-02");
  stats.pathSteps++;
  expect(/id=["']resilience-03["']/.test(paths), pathFile + ": missing resilience-03");
  stats.pathSteps++;
  expect(/id=["']resilience-04["']/.test(paths), pathFile + ": missing resilience-04");
  stats.pathSteps++;
}

console.log("Internal link audit:");
console.log(JSON.stringify(stats, null, 2));
console.log("Failures: " + failures.length);
if (failures.length) { console.error(failures.join("\n")); process.exitCode = 1; }
else console.log("ALL INTERNAL LINK CHECKS PASSED");
