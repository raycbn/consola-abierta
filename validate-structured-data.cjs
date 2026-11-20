const fs = require("fs");
const path = require("path");

const ROOT = __dirname;
const ORIGIN = "https://consolaabierta.com";
const failures = [];
const counts = {};
function fail(message) { failures.push(message); }
function attrs(tag) {
  const out = {};
  for (const m of tag.matchAll(/([a-zA-Z_:][-a-zA-Z0-9_:]*)\s*=\s*(["'])(.*?)\2/g)) out[m[1].toLowerCase()] = m[3];
  return out;
}
function decode(s) {
  return String(s || "").replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/&#39;|&apos;/g, "'").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&nbsp;/g, " ")
    .replace(/&#(\d+);/g, function(_, n) { return String.fromCodePoint(Number(n)); })
    .replace(/&#x([0-9a-f]+);/gi, function(_, n) { return String.fromCodePoint(parseInt(n, 16)); });
}
function textContent(s) { return decode(String(s || "").replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim()); }
function getCanonical(html) {
  for (const tag of html.match(/<link\b[^>]*>/gi) || []) {
    const a = attrs(tag);
    if ((a.rel || "").toLowerCase() === "canonical") return a.href || "";
  }
  return "";
}
function jsonLd(html, file) {
  const blocks = [...html.matchAll(/<script\b[^>]*type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi)];
  const out = [];
  for (const block of blocks) {
    try {
      const v = JSON.parse(block[1]);
      if (Array.isArray(v)) out.push(...v);
      else if (v && Array.isArray(v["@graph"])) out.push(...v["@graph"]);
      else out.push(v);
    } catch (e) { fail(file + ": JSON-LD parse error: " + e.message); }
  }
  return out;
}
function getBreadcrumbHtml(html) {
  const m = html.match(/<(div|nav)\b[^>]*class=["'][^"']*\bbreadcrumbs\b[^"']*["'][^>]*>([\s\S]*?)<\/\1>/i);
  if (!m) return null;
  const inner = m[2];
  const links = [...inner.matchAll(/<a\b([^>]*)>([\s\S]*?)<\/a>/gi)].map(function(a) {
    const at = attrs("<a " + a[1] + ">");
    return { name: textContent(a[2]), item: at.href || "" };
  });
  const end = links.length ? inner.lastIndexOf("</a>") + 4 : 0;
  const last = textContent(inner.slice(end)).replace(/^[\s/|â€ºÂ»>Â·-]+|[\s/|â€ºÂ»>Â·-]+$/g, "");
  if (last) links.push({ name: last, item: "" });
  return links;
}
function pathToFile(url) {
  const pathname = decodeURIComponent(new URL(url).pathname);
  return path.join(ROOT, pathname === "/" ? "index.html" : pathname.replace(/^\/|\/$/g, "") + path.sep + "index.html");
}
function findType(objects, type) { return objects.find(function(x) { return x && x["@type"] === type; }); }
function meta(html, selector, expected) {
  for (const tag of html.match(/<meta\b[^>]*>/gi) || []) {
    const a = attrs(tag);
    if ((a[selector] || "").toLowerCase() === expected.toLowerCase()) return a.content || "";
  }
  return "";
}
function expect(condition, message) { if (!condition) fail(message); }

const sitemap = fs.readFileSync(path.join(ROOT, "sitemap.xml"), "utf8");
const urls = [...sitemap.matchAll(/<loc>(.*?)<\/loc>/g)].map(function(m) { return m[1]; });
expect(urls.length === 38, "Expected 38 sitemap URLs; got " + urls.length);
for (const url of urls) {
  const file = pathToFile(url);
  const rel = path.relative(ROOT, file);
  if (!fs.existsSync(file)) { fail("Missing sitemap page: " + rel); continue; }
  const html = fs.readFileSync(file, "utf8");
  const canonical = getCanonical(html);
  expect(canonical === url, rel + ": canonical mismatch");
  const objects = jsonLd(html, rel);
  const types = objects.map(function(x) { return x && x["@type"]; }).filter(Boolean);
  const pagePath = new URL(url).pathname;
  const isHome = pagePath === "/" || pagePath === "/en/";
  const crumbs = findType(objects, "BreadcrumbList");
  const visibleCrumbs = getBreadcrumbHtml(html);
  if (!isHome) {
    expect(!!crumbs, rel + ": missing BreadcrumbList");
    expect(!!visibleCrumbs, rel + ": BreadcrumbList has no visible breadcrumb");
    if (crumbs) {
      const list = crumbs.itemListElement || [];
      expect(list.length >= 2, rel + ": BreadcrumbList must contain at least 2 items");
      if (visibleCrumbs) {
        expect(list.length === visibleCrumbs.length, rel + ": visible breadcrumb count differs from JSON-LD");
        const lastList = list[list.length - 1];
        const lastVisible = visibleCrumbs[visibleCrumbs.length - 1];
        expect(lastList && lastList.name === lastVisible.name, rel + ": final breadcrumb label differs from visible page");
      }
      for (const item of list) {
        expect(item && item["@type"] === "ListItem" && Number.isInteger(item.position) && item.position > 0 && !!item.name, rel + ": malformed BreadcrumbList item");
      }
    }
  }
  if (pagePath.match(/^\/(es|en)\/(analisis|analysis)\//)) {
    const article = findType(objects, "Article");
    expect(!!article, rel + ": missing supported Article type");
    expect(!types.includes("TechArticle"), rel + ": TechArticle should be normalized to Article for Google Article markup");
    if (article) {
      expect(!!article.headline, rel + ": Article missing headline");
      const visibleHeadline = textContent((html.match(/<h1\b[^>]*>([\s\S]*?)<\/h1>/i) || [])[1]);
      expect(article.headline === visibleHeadline, rel + ": Article headline differs from visible H1");
      expect(!!article.description, rel + ": Article missing description");
      expect(article.description === meta(html, "name", "description"), rel + ": Article description differs from meta description");
      expect(!!article.datePublished, rel + ": Article missing datePublished");
      expect(!!article.dateModified, rel + ": Article missing dateModified");
      expect(article.image && (Array.isArray(article.image) ? article.image.length : true), rel + ": Article missing image");
      expect(article.author && article.author["@type"] === "Person" && article.author.name && article.author.url, rel + ": Article author fields incomplete");
      expect(article.publisher && article.publisher["@type"] === "Organization" && article.publisher.name && article.publisher.url, rel + ": Article publisher fields incomplete");
      expect(article.mainEntityOfPage && article.mainEntityOfPage["@id"] === url, rel + ": Article mainEntityOfPage mismatch");
    }
  } else if (pagePath === "/") {
    expect(!!findType(objects, "WebSite"), rel + ": missing WebSite");
    expect(!!findType(objects, "Organization"), rel + ": missing Organization");
    expect(!!findType(objects, "Person"), rel + ": missing Person");
  } else if (pagePath === "/en/") {
    expect(!!findType(objects, "WebPage"), rel + ": missing WebPage");
  } else if (/\/(articulos|articles|series\/llm-engineering|laboratorios|labs|rutas|paths)\/$/.test(pagePath)) {
    const collection = findType(objects, "CollectionPage");
    expect(!!collection, rel + ": missing CollectionPage");
    if (collection) {
      expect(!!collection.name && !!collection.description && collection.url === url, rel + ": CollectionPage fields incomplete");
      const list = collection.mainEntity && collection.mainEntity.itemListElement;
      expect(Array.isArray(list) && list.length > 0, rel + ": CollectionPage ItemList empty");
      if (list && collection.mainEntity.numberOfItems !== list.length) fail(rel + ": ItemList numberOfItems mismatch");
      counts[pagePath] = list ? list.length : 0;
    }
  } else if (/\/(sobre|about)\/$/.test(pagePath)) {
    expect(!!findType(objects, "AboutPage"), rel + ": missing AboutPage");
  } else if (/\/(contacto|contact)\/$/.test(pagePath)) {
    expect(!!findType(objects, "ContactPage"), rel + ": missing ContactPage");
  }
  const ogImage = meta(html, "property", "og:image");
  if (ogImage) {
    const imageUrl = new URL(ogImage, ORIGIN);
    expect(imageUrl.origin === ORIGIN, rel + ": og:image not on canonical host");
    if (imageUrl.origin === ORIGIN) {
      const localImage = path.join(ROOT, decodeURIComponent(imageUrl.pathname).replace(/^\//, ""));
      expect(fs.existsSync(localImage), rel + ": missing OG image asset " + imageUrl.pathname);
    }
  }
}
const typesCount = {};
for (const url of urls) {
  const html = fs.readFileSync(pathToFile(url), "utf8");
  for (const obj of jsonLd(html, url)) if (obj && obj["@type"]) typesCount[obj["@type"]] = (typesCount[obj["@type"]] || 0) + 1;
}
console.log("Structured data audit: " + urls.length + " sitemap pages");
console.log("Types: " + JSON.stringify(typesCount));
console.log("Collection item counts: " + JSON.stringify(counts));
console.log("Failures: " + failures.length);
if (failures.length) { console.error(failures.join("\n")); process.exitCode = 1; }
else console.log("ALL STRUCTURED DATA CHECKS PASSED");
