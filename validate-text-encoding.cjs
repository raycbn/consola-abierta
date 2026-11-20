const fs = require("fs");
const path = require("path");

const ROOT = __dirname;
const INCLUDED = new Set([".html", ".json", ".md", ".txt", ".xml", ".yml", ".yaml"]);
const IGNORED_DIRS = new Set([".git", "node_modules"]);
const CHECKS = [
  { label: "replacement character", pattern: /\uFFFD/g },
  { label: "UTF-8 mojibake", pattern: /(?:Ã[\u0080-\u00BF]|Â[\u0080-\u00BF])/g },
  { label: "letter replaced by question mark inside a word", pattern: /\p{L}{2,}\?[\p{L}]{1,}\b/giu },
  { label: "standalone separator replaced by question mark", pattern: /\s\?\s/g }
];

const failures = [];
let scanned = 0;

function cleanForTextChecks(line) {
  return line
    .replace(/<script\b[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[\s\S]*?<\/style>/gi, " ")
    .replace(/<(pre|code)\b[\s\S]*?<\/\1>/gi, " ")
    .replace(/\b(?:href|src|action|poster)\s*=\s*(["'])[^"']*\1/gi, " ")
    .replace(/https?:\/\/[^\s<"']+/gi, " ");
}

function walk(directory) {
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      if (!IGNORED_DIRS.has(entry.name)) walk(path.join(directory, entry.name));
      continue;
    }
    if (!entry.isFile() || !INCLUDED.has(path.extname(entry.name).toLowerCase())) continue;

    const file = path.join(directory, entry.name);
    const relative = path.relative(ROOT, file).split(path.sep).join("/");
    const content = fs.readFileSync(file, "utf8");
    scanned++;

    const lines = content.split(/\r?\n/);
    for (let index = 0; index < lines.length; index++) {
      const searchable = cleanForTextChecks(lines[index]);
      for (const check of CHECKS) {
        check.pattern.lastIndex = 0;
        const match = check.pattern.exec(searchable);
        if (match) {
          failures.push(relative + ":" + (index + 1) + ": " + check.label + " — " + lines[index].trim().slice(0, 180));
        }
      }
    }
  }
}

walk(ROOT);
console.log("Text encoding audit: " + scanned + " source files scanned.");
console.log("Failures: " + failures.length);
if (failures.length) {
  console.error(failures.join("\n"));
  process.exitCode = 1;
} else {
  console.log("ALL TEXT ENCODING CHECKS PASSED");
}
