import "dotenv/config";
import crypto from "node:crypto";
import express from "express";
import fs from "fs/promises";
import { watch as fsWatch } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import pdfParse from "pdf-parse";
import { buildCheckerPrompt } from "./prompts.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** PDF.js (pdf-parse) пишет в console.log шум про TrueType hinting; на текст не влияет. */
const origConsoleLog = console.log;
let pdfParseLogGuard = 0;

async function withSuppressedPdfJsTtWarnings(fn) {
  pdfParseLogGuard += 1;
  if (pdfParseLogGuard === 1) {
    console.log = (...args) => {
      const first = args[0];
      if (typeof first === "string" && first.startsWith("Warning: TT:")) return;
      origConsoleLog.apply(console, args);
    };
  }
  try {
    return await fn();
  } finally {
    pdfParseLogGuard -= 1;
    if (pdfParseLogGuard === 0) console.log = origConsoleLog;
  }
}

function envStr(name, fallback) {
  const v = process.env[name];
  if (v == null || String(v).trim() === "") return fallback;
  return String(v);
}

function envInt(name, fallback) {
  const v = process.env[name];
  if (v == null || String(v).trim() === "") return fallback;
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? n : fallback;
}

/** Для RESUME_/VACANCY_ лимитов: 0 = без лимита; пустая строка в .env = значение по умолчанию ниже. */
function envMaxChars(name, fallback) {
  const v = process.env[name];
  if (v == null || String(v).trim() === "") return fallback;
  const n = parseInt(v, 10);
  if (!Number.isFinite(n) || n < 0) return fallback;
  return n;
}

const CONFIG = {
  port: envInt("PORT", 3000),
  ollamaBaseUrl: envStr("OLLAMA_BASE_URL", "http://127.0.0.1:11434").replace(/\/$/, ""),
  ollamaModel: envStr("OLLAMA_MODEL", "qwen3:1.7b"),
  ollamaNumCtx: envInt("OLLAMA_NUM_CTX", 4096),
  resumesDir: envStr("RESUMES_DIR", "./resumes"),
  vacanciesDir: envStr("VACANCIES_DIR", "./vacancies"),
  vacancyFile: envStr("VACANCY_FILE", ""),
  resumeTextMaxChars: envMaxChars("RESUME_TEXT_MAX_CHARS", 12000),
  vacancyTextMaxChars: envMaxChars("VACANCY_TEXT_MAX_CHARS", 8000),
  jsonBodyLimit: envStr("JSON_BODY_LIMIT", "2mb"),
  batchTakeMax: Math.max(1, envInt("BATCH_TAKE_MAX", 500)),
  livereloadDebounceMs: Math.max(0, envInt("LIVERELOAD_DEBOUNCE_MS", 300)),
  resultsFile: envStr("RESULTS_FILE", "results.json"),
};

const RESULTS_FILE = path.isAbsolute(CONFIG.resultsFile)
  ? CONFIG.resultsFile
  : path.join(__dirname, CONFIG.resultsFile);

/** Порядок первых вакансий в списке (имена файлов в папке vacancies). Остальные — по алфавиту. */
const VACANCY_ORDER_FIRST = [
  "инженер-конструктор.vc",
  "инженер-программист.vc",
  "технолог.vc",
  "инженер-электроник.vc",
];

function vacancyOrderRank(name) {
  const low = name.toLowerCase();
  const i = VACANCY_ORDER_FIRST.findIndex((p) => p.toLowerCase() === low);
  return i === -1 ? VACANCY_ORDER_FIRST.length : i;
}

function sortVacancyFilenames(names) {
  return [...names].sort((a, b) => {
    const ra = vacancyOrderRank(a);
    const rb = vacancyOrderRank(b);
    if (ra !== rb) return ra - rb;
    return a.toLowerCase().localeCompare(b.toLowerCase());
  });
}

async function readResults() {
  try {
    const raw = await fs.readFile(RESULTS_FILE, "utf-8");
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

async function appendResult(entry) {
  const all = await readResults();
  all.push(entry);
  await fs.writeFile(RESULTS_FILE, JSON.stringify(all, null, 2), "utf-8");
}

function matchesStoredVacancy(entry, vacancyKey, legacyVacancyId) {
  if (entry.vacancyKey === vacancyKey) return true;
  // Тот же файл вакансии по id, даже если vacancyKey в старых записях отличался (переименования и т.п.)
  if (legacyVacancyId && String(entry.vacancyId || "").trim() === String(legacyVacancyId).trim()) return true;
  return false;
}

/** Последняя успешная оценка (file + вакансия), без повторного вызова модели. */
function findCachedEvaluation(results, file, vacancyKey, legacyVacancyId) {
  const candidates = results.filter(
    (e) =>
      e.file === file &&
      matchesStoredVacancy(e, vacancyKey, legacyVacancyId) &&
      e.error !== true &&
      e.score != null &&
      Number.isFinite(Number(e.score)),
  );
  if (!candidates.length) return null;
  candidates.sort((a, b) => String(b.timestamp || "").localeCompare(String(a.timestamp || "")));
  return candidates[0];
}

const app = express();
app.use(express.json({ limit: CONFIG.jsonBodyLimit }));

function sendSse(res, obj) {
  res.write(`data: ${JSON.stringify(obj)}\n\n`);
}

function trimSpace(text) {
  return text.replace(/\s+/g, " ").trim();
}

function truncateForModel(text, maxChars) {
  if (maxChars <= 0) return String(text ?? "");
  const t = String(text ?? "");
  if (t.length <= maxChars) return t;
  const tail = "\n… [обрезано по лимиту контекста]";
  const sliceLen = Math.max(0, maxChars - tail.length);
  return t.slice(0, sliceLen) + tail;
}

function tryJsonParseObject(str) {
  try {
    const o = JSON.parse(str);
    if (o && typeof o === "object" && !Array.isArray(o)) return o;
  } catch {
    /* fall through */
  }
  return null;
}

/**
 * Если модель выдала кривой JSON (explanation без кавычек, перенос в ключе "explanation" и т.д.).
 */
function looseParseCheckerResponse(s) {
  const scoreM = s.match(/"score"\s*:\s*(\d+)/);
  const valM = s.match(/"value"\s*:\s*(true|false)\b/i);
  if (!scoreM || !valM) return null;
  const score = Number(scoreM[1]);
  const value = valM[1].toLowerCase() === "true";
  if (!Number.isFinite(score)) return null;

  let explanation = "";
  const explRe = /"(?:explanation|expla\s*nation)"\s*:\s*/gi;
  explRe.lastIndex = 0;
  const km = explRe.exec(s);
  if (km) {
    let rest = s.slice(km.index + km[0].length).trim();
    if (rest.startsWith('"')) {
      let i = 1;
      let out = "";
      while (i < rest.length) {
        const c = rest[i];
        if (c === "\\" && i + 1 < rest.length) {
          const n = rest[i + 1];
          if (n === "n") out += "\n";
          else if (n === "r") out += "\r";
          else if (n === "t") out += "\t";
          else out += n;
          i += 2;
          continue;
        }
        if (c === '"') break;
        out += c;
        i += 1;
      }
      explanation = out;
    } else {
      const braceAt = rest.indexOf("}");
      const body = braceAt >= 0 ? rest.slice(0, braceAt) : rest;
      explanation = body.replace(/,\s*$/, "").trim();
    }
  }

  return { score, value, explanation };
}

function parseModelJson(raw) {
  let s = String(raw ?? "").trim();
  s = s.replace(/^\uFEFF/, "");
  s = s.replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/i, "").trim();

  let o = tryJsonParseObject(s);
  if (o) return o;

  const m = s.match(/\{[\s\S]*\}/);
  if (m) {
    o = tryJsonParseObject(m[0]);
    if (o) return o;
    const loose = looseParseCheckerResponse(m[0]);
    if (loose) return loose;
  }

  const loose = looseParseCheckerResponse(s);
  if (loose) return loose;

  throw new Error("Model response is not valid JSON");
}

async function readPdfText(filePath) {
  const buf = await fs.readFile(filePath);
  const data = await withSuppressedPdfJsTtWarnings(() => pdfParse(buf));
  return trimSpace(data.text || "");
}

async function ollamaGenerate(prompt) {
  const baseUrl = CONFIG.ollamaBaseUrl;
  const model = CONFIG.ollamaModel;
  const url = `${baseUrl}/api/generate`;
  const numCtx = CONFIG.ollamaNumCtx;
  const payload = {
    model,
    prompt,
    stream: false,
  };
  if (Number.isFinite(numCtx) && numCtx > 0) {
    payload.options = { num_ctx: numCtx };
  }
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(`Ollama ${r.status}: ${t || r.statusText}`);
  }
  const j = await r.json();
  return j.response ?? j;
}

function resolveResumesDir() {
  const raw = CONFIG.resumesDir.trim();
  return path.isAbsolute(raw) ? raw : path.resolve(__dirname, raw);
}

async function listPdfs(dir, take) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const pdfs = entries
    .filter((e) => e.isFile() && e.name.toLowerCase().endsWith(".pdf"))
    .map((e) => e.name)
    .sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()))
    .slice(0, take)
    .map((name) => path.join(dir, name));
  return pdfs;
}

/** Только имя файла .pdf без пути (защита от path traversal). */
function safePdfBasename(name) {
  if (name == null || typeof name !== "string") return null;
  const base = path.basename(name.trim());
  if (!base || base !== name.trim()) return null;
  if (!base.toLowerCase().endsWith(".pdf")) return null;
  if (base.includes("\0") || base.includes("..")) return null;
  return base;
}

function pdfDownloadUrl(filename) {
  const safe = safePdfBasename(filename);
  return safe ? `/api/resumes/${encodeURIComponent(safe)}` : null;
}

function isPathInsideDir(filePath, dirPath) {
  const f = path.resolve(filePath);
  const d = path.resolve(dirPath);
  const rel = path.relative(d, f);
  return rel !== "" && !rel.startsWith("..") && !path.isAbsolute(rel);
}

function resolveVacanciesDir() {
  const raw = CONFIG.vacanciesDir.trim();
  return path.isAbsolute(raw) ? raw : path.resolve(__dirname, raw);
}

function safeVacancyBasename(name) {
  if (name == null || typeof name !== "string") return null;
  const trimmed = name.trim();
  const base = path.basename(trimmed);
  if (!base || base !== trimmed) return null;
  // разрешаем .vc и .txt
  const lower = base.toLowerCase();
  if (!lower.endsWith(".vc") && !lower.endsWith(".txt")) return null;
  if (base.includes("\0") || base.includes("..")) return null;
  return base;
}

app.get("/api/config", (req, res) => {
  res.json({ ollamaModel: CONFIG.ollamaModel });
});

app.get("/api/results", async (req, res) => {
  res.json(await readResults());
});

app.post("/api/results/delete", async (req, res) => {
  try {
    const { file, vacancyId, timestamp } = req.body || {};
    const safe = safePdfBasename(file);
    if (!safe) {
      res.status(400).json({ ok: false, message: "Некорректное имя файла" });
      return;
    }
    const ts = timestamp == null ? "" : String(timestamp).trim();
    if (!ts) {
      res.status(400).json({ ok: false, message: "Нужен timestamp записи" });
      return;
    }
    const vid = vacancyId == null ? "" : String(vacancyId);
    const all = await readResults();
    const next = all.filter(
      (e) => !(e.file === safe && String(e.timestamp || "") === ts && String(e.vacancyId || "") === vid),
    );
    if (next.length === all.length) {
      res.status(404).json({ ok: false, message: "Запись не найдена" });
      return;
    }
    await fs.writeFile(RESULTS_FILE, JSON.stringify(next, null, 2), "utf-8");
    res.json({ ok: true, removed: all.length - next.length });
  } catch (e) {
    res.status(500).json({ ok: false, message: e.message || String(e) });
  }
});

app.get("/api/resumes/count", async (req, res) => {
  try {
    const dir = resolveResumesDir();
    const entries = await fs.readdir(dir, { withFileTypes: true });
    const count = entries.filter((e) => e.isFile() && e.name.toLowerCase().endsWith(".pdf")).length;
    res.json({ count });
  } catch {
    res.json({ count: 0 });
  }
});

app.get("/api/resumes/:filename", async (req, res) => {
  try {
    const safe = safePdfBasename(req.params.filename);
    if (!safe) {
      res.status(400).type("text/plain").send("Некорректное имя файла");
      return;
    }
    const dir = resolveResumesDir();
    const full = path.join(dir, safe);
    if (!isPathInsideDir(full, dir)) {
      res.status(403).end();
      return;
    }
    await fs.access(full);
    res.download(full, safe);
  } catch {
    res.status(404).type("text/plain").send("Файл не найден");
  }
});

app.get("/api/vacancies", async (req, res) => {
  try {
    const dir = resolveVacanciesDir();
    const entries = await fs.readdir(dir, { withFileTypes: true });
    const files = entries
      .filter((e) => e.isFile())
      .map((e) => e.name)
      .filter((n) => {
        const low = n.toLowerCase();
        return low.endsWith(".vc") || low.endsWith(".txt");
      });
    const sorted = sortVacancyFilenames(files);
    const out = sorted.map((name) => ({
      id: name,
      label: path.basename(name, path.extname(name)),
    }));
    res.json({ files: out });
  } catch {
    res.json({ files: [] });
  }
});

app.get("/api/vacancies/:filename", async (req, res) => {
  try {
    const safe = safeVacancyBasename(req.params.filename);
    if (!safe) {
      res.status(400).type("text/plain").send("Некорректное имя файла");
      return;
    }
    const dir = resolveVacanciesDir();
    const full = path.join(dir, safe);
    if (!isPathInsideDir(full, dir)) {
      res.status(403).end();
      return;
    }
    const text = await fs.readFile(full, "utf-8");
    res.json({ id: safe, label: path.basename(safe, path.extname(safe)), text });
  } catch {
    res.status(404).type("text/plain").send("Файл не найден");
  }
});

app.post("/api/data", async (req, res) => {
  const startedAll = Date.now();

  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  if (typeof res.flushHeaders === "function") res.flushHeaders();

  const elapsed = () => (Date.now() - startedAll) / 1000;

  try {
    const take = Math.min(
      CONFIG.batchTakeMax,
      Math.max(1, parseInt(req.body?.take ?? "10", 10) || 10),
    );
    let jobDescription = (req.body?.vacancyText ?? "").trim();
    const vacancyId = (req.body?.vacancyId ?? "").trim();
    let vacancyKey = "";

    if (!jobDescription && vacancyId) {
      const safe = safeVacancyBasename(vacancyId);
      if (!safe) {
        sendSse(res, { type: "error", message: "Некорректный vacancyId", elapsed: elapsed() });
        res.end();
        return;
      }
      const dir = resolveVacanciesDir();
      const full = path.join(dir, safe);
      if (!isPathInsideDir(full, dir)) {
        sendSse(res, { type: "error", message: "vacancyId вне папки vacancies", elapsed: elapsed() });
        res.end();
        return;
      }
      jobDescription = trimSpace(await fs.readFile(full, "utf-8"));
      vacancyKey = `id:${safe}`;
      sendSse(res, { type: "status", message: `Вакансия: ${safe}`, elapsed: elapsed() });
    } else if (!jobDescription) {
      const vacFile = CONFIG.vacancyFile.trim();
      if (!vacFile) {
        sendSse(res, { type: "error", message: "Укажите текст вакансии в форме или VACANCY_FILE в .env", elapsed: elapsed() });
        res.end();
        return;
      }
      const vacPath = path.isAbsolute(vacFile) ? vacFile : path.resolve(__dirname, vacFile);
      jobDescription = trimSpace(await fs.readFile(vacPath, "utf-8"));
      vacancyKey = `file:${path.basename(vacPath)}`;
      sendSse(res, { type: "status", message: `Вакансия: ${path.basename(vacPath)}`, elapsed: elapsed() });
    } else {
      sendSse(res, { type: "status", message: "Вакансия из формы", elapsed: elapsed() });
      const safeVac = vacancyId ? safeVacancyBasename(vacancyId) : null;
      vacancyKey = safeVac ? `id:${safeVac}` : `text:${crypto.createHash("sha256").update(jobDescription).digest("hex")}`;
    }

    const resumesDir = resolveResumesDir();
    await fs.access(resumesDir).catch(() => {
      throw new Error(`Папка резюме не найдена: ${resumesDir}`);
    });

    const pdfPaths = await listPdfs(resumesDir, take);
    if (pdfPaths.length === 0) {
      sendSse(res, { type: "error", message: `Нет PDF в ${resumesDir}`, elapsed: elapsed() });
      res.end();
      return;
    }

    sendSse(res, {
      type: "batch",
      total: pdfPaths.length,
      take,
      resumesDir,
      elapsed: elapsed(),
    });

    const priorResults = await readResults();
    const resumeMax = CONFIG.resumeTextMaxChars;
    const vacancyMax = CONFIG.vacancyTextMaxChars;
    const vacancyForPrompt = truncateForModel(jobDescription, vacancyMax);
    const rows = [];

    for (let i = 0; i < pdfPaths.length; i++) {
      const pdfPath = pdfPaths[i];
      const file = path.basename(pdfPath);
      const pdfUrl = pdfDownloadUrl(file);
      const t0 = Date.now();

      const cached = findCachedEvaluation(priorResults, file, vacancyKey, vacancyId);
      sendSse(res, {
        type: "resume_start",
        file,
        pdfUrl,
        index: i + 1,
        total: pdfPaths.length,
        elapsed: elapsed(),
        cached: Boolean(cached),
      });

      if (cached) {
        let value = cached.value;
        if (typeof value === "string") {
          value = value.toLowerCase() === "true";
        }
        const row = {
          file,
          pdfUrl,
          score: cached.score,
          value,
          explanation: cached.explanation ?? "",
          seconds: 0,
          elapsedTotal: elapsed(),
          error: false,
          cached: true,
        };
        rows.push(row);
        sendSse(res, { type: "resume_result", ...row });
        continue;
      }

      const resumeText = await readPdfText(pdfPath);
      if (!resumeText) {
        sendSse(res, {
          type: "resume_result",
          file,
          pdfUrl,
          score: null,
          value: null,
          explanation: "Не удалось извлечь текст из PDF",
          seconds: (Date.now() - t0) / 1000,
          elapsedTotal: elapsed(),
          error: true,
        });
        continue;
      }

      const resumeForPrompt = truncateForModel(resumeText, resumeMax);
      const prompt = buildCheckerPrompt(resumeForPrompt, vacancyForPrompt);
      const raw = await ollamaGenerate(prompt);
      let parsed;
      try {
        parsed = parseModelJson(raw);
      } catch (e) {
        sendSse(res, {
          type: "resume_result",
          file,
          pdfUrl,
          score: null,
          value: null,
          explanation: `Ошибка разбора ответа модели: ${e.message}`,
          seconds: (Date.now() - t0) / 1000,
          elapsedTotal: elapsed(),
          error: true,
        });
        continue;
      }

      const sec = (Date.now() - t0) / 1000;
      let value = parsed.value;
      if (typeof value === "string") {
        value = value.toLowerCase() === "true";
      }

      const row = {
        file,
        pdfUrl,
        score: parsed.score,
        value,
        explanation: parsed.explanation ?? "",
        seconds: sec,
        elapsedTotal: elapsed(),
        error: false,
        cached: false,
      };
      rows.push(row);
      sendSse(res, { type: "resume_result", ...row });

      await appendResult({
        file,
        vacancyId,
        vacancyKey,
        vacancyLabel: path.basename(vacancyId, path.extname(vacancyId)),
        score: parsed.score,
        value,
        explanation: parsed.explanation ?? "",
        seconds: sec,
        timestamp: new Date().toISOString(),
      });
    }

    const sorted = [...rows].sort((a, b) => Number(b.score ?? -1) - Number(a.score ?? -1));
    const totalSeconds = (Date.now() - startedAll) / 1000;
    sendSse(res, { type: "summary", rows: sorted, totalSeconds, elapsed: elapsed() });
    sendSse(res, { type: "done", totalSeconds });
    res.end();
  } catch (e) {
    sendSse(res, { type: "error", message: e.message || String(e), elapsed: elapsed() });
    res.end();
  }
});

// --- live reload ---
const liveClients = new Set();

app.get("/api/livereload", (req, res) => {
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  if (typeof res.flushHeaders === "function") res.flushHeaders();
  liveClients.add(res);
  req.on("close", () => liveClients.delete(res));
});

function notifyReload() {
  for (const c of liveClients) {
    c.write(`data: reload\n\n`);
  }
}

let reloadTimeout = null;
function debouncedReload() {
  if (reloadTimeout) clearTimeout(reloadTimeout);
  reloadTimeout = setTimeout(notifyReload, CONFIG.livereloadDebounceMs);
}

for (const dir of [path.join(__dirname, "public"), path.join(__dirname, "vacancies")]) {
  try { fsWatch(dir, { recursive: true }, debouncedReload); } catch {}
}

app.use(express.static(path.join(__dirname, "public")));

const port = CONFIG.port;
app.listen(port, () => {
  console.log(`resume-checker listening on http://localhost:${port}`);
}).on("error", (err) => {
  console.error(`Не удалось запустить сервер на порту ${port}:`, err.message);
  process.exit(1);
});
