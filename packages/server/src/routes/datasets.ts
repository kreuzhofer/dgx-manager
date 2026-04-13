import { Router } from "express";
import {
  mkdirSync,
  statSync,
  createReadStream,
  copyFileSync,
  unlinkSync,
  existsSync,
} from "fs";
import { rm } from "fs/promises";
import { createInterface } from "readline";
import multer from "multer";
import { prisma } from "../prisma.js";
import { SHARED_STORAGE } from "../env.js";
import { broadcast as sseBroadcast } from "../sse.js";

export const datasetsRouter = Router();

const DATASETS_DIR = `${SHARED_STORAGE}/datasets`;
mkdirSync(DATASETS_DIR, { recursive: true });

const upload = multer({ dest: "/tmp/dgx-uploads" });

/** Auto-detect dataset format from a parsed JSON row. */
function detectFormat(row: Record<string, unknown>): string {
  if ("conversations" in row) return "sharegpt";
  if ("messages" in row) return "openai";
  if ("instruction" in row) return "instruct";
  if ("question" in row || "context" in row) return "qa";
  return "jsonl";
}

/** Count lines and detect format from a JSONL file. */
async function analyzeFile(
  filePath: string
): Promise<{ sampleCount: number; detectedFormat: string }> {
  let lineCount = 0;
  let detectedFormat = "jsonl";
  const rl = createInterface({ input: createReadStream(filePath) });
  for await (const line of rl) {
    if (lineCount === 0 && line.trim()) {
      try {
        const parsed = JSON.parse(line);
        detectedFormat = detectFormat(parsed);
      } catch {
        // not valid JSON — leave as jsonl
      }
    }
    if (line.trim()) lineCount++;
  }
  return { sampleCount: lineCount, detectedFormat };
}

// List all datasets
datasetsRouter.get("/", async (_req, res) => {
  const datasets = await prisma.dataset.findMany({
    orderBy: { createdAt: "desc" },
  });
  res.json(datasets);
});

// Get single dataset
datasetsRouter.get("/:id", async (req, res) => {
  const dataset = await prisma.dataset.findUnique({
    where: { id: req.params.id },
  });
  if (!dataset) return res.status(404).json({ error: "Dataset not found" });
  res.json(dataset);
});

// Preview first N rows
datasetsRouter.get("/:id/preview", async (req, res) => {
  const dataset = await prisma.dataset.findUnique({
    where: { id: req.params.id },
  });
  if (!dataset) return res.status(404).json({ error: "Dataset not found" });

  if (dataset.source === "huggingface") {
    return res.json({
      preview: [],
      message: "Preview not available for HuggingFace datasets",
    });
  }

  if (!dataset.path || !existsSync(dataset.path)) {
    return res.status(404).json({ error: "Dataset file not found on disk" });
  }

  const limit = Math.min(parseInt(req.query.limit as string) || 10, 100);
  const rows: unknown[] = [];
  const rl = createInterface({ input: createReadStream(dataset.path) });

  for await (const line of rl) {
    if (!line.trim()) continue;
    try {
      rows.push(JSON.parse(line));
    } catch {
      rows.push({ _raw: line });
    }
    if (rows.length >= limit) break;
  }
  rl.close();

  res.json({ preview: rows });
});

// Create dataset — multipart upload or JSON body
datasetsRouter.post("/", upload.single("file"), async (req, res) => {
  const { name, description, format, source, path, huggingfaceId } = req.body;

  if (!name) {
    return res.status(400).json({ error: "name is required" });
  }

  const resolvedSource = source || (req.file ? "upload" : "path");

  // Handle file upload
  if (req.file) {
    const dataset = await prisma.dataset.create({
      data: {
        name,
        description: description || null,
        format: format || "jsonl",
        source: "upload",
      },
    });

    const destDir = `${DATASETS_DIR}/${dataset.id}`;
    mkdirSync(destDir, { recursive: true });
    const destPath = `${destDir}/${req.file.originalname}`;
    copyFileSync(req.file.path, destPath);
    unlinkSync(req.file.path);

    const stats = statSync(destPath);
    const { sampleCount, detectedFormat } = await analyzeFile(destPath);

    const updated = await prisma.dataset.update({
      where: { id: dataset.id },
      data: {
        path: destPath,
        size: stats.size,
        sampleCount,
        format: format && format !== "auto" ? format : detectedFormat,
      },
    });

    sseBroadcast({ type: "dataset:created", payload: updated });
    return res.status(201).json(updated);
  }

  // Handle HuggingFace import
  if (resolvedSource === "huggingface") {
    if (!huggingfaceId) {
      return res
        .status(400)
        .json({ error: "huggingfaceId is required for HuggingFace datasets" });
    }
    const dataset = await prisma.dataset.create({
      data: {
        name,
        description: description || null,
        format: format || "jsonl",
        source: "huggingface",
        huggingfaceId,
      },
    });
    sseBroadcast({ type: "dataset:created", payload: dataset });
    return res.status(201).json(dataset);
  }

  // Handle existing path reference
  if (!path) {
    return res
      .status(400)
      .json({ error: "path is required, or upload a file" });
  }

  let size: number | null = null;
  let sampleCount: number | null = null;
  let detectedFormat = format || "jsonl";

  if (existsSync(path)) {
    const stats = statSync(path);
    size = stats.size;
    const analysis = await analyzeFile(path);
    sampleCount = analysis.sampleCount;
    if (!format || format === "auto") {
      detectedFormat = analysis.detectedFormat;
    }
  }

  const dataset = await prisma.dataset.create({
    data: {
      name,
      description: description || null,
      format: detectedFormat,
      source: "path",
      path,
      size,
      sampleCount,
    },
  });

  sseBroadcast({ type: "dataset:created", payload: dataset });
  res.status(201).json(dataset);
});

// Delete dataset
datasetsRouter.delete("/:id", async (req, res) => {
  const dataset = await prisma.dataset.findUnique({
    where: { id: req.params.id },
  });
  if (!dataset) return res.status(404).json({ error: "Dataset not found" });

  await prisma.dataset.delete({ where: { id: req.params.id } });

  // Clean up uploaded files
  if (dataset.source === "upload") {
    const dir = `${DATASETS_DIR}/${dataset.id}`;
    await rm(dir, { recursive: true, force: true }).catch(() => {});
  }

  sseBroadcast({ type: "dataset:deleted", payload: { id: dataset.id } });
  res.json({ deleted: true });
});
