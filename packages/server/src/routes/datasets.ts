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

/**
 * @openapi
 * /api/datasets:
 *   get:
 *     tags: [Datasets]
 *     summary: List all registered datasets
 *     description: >
 *       Returns every Dataset record ordered by creation date descending. Datasets
 *       can be local JSONL files (uploaded or path-referenced) or HuggingFace
 *       repository identifiers. The `format` field is auto-detected from the first
 *       line (sharegpt, openai, instruct, qa, jsonl). Used by the fine-tune job form
 *       to populate the dataset selector.
 *     responses:
 *       '200':
 *         description: Array of dataset records
 */
datasetsRouter.get("/", async (_req, res) => {
  const datasets = await prisma.dataset.findMany({
    orderBy: { createdAt: "desc" },
  });
  res.json(datasets);
});

/**
 * @openapi
 * /api/datasets/{id}:
 *   get:
 *     tags: [Datasets]
 *     summary: Get a single dataset record
 *     description: >
 *       Returns the Dataset row by ID, including its path, format, sample count,
 *       and source type. Use this to confirm dataset metadata before attaching it
 *       to a fine-tune job.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '200':
 *         description: Dataset record
 *       '404':
 *         description: Dataset not found
 */
datasetsRouter.get("/:id", async (req, res) => {
  const dataset = await prisma.dataset.findUnique({
    where: { id: req.params.id },
  });
  if (!dataset) return res.status(404).json({ error: "Dataset not found" });
  res.json(dataset);
});

/**
 * @openapi
 * /api/datasets/{id}/preview:
 *   get:
 *     tags: [Datasets]
 *     summary: Preview the first N rows of a dataset file
 *     description: >
 *       Reads up to `limit` lines (default 10, max 100) from the dataset's JSONL
 *       file and returns them as a parsed JSON array. Lines that are not valid JSON
 *       are returned as `{ _raw: string }`. Not available for HuggingFace datasets
 *       (returns empty preview with a message). Returns 404 if the file is missing
 *       from disk.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *       - in: query
 *         name: limit
 *         required: false
 *         schema: { type: integer, minimum: 1, maximum: 100 }
 *         description: Number of rows to return (default 10)
 *     responses:
 *       '200':
 *         description: '{ preview: any[] } or { preview: [], message: string } for HuggingFace datasets'
 *       '404':
 *         description: Dataset or file not found
 */
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

/**
 * @openapi
 * /api/datasets:
 *   post:
 *     tags: [Datasets]
 *     summary: Register a new dataset (upload, path, or HuggingFace)
 *     description: >
 *       Supports three intake modes selected by the request: (1) **File upload**
 *       — multipart/form-data with a `file` field; the file is moved to
 *       `$SHARED_STORAGE/datasets/{id}/{filename}`, line count and format are
 *       auto-detected. (2) **Path reference** — JSON body with `path` pointing to
 *       an existing file on shared storage; metadata is read if the file exists.
 *       (3) **HuggingFace** — JSON body with `source: "huggingface"` and
 *       `huggingfaceId` (e.g. `"tatsu-lab/alpaca"`); no file is stored server-side.
 *       The `format` field is auto-detected from the first JSONL line if not
 *       explicitly provided (or set to `"auto"`).
 *     requestBody:
 *       required: true
 *       content:
 *         multipart/form-data:
 *           schema:
 *             type: object
 *             properties:
 *               file: { type: string, format: binary, description: "Dataset file to upload (JSONL)." }
 *               name: { type: string, description: "Human-readable dataset name." }
 *               description: { type: string }
 *               format: { type: string, description: "Format override: jsonl, sharegpt, openai, instruct, qa, auto." }
 *         application/json:
 *           schema:
 *             type: object
 *             required: [name]
 *             properties:
 *               name: { type: string }
 *               description: { type: string }
 *               format: { type: string }
 *               source: { type: string, enum: [path, huggingface] }
 *               path: { type: string, description: "Absolute path on shared storage for source=path." }
 *               huggingfaceId: { type: string, description: "HuggingFace dataset repo id for source=huggingface." }
 *     responses:
 *       '201':
 *         description: Created dataset record
 *       '400':
 *         description: name required, or missing path/huggingfaceId for the chosen source
 */
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

/**
 * @openapi
 * /api/datasets/{id}:
 *   delete:
 *     tags: [Datasets]
 *     summary: Delete a dataset record and its uploaded files
 *     description: >
 *       Removes the Dataset row from the database. For `source: "upload"` datasets,
 *       also recursively deletes the dataset's directory under
 *       `$SHARED_STORAGE/datasets/{id}`. Path-reference and HuggingFace datasets
 *       have no server-side files to clean up. Broadcasts `dataset:deleted` over SSE.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '200':
 *         description: '{ deleted: true }'
 *       '404':
 *         description: Dataset not found
 */
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
