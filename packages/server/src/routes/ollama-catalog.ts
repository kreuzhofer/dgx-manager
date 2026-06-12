import { Router } from "express";
import {
  readCatalog,
  readEnabled,
  writeEnabled,
  readAvailable,
} from "../ollama/catalog-store.js";
import { refreshCatalog } from "../ollama/catalog-fetcher.js";
import { broadcast as sseBroadcast } from "../sse.js";

export const ollamaCatalogRouter = Router();

/**
 * @openapi
 * /api/ollama-catalog/catalog:
 *   get:
 *     tags: [Models]
 *     summary: Return the cached Ollama model catalog
 *     description: >
 *       Returns the locally-cached Ollama model catalog fetched from ollama.com.
 *       Each entry includes `name`, available `sizes` (e.g. ["7b", "13b"]), and
 *       VRAM estimates per size. The catalog is used by the deploy form to populate
 *       the Ollama model selector with accurate size/VRAM information before the
 *       model is pulled to any node. Refresh the catalog via
 *       POST /api/ollama-catalog/catalog/refresh.
 *     responses:
 *       '200':
 *         description: Catalog object with `entries` array and `fetchedAt` timestamp
 */
ollamaCatalogRouter.get("/catalog", async (_req, res) => {
  res.json(await readCatalog());
});

/**
 * @openapi
 * /api/ollama-catalog/catalog/refresh:
 *   post:
 *     tags: [Models]
 *     summary: Fetch and cache the latest Ollama model catalog from ollama.com
 *     description: >
 *       Scrapes or fetches the current Ollama model catalog from the upstream source
 *       and persists it to the local cache file. Broadcasts `ollama-catalog:updated`
 *       over SSE so the dashboard can re-render the model selector without a reload.
 *       Call this after new models are published on ollama.com to pick them up in
 *       the deploy UI. Returns the refreshed catalog. May fail with 502 if the
 *       upstream is unreachable.
 *     responses:
 *       '200':
 *         description: Refreshed catalog object
 *       '502':
 *         description: Failed to fetch from upstream
 */
ollamaCatalogRouter.post("/catalog/refresh", async (_req, res) => {
  try {
    const result = await refreshCatalog();
    sseBroadcast({ type: "ollama-catalog:updated", payload: { reason: "refresh" } });
    res.json(result);
  } catch (err) {
    res.status(502).json({ error: String((err as Error).message || err) });
  }
});

/**
 * @openapi
 * /api/ollama-catalog/enabled:
 *   get:
 *     tags: [Models]
 *     summary: Get the list of enabled Ollama models
 *     description: >
 *       Returns the operator-curated list of Ollama model names that are shown in
 *       the deploy form. Only models in this list appear as deploy options, keeping
 *       the UI manageable on clusters with large catalogs.
 *     responses:
 *       '200':
 *         description: '{ enabled: string[] } — array of enabled model names'
 */
ollamaCatalogRouter.get("/enabled", async (_req, res) => {
  res.json({ enabled: await readEnabled() });
});

/**
 * @openapi
 * /api/ollama-catalog/enabled:
 *   put:
 *     tags: [Models]
 *     summary: Replace the enabled Ollama models list
 *     description: >
 *       Replaces the full list of enabled Ollama model names. Pass an array of model
 *       name strings (bare names without size suffix, e.g. `["llama3", "mistral"]`).
 *       Broadcasts `ollama-catalog:updated` over SSE. The deploy form filters the
 *       catalog to only show models in this list.
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required: [enabled]
 *             properties:
 *               enabled: { type: array, items: { type: string }, description: "Full replacement list of enabled model names." }
 *     responses:
 *       '200':
 *         description: '{ enabled: string[] } — the saved list'
 *       '400':
 *         description: enabled must be an array of strings
 */
ollamaCatalogRouter.put("/enabled", async (req, res) => {
  const body = req.body as { enabled?: unknown };
  if (!Array.isArray(body.enabled)) {
    return res.status(400).json({ error: "enabled must be an array of strings" });
  }
  const names = body.enabled.filter((x): x is string => typeof x === "string");
  await writeEnabled(names);
  sseBroadcast({ type: "ollama-catalog:updated", payload: { reason: "enabled-changed" } });
  res.json({ enabled: names });
});

/**
 * @openapi
 * /api/ollama-catalog/available:
 *   get:
 *     tags: [Models]
 *     summary: Get the intersection of enabled and catalogued Ollama models
 *     description: >
 *       Returns models that are both in the enabled list and present in the cached
 *       catalog, enriched with size and VRAM estimate data. This is the filtered
 *       view used by the deploy form — it excludes enabled names that don't appear
 *       in the catalog (useful when the catalog is stale or a model was removed).
 *     responses:
 *       '200':
 *         description: Array of available catalog entries with size/VRAM data
 */
ollamaCatalogRouter.get("/available", async (_req, res) => {
  res.json(await readAvailable());
});
