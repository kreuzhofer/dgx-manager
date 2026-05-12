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

ollamaCatalogRouter.get("/catalog", async (_req, res) => {
  res.json(await readCatalog());
});

ollamaCatalogRouter.post("/catalog/refresh", async (_req, res) => {
  try {
    const result = await refreshCatalog();
    sseBroadcast({ type: "ollama-catalog:updated", payload: { reason: "refresh" } });
    res.json(result);
  } catch (err) {
    res.status(502).json({ error: String((err as Error).message || err) });
  }
});

ollamaCatalogRouter.get("/enabled", async (_req, res) => {
  res.json({ enabled: await readEnabled() });
});

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

ollamaCatalogRouter.get("/available", async (_req, res) => {
  res.json(await readAvailable());
});
