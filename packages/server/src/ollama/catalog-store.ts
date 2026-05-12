import { prisma } from "../prisma.js";
import type { OllamaCatalogEntry } from "./catalog-parser.js";

const CATALOG_KEY = "ollama.catalog.json";
const FETCHED_AT_KEY = "ollama.catalog.fetchedAt";
const ENABLED_KEY = "ollama.enabled.json";

export interface CatalogSnapshot {
  entries: OllamaCatalogEntry[];
  fetchedAt: string | null;
}

/**
 * A flat, deployable tag — what the deployments page renders in its dropdown
 * and what the user actually pulls with `ollama pull`. Sized models flatten
 * into one row per size ("llama3.1:8b", "llama3.1:70b"); sizeless models
 * (nomic-embed-text, wizardlm) flatten into a single row with `size === null`
 * and `tag === modelName`.
 */
export interface AvailableTag {
  tag: string;            // "llama3.1:8b" or "nomic-embed-text"
  modelName: string;      // "llama3.1"
  size: string | null;    // "8b" or null for sizeless models
  type: "chat" | "embedding";
  description: string;
  capabilities: string[];
}

async function readSetting(key: string): Promise<string | null> {
  const row = await prisma.setting.findUnique({ where: { key } });
  return row?.value ?? null;
}

async function writeSetting(key: string, value: string): Promise<void> {
  await prisma.setting.upsert({
    where: { key },
    create: { key, value },
    update: { value },
  });
}

export async function readCatalog(): Promise<CatalogSnapshot> {
  const [raw, fetchedAt] = await Promise.all([
    readSetting(CATALOG_KEY),
    readSetting(FETCHED_AT_KEY),
  ]);
  if (!raw) return { entries: [], fetchedAt };
  try {
    const parsed = JSON.parse(raw) as OllamaCatalogEntry[];
    return { entries: Array.isArray(parsed) ? parsed : [], fetchedAt };
  } catch {
    return { entries: [], fetchedAt };
  }
}

export async function writeCatalog(entries: OllamaCatalogEntry[]): Promise<string> {
  const fetchedAt = new Date().toISOString();
  await writeSetting(CATALOG_KEY, JSON.stringify(entries));
  await writeSetting(FETCHED_AT_KEY, fetchedAt);
  return fetchedAt;
}

export async function readEnabled(): Promise<string[]> {
  const raw = await readSetting(ENABLED_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === "string") : [];
  } catch {
    return [];
  }
}

export async function writeEnabled(tags: string[]): Promise<void> {
  // Dedupe + drop non-strings defensively — the PUT route validates, but the
  // DB shouldn't trust callers.
  const cleaned = Array.from(new Set(tags.filter((t) => typeof t === "string" && t.length > 0)));
  await writeSetting(ENABLED_KEY, JSON.stringify(cleaned));
}

/**
 * Flatten the catalog × enabled-set into the deployable rows the deployments
 * page actually wants. Enabled tags reference either:
 *   - "model:size" — must match a `OllamaCatalogEntry` whose `sizes` list
 *     contains `size`, OR
 *   - "model" (no colon) — must match an `OllamaCatalogEntry` with the same
 *     bare name and an empty `sizes` list (sizeless / embedding models).
 *
 * Entries that no longer exist in the catalog (model removed upstream) are
 * silently dropped — the UI surfaces this as the model disappearing from
 * the dropdown, which is the right outcome.
 */
export async function readAvailable(): Promise<AvailableTag[]> {
  const [{ entries }, enabled] = await Promise.all([readCatalog(), readEnabled()]);
  const byName = new Map(entries.map((e) => [e.name, e]));
  const rows: AvailableTag[] = [];
  for (const tag of enabled) {
    const [modelName, size] = tag.includes(":") ? tag.split(":", 2) : [tag, null];
    const entry = byName.get(modelName);
    if (!entry) continue;
    if (size === null) {
      // Sizeless: only valid if the catalog entry has no sizes either.
      if (entry.sizes.length > 0) continue;
    } else {
      if (!entry.sizes.includes(size)) continue;
    }
    rows.push({
      tag,
      modelName,
      size,
      type: entry.type,
      description: entry.description,
      capabilities: entry.capabilities,
    });
  }
  return rows;
}
