import { parseCatalogHtml, type OllamaCatalogEntry } from "./catalog-parser.js";
import { writeCatalog } from "./catalog-store.js";

const LIBRARY_URL = process.env.OLLAMA_LIBRARY_URL || "https://ollama.com/library";
const FETCH_TIMEOUT_MS = 15_000;

export interface RefreshResult {
  entries: OllamaCatalogEntry[];
  fetchedAt: string;
}

/**
 * Fetch the public Ollama library page, parse it, and write the result into
 * the Setting table. Throws on network failure or non-2xx — the route handler
 * surfaces that to the user (refresh is explicit, so failure should be loud).
 *
 * If parsing succeeds but yields zero entries, we still persist the empty
 * list and the timestamp — that's a real catalog state (Ollama outage,
 * markup overhaul) and we want the UI to reflect it, not show stale data.
 */
export async function refreshCatalog(): Promise<RefreshResult> {
  const res = await fetch(LIBRARY_URL, {
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    headers: { "User-Agent": "dgx-manager/ollama-catalog" },
  });
  if (!res.ok) {
    throw new Error(`Ollama library fetch failed: HTTP ${res.status}`);
  }
  const html = await res.text();
  const entries = parseCatalogHtml(html);
  const fetchedAt = await writeCatalog(entries);
  return { entries, fetchedAt };
}
