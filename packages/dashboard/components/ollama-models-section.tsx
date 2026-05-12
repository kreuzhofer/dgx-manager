"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { apiFetch } from "@/lib/api";
import { useDebouncedCallback } from "@/lib/use-debounced-callback";

interface CatalogEntry {
  name: string;
  description: string;
  type: "chat" | "embedding";
  sizes: string[];
  capabilities: string[];
}

interface CatalogResponse {
  entries: CatalogEntry[];
  fetchedAt: string | null;
}

type SaveState = "idle" | "saving" | "saved" | "error";

interface Row {
  tag: string;          // "llama3.1:8b" or "nomic-embed-text"
  modelName: string;
  size: string | null;
  description: string;
  type: "chat" | "embedding";
  capabilities: string[];
}

/** Flatten catalog entries into one row per deployable tag, alphabetised by tag. */
function flatten(entries: CatalogEntry[]): Row[] {
  const rows: Row[] = [];
  for (const e of entries) {
    if (e.sizes.length === 0) {
      rows.push({ tag: e.name, modelName: e.name, size: null, description: e.description, type: e.type, capabilities: e.capabilities });
    } else {
      for (const size of e.sizes) {
        rows.push({ tag: `${e.name}:${size}`, modelName: e.name, size, description: e.description, type: e.type, capabilities: e.capabilities });
      }
    }
  }
  return rows.sort((a, b) => a.tag.localeCompare(b.tag));
}

/**
 * Token-based fuzzy match: every whitespace-separated token in `query` must
 * appear as a substring of the haystack (tag + description + capabilities),
 * order-independent. So "qwen emb" matches "qwen3-embedding:8b" because both
 * "qwen" and "emb" are substrings of the row's combined text. Empty query
 * matches everything.
 */
function fuzzyMatch(row: Row, query: string): boolean {
  const tokens = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return true;
  const haystack = `${row.tag} ${row.description} ${row.capabilities.join(" ")}`.toLowerCase();
  return tokens.every((t) => haystack.includes(t));
}

export function OllamaModelsSection() {
  const [catalog, setCatalog] = useState<CatalogEntry[]>([]);
  const [fetchedAt, setFetchedAt] = useState<string | null>(null);
  const [enabled, setEnabled] = useState<Set<string>>(new Set());
  const [refreshing, setRefreshing] = useState(false);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [filter, setFilter] = useState("");

  const rows = useMemo(() => flatten(catalog), [catalog]);
  const visibleRows = useMemo(
    () => rows.filter((r) => fuzzyMatch(r, filter)),
    [rows, filter],
  );

  const load = useCallback(async () => {
    try {
      const [cat, en] = await Promise.all([
        apiFetch<CatalogResponse>("/api/ollama-catalog/catalog"),
        apiFetch<{ enabled: string[] }>("/api/ollama-catalog/enabled"),
      ]);
      setCatalog(cat.entries);
      setFetchedAt(cat.fetchedAt);
      setEnabled(new Set(en.enabled));
    } catch (err) {
      setRefreshError(String(err));
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const persistEnabled = useCallback(async (tags: string[]) => {
    setSaveState("saving");
    try {
      await apiFetch("/api/ollama-catalog/enabled", {
        method: "PUT",
        body: JSON.stringify({ enabled: tags }),
      });
      setSaveState("saved");
      setTimeout(() => setSaveState((s) => (s === "saved" ? "idle" : s)), 1500);
    } catch {
      setSaveState("error");
    }
  }, []);

  const debouncedPersist = useDebouncedCallback(persistEnabled, 400);

  const toggle = (tag: string) => {
    setEnabled((prev) => {
      const next = new Set(prev);
      if (next.has(tag)) next.delete(tag);
      else next.add(tag);
      debouncedPersist(Array.from(next));
      return next;
    });
  };

  const refresh = async () => {
    setRefreshing(true);
    setRefreshError(null);
    try {
      const res = await apiFetch<CatalogResponse>("/api/ollama-catalog/catalog/refresh", {
        method: "POST",
      });
      setCatalog(res.entries);
      setFetchedAt(res.fetchedAt);
    } catch (err) {
      setRefreshError(String(err));
    } finally {
      setRefreshing(false);
    }
  };

  const saveLabel: Record<SaveState, string> = {
    idle: "",
    saving: "Saving…",
    saved: "Saved",
    error: "Save failed",
  };

  return (
    <section className="bg-gray-900 rounded-lg p-6 border border-gray-800">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-lg font-semibold">Ollama Models</h2>
        <div className="flex items-center gap-3">
          <span className={`text-xs ${saveState === "error" ? "text-red-400" : "text-gray-500"}`}>
            {saveLabel[saveState]}
          </span>
          <button
            type="button"
            onClick={refresh}
            disabled={refreshing}
            className="px-3 py-1 text-xs rounded bg-green-700/30 border border-green-700/60 text-green-300 hover:bg-green-700/50 disabled:opacity-50"
          >
            {refreshing ? "Refreshing…" : "Refresh catalog"}
          </button>
        </div>
      </div>

      <p className="text-xs text-gray-500 mb-3">
        Pulled from <code>ollama.com/library</code> (cloud-only models excluded).{" "}
        {fetchedAt
          ? `Last refreshed ${new Date(fetchedAt).toLocaleString()}.`
          : "Never refreshed."}{" "}
        Check the model:tag combinations you want available on the Deployments page.
      </p>

      {refreshError && (
        <p className="text-xs text-red-400 mb-3">Refresh failed: {refreshError}</p>
      )}

      {catalog.length === 0 ? (
        <p className="text-sm text-gray-600 py-4">
          Catalog is empty. Click <strong>Refresh catalog</strong> to pull the latest from ollama.com.
        </p>
      ) : (
        <>
          <input
            type="text"
            placeholder="Filter by name or description"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="w-full mb-2 bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-xs focus:outline-none focus:border-green-500"
          />
          <ul className="max-h-96 overflow-y-auto divide-y divide-gray-800/60 border border-gray-800 rounded">
            {visibleRows.map((r) => {
              const checked = enabled.has(r.tag);
              return (
                <li key={r.tag} className="px-3 py-2 flex items-center gap-3 hover:bg-gray-800/30">
                  <input
                    type="checkbox"
                    className="accent-green-500"
                    checked={checked}
                    onChange={() => toggle(r.tag)}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-gray-200 truncate">
                      {r.tag}{" "}
                      <span className="text-[10px] uppercase tracking-wide text-gray-500 ml-1">
                        {r.type}
                      </span>
                      {r.capabilities.length > 0 && (
                        <span className="text-[10px] text-indigo-400 ml-2">
                          {r.capabilities.join(" · ")}
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-gray-500 truncate">{r.description}</div>
                  </div>
                </li>
              );
            })}
            {visibleRows.length === 0 && (
              <li className="px-3 py-4 text-center text-gray-600 text-sm">No matches.</li>
            )}
          </ul>
        </>
      )}
    </section>
  );
}
