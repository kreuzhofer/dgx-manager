"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { apiFetch } from "@/lib/api";
import { useSSE, type SseEvent } from "@/lib/sse";
import {
  formatBytes, sortRepos,
  type CacheGroup, type CacheRepo, type SortKey,
} from "@/lib/hf-cache";

function fmtDate(iso: string | null): string {
  return iso ? new Date(iso).toLocaleString() : "—";
}

export default function ModelsPage() {
  const [caches, setCaches] = useState<CacheGroup[] | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("size");
  const [scanning, setScanning] = useState(false);
  const autoScanned = useRef(false);

  const load = useCallback(async (): Promise<CacheGroup[] | null> => {
    try {
      const res = await apiFetch<{ caches: CacheGroup[] }>("/api/hf-cache");
      setCaches(res.caches);
      return res.caches;
    } catch (err) {
      toast.error("Failed to load cache inventory", { description: String(err) });
      return null;
    }
  }, []);

  const rescan = useCallback(async (silent = false) => {
    setScanning(true);
    try {
      const res = await apiFetch<{ requested: number }>("/api/hf-cache/scan", { method: "POST" });
      if (!silent) toast.info(`Scan requested on ${res.requested} agent(s)`);
    } catch (err) {
      // 503 (no agents) lands here — the empty state below explains the situation
      if (!silent) toast.error("Scan failed", { description: String(err) });
    } finally {
      setScanning(false);
    }
  }, []);

  useEffect(() => {
    (async () => {
      const initial = await load();
      // First visit: the server holds no inventory until agents have scanned.
      if (initial !== null && initial.length === 0 && !autoScanned.current) {
        autoScanned.current = true;
        rescan(true);
      }
    })();
  }, [load, rescan]);

  useSSE(
    useCallback((event: SseEvent) => {
      // Enrichment (inUse/lastDeployedAt) lives server-side — refetch rather
      // than patching the SSE payload in.
      if (event.type === "hf-cache:inventory") load();
    }, [load]),
    load,
  );

  async function deleteRepo(cache: CacheGroup, repo: CacheRepo) {
    const msg =
      `Delete ${repo.repoId} (${formatBytes(repo.sizeBytes)}) from ${cache.hfHome}?\n\n` +
      "The next deployment of this model will re-download it.";
    if (!confirm(msg)) return;
    try {
      await apiFetch(
        `/api/hf-cache/${encodeURIComponent(cache.cacheId)}` +
          `?repoId=${encodeURIComponent(repo.repoId)}&kind=${repo.kind}`,
        { method: "DELETE" },
      );
      toast.success(`Deleting ${repo.repoId}…`, {
        description: "The inventory will refresh when the agent has rescanned.",
      });
    } catch (err) {
      toast.error("Delete failed", { description: String(err) });
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Models</h1>
          <p className="text-gray-400 text-sm mt-1">
            Hugging Face download cache (HF_HOME) — inspect and clean up cached model weights.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <label className="text-sm text-gray-400">
            Sort{" "}
            <select
              value={sortKey}
              onChange={(e) => setSortKey(e.target.value as SortKey)}
              className="ml-1 bg-gray-900 border border-gray-700 rounded px-2 py-1 text-gray-200"
            >
              <option value="size">Largest first</option>
              <option value="lastDeployed">Stalest first</option>
            </select>
          </label>
          <button
            onClick={() => rescan()}
            disabled={scanning}
            className="px-3 py-1.5 rounded bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-sm font-medium"
          >
            {scanning ? "Scanning…" : "Rescan"}
          </button>
        </div>
      </div>

      {caches === null && <p className="text-gray-400">Loading…</p>}

      {caches !== null && caches.length === 0 && (
        <div className="border border-gray-800 rounded-lg p-8 text-center text-gray-400">
          No cache inventory yet. Agents report their HF cache when scanned — if nodes are
          connected, hit <span className="text-gray-200">Rescan</span>; otherwise connect an
          agent first.
        </div>
      )}

      {caches?.map((cache) => {
        const title =
          cache.nodes.length > 1
            ? `Shared cache — ${cache.nodes.map((n) => n.name).join(", ")}`
            : `Cache on ${cache.nodes[0]?.name ?? cache.cacheId}`;
        return (
          <section key={cache.cacheId} className="mb-8">
            <div className="flex items-baseline justify-between mb-2">
              <h2 className="text-lg font-semibold">{title}</h2>
              <span className="text-sm text-gray-400">
                {formatBytes(cache.totalBytes)} used · {formatBytes(cache.diskFreeBytes)} free ·{" "}
                <span className="font-mono">{cache.hfHome}</span> · scanned {fmtDate(cache.scannedAt)}
              </span>
            </div>

            {cache.error && (
              <div className="mb-3 px-3 py-2 rounded border border-red-800 bg-red-950 text-red-300 text-sm">
                {cache.error}
              </div>
            )}

            {cache.repos.length === 0 && !cache.error && (
              <p className="text-gray-500 text-sm">Cache is empty.</p>
            )}

            {cache.repos.length > 0 && (
              <table className="min-w-full text-sm">
                <thead className="text-left text-gray-400 border-b border-gray-800">
                  <tr>
                    <th className="px-2 py-1 font-medium">Repo</th>
                    <th className="px-2 py-1 font-medium">Kind</th>
                    <th className="px-2 py-1 font-medium text-right">Size</th>
                    <th className="px-2 py-1 font-medium text-right">Revisions</th>
                    <th className="px-2 py-1 font-medium">Downloaded</th>
                    <th className="px-2 py-1 font-medium">Last deployed</th>
                    <th className="px-2 py-1 font-medium" />
                  </tr>
                </thead>
                <tbody>
                  {sortRepos(cache.repos, sortKey).map((repo) => (
                    <tr key={`${repo.kind}:${repo.repoId}`} className="border-b border-gray-900">
                      <td className="px-2 py-1.5 font-mono">{repo.repoId}</td>
                      <td className="px-2 py-1.5 text-gray-400">{repo.kind}</td>
                      <td className="px-2 py-1.5 text-right">{formatBytes(repo.sizeBytes)}</td>
                      <td className="px-2 py-1.5 text-right text-gray-400">{repo.revisions}</td>
                      <td className="px-2 py-1.5 text-gray-400">{fmtDate(repo.lastModified)}</td>
                      <td className="px-2 py-1.5 text-gray-400">{fmtDate(repo.lastDeployedAt)}</td>
                      <td className="px-2 py-1.5 text-right">
                        {repo.inUse ? (
                          <span
                            className="inline-block px-2 py-0.5 rounded bg-amber-900 text-amber-300 text-xs"
                            title={`In use by: ${repo.inUseBy.join(", ")} — stop those deployments to delete`}
                          >
                            in use
                          </span>
                        ) : (
                          <button
                            onClick={() => deleteRepo(cache, repo)}
                            className="px-2 py-0.5 rounded bg-red-900 hover:bg-red-800 text-red-200 text-xs"
                          >
                            Delete
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        );
      })}
    </div>
  );
}
