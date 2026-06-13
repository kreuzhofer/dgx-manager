"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { apiFetch } from "@/lib/api";
import { useSSE, type SseEvent } from "@/lib/sse";
import {
  formatBytes, sortRepos,
  type CacheGroup, type CacheRepo, type SortColumn, type SortDir,
} from "@/lib/hf-cache";

function fmtDate(iso: string | null): string {
  return iso ? new Date(iso).toLocaleString() : "—";
}

/** A clickable, sortable column header. Shows ▲/▼ on the active column and
 *  marks it with aria-sort for assistive tech. */
function Th({ label, col, align, sortCol, sortDir, onSort }: {
  label: string;
  col: SortColumn;
  align?: "right";
  sortCol: SortColumn;
  sortDir: SortDir;
  onSort: (col: SortColumn) => void;
}) {
  const active = sortCol === col;
  return (
    <th
      className={`px-2 py-1 font-medium ${align === "right" ? "text-right" : ""}`}
      aria-sort={active ? (sortDir === "asc" ? "ascending" : "descending") : "none"}
    >
      <button
        type="button"
        onClick={() => onSort(col)}
        className={`inline-flex items-center gap-1 font-medium hover:text-gray-200 ${active ? "text-gray-200" : "text-gray-400"}`}
      >
        {label}
        <span className="text-xs w-2">{active ? (sortDir === "asc" ? "▲" : "▼") : ""}</span>
      </button>
    </th>
  );
}

// Direction a column gets the first time it's selected: text reads naturally
// ascending; numeric/date columns lead with the "most" (biggest, newest).
const DEFAULT_DIR: Record<SortColumn, SortDir> = {
  repoId: "asc", kind: "asc", size: "desc", revisions: "desc",
  downloaded: "desc", lastDeployed: "desc",
};

export default function ModelsPage() {
  const [caches, setCaches] = useState<CacheGroup[] | null>(null);
  // Default view: most recently downloaded first.
  const [sortCol, setSortCol] = useState<SortColumn>("downloaded");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [scanning, setScanning] = useState(false);
  const autoScanned = useRef(false);

  // Click a header: toggle direction if it's the active column, else switch to
  // it at that column's natural default direction.
  function toggleSort(col: SortColumn) {
    if (col === sortCol) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortCol(col);
      setSortDir(DEFAULT_DIR[col]);
    }
  }

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
        <button
          onClick={() => rescan()}
          disabled={scanning}
          className="px-3 py-1.5 rounded bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-sm font-medium"
        >
          {scanning ? "Scanning…" : "Rescan"}
        </button>
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
                    <Th label="Repo" col="repoId" sortCol={sortCol} sortDir={sortDir} onSort={toggleSort} />
                    <Th label="Kind" col="kind" sortCol={sortCol} sortDir={sortDir} onSort={toggleSort} />
                    <Th label="Size" col="size" align="right" sortCol={sortCol} sortDir={sortDir} onSort={toggleSort} />
                    <Th label="Revisions" col="revisions" align="right" sortCol={sortCol} sortDir={sortDir} onSort={toggleSort} />
                    <Th label="Downloaded" col="downloaded" sortCol={sortCol} sortDir={sortDir} onSort={toggleSort} />
                    <Th label="Last deployed" col="lastDeployed" sortCol={sortCol} sortDir={sortDir} onSort={toggleSort} />
                    <th className="px-2 py-1 font-medium" />
                  </tr>
                </thead>
                <tbody>
                  {sortRepos(cache.repos, sortCol, sortDir).map((repo) => (
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
