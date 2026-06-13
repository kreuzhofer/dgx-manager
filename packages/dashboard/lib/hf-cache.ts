export type SortColumn = "repoId" | "kind" | "size" | "revisions" | "downloaded" | "lastDeployed";
export type SortDir = "asc" | "desc";

export interface CacheRepo {
  repoId: string;
  kind: "model" | "dataset";
  sizeBytes: number;
  nFiles: number;
  revisions: number;
  lastModified: string;
  lastDeployedAt: string | null;
  inUse: boolean;
  inUseBy: string[];
}

export interface CacheNode { nodeId: string; name: string; connected: boolean }

export interface CacheGroup {
  cacheId: string;
  nodes: CacheNode[];
  hfHome: string;
  scannedAt: string;
  totalBytes: number;
  diskFreeBytes: number;
  error?: string;
  repos: CacheRepo[];
}

export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return `${v >= 100 ? Math.round(v) : v.toFixed(1)} ${units[i]}`;
}

/** Ascending comparators per column. A null `lastDeployedAt` (never deployed)
 *  sorts as the smallest value, so ascending puts never-deployed first (the
 *  stalest-first cleanup view) and descending puts them last. ISO-8601 date
 *  strings compare correctly lexicographically. */
const ASC: Record<SortColumn, (a: CacheRepo, b: CacheRepo) => number> = {
  repoId: (a, b) => a.repoId.localeCompare(b.repoId),
  kind: (a, b) => a.kind.localeCompare(b.kind),
  size: (a, b) => a.sizeBytes - b.sizeBytes,
  revisions: (a, b) => a.revisions - b.revisions,
  downloaded: (a, b) => (a.lastModified < b.lastModified ? -1 : a.lastModified > b.lastModified ? 1 : 0),
  lastDeployed: (a, b) => {
    if (a.lastDeployedAt === b.lastDeployedAt) return 0;
    if (a.lastDeployedAt === null) return -1;
    if (b.lastDeployedAt === null) return 1;
    return a.lastDeployedAt < b.lastDeployedAt ? -1 : 1;
  },
};

/** Non-mutating sort of cache repos by a column and direction. Ties always
 *  break by repoId ascending so the order is deterministic regardless of input
 *  order or sort direction (a stable result the user can rely on). */
export function sortRepos(repos: CacheRepo[], column: SortColumn, dir: SortDir): CacheRepo[] {
  const cmp = ASC[column];
  const sign = dir === "asc" ? 1 : -1;
  return [...repos].sort((a, b) => {
    const primary = cmp(a, b);
    return primary !== 0 ? sign * primary : a.repoId.localeCompare(b.repoId);
  });
}
