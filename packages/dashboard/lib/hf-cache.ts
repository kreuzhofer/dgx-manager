export type SortKey = "size" | "lastDeployed";

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

/** size: biggest first (what's eating the disk). lastDeployed: most stale
 *  first — never-deployed repos lead, then oldest lastDeployedAt; ties break
 *  by size so the biggest reclaim is always on top. */
export function sortRepos(repos: CacheRepo[], key: SortKey): CacheRepo[] {
  const copy = [...repos];
  if (key === "size") return copy.sort((a, b) => b.sizeBytes - a.sizeBytes);
  return copy.sort((a, b) => {
    if (a.lastDeployedAt === b.lastDeployedAt) return b.sizeBytes - a.sizeBytes;
    if (a.lastDeployedAt === null) return -1;
    if (b.lastDeployedAt === null) return 1;
    return a.lastDeployedAt < b.lastDeployedAt ? -1 : 1;
  });
}
