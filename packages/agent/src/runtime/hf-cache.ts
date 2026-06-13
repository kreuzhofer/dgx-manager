import {
  existsSync, lstatSync, readdirSync, readFileSync, rmSync, statfsSync, writeFileSync,
} from "node:fs";
import { join, resolve, sep } from "node:path";
import { randomUUID } from "node:crypto";

export type RepoKind = "model" | "dataset";

export interface CachedRepo {
  repoId: string;
  kind: RepoKind;
  sizeBytes: number;
  nFiles: number;
  revisions: number;     // snapshot dir count
  lastModified: string;  // ISO — max file mtime = download/update time
}

export interface HfCacheInventoryPayload {
  cacheId: string;
  hfHome: string;
  scannedAt: string;
  totalBytes: number;
  diskFreeBytes: number;
  error?: string;
  repos: CachedRepo[];
}

const KIND_BY_PREFIX: Record<string, RepoKind> = { models: "model", datasets: "dataset" };

/** Decode an HF cache dir name (`models--org--name`) into kind + repoId.
 *  Mirrors huggingface_hub: split on `--`, first part is the kind, the rest
 *  joined with `/`. Returns null for hub sidecars (`.locks`, `version.txt`)
 *  and unsupported kinds (`spaces--…`). */
export function parseRepoDirName(dirName: string): { kind: RepoKind; repoId: string } | null {
  const parts = dirName.split("--");
  if (parts.length < 2) return null;
  const kind = KIND_BY_PREFIX[parts[0]];
  if (!kind) return null;
  const segments = parts.slice(1);
  if (segments.some((s) => s.length === 0)) return null;
  return { kind, repoId: segments.join("/") };
}

/** Encode a repoId into an HF cache dir name: org/name → {kind}s--org--name. Inverse of parseRepoDirName. */
export function repoDirName(kind: RepoKind, repoId: string): string {
  return `${kind}s--${repoId.split("/").join("--")}`;
}

const REPO_SEGMENT_RE = /^[A-Za-z0-9._-]+$/;

/** A repoId is safe to turn into a deletable path iff it is 1–2 segments of
 *  HF-legal characters, with `.`/`..` explicitly excluded (the char class
 *  alone would admit them). Org-less legacy ids ("gpt2") are 1 segment. */
export function isSafeRepoId(repoId: string): boolean {
  const segments = repoId.split("/");
  if (segments.length > 2) return false;
  return segments.every((s) => REPO_SEGMENT_RE.test(s) && s !== "." && s !== "..");
}

/** SECURITY BOUNDARY: the only code that turns a wire-supplied repoId into a
 *  filesystem path. Validates the id, then verifies the resolved target is a
 *  strict descendant of hfHome/hub before rm -rf. */
export function deleteCachedRepo(hfHome: string, kind: RepoKind, repoId: string): void {
  if (!isSafeRepoId(repoId)) throw new Error(`invalid repoId: ${JSON.stringify(repoId)}`);
  const hub = resolve(hfHome, "hub");
  const target = resolve(hub, repoDirName(kind, repoId));
  if (!target.startsWith(hub + sep)) {
    throw new Error(`refusing to delete outside hub/: ${JSON.stringify(repoId)}`);
  }
  if (!existsSync(target)) throw new Error(`not in cache: ${repoId} (${kind})`);
  // force: true so a concurrent HF download mutating the tree can't turn a
  // valid delete into a spurious ENOENT on a live node.
  rmSync(target, { recursive: true, force: true });
}

interface WalkStats { sizeBytes: number; nFiles: number; lastModifiedMs: number }

/** Recursive size walk using lstat: snapshot symlinks count as their own
 *  ~0-byte entries, so blob bytes are never counted twice. */
function walk(dir: string): WalkStats {
  const acc: WalkStats = { sizeBytes: 0, nFiles: 0, lastModifiedMs: 0 };
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) {
      const sub = walk(p);
      acc.sizeBytes += sub.sizeBytes;
      acc.nFiles += sub.nFiles;
      acc.lastModifiedMs = Math.max(acc.lastModifiedMs, sub.lastModifiedMs);
    } else {
      const st = lstatSync(p);
      acc.sizeBytes += st.size;
      acc.nFiles += 1;
      acc.lastModifiedMs = Math.max(acc.lastModifiedMs, st.mtimeMs);
    }
  }
  return acc;
}

export function scanHfCache(hfHome: string): CachedRepo[] {
  if (!existsSync(hfHome)) throw new Error(`HF_HOME does not exist: ${hfHome}`);
  const hub = join(hfHome, "hub");
  if (!existsSync(hub)) return []; // fresh cache — nothing downloaded yet
  const repos: CachedRepo[] = [];
  for (const entry of readdirSync(hub, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const parsed = parseRepoDirName(entry.name);
    if (!parsed) continue; // .locks and other sidecars
    const repoDir = join(hub, entry.name);
    const stats = walk(repoDir);
    const snapshotsDir = join(repoDir, "snapshots");
    const revisions = existsSync(snapshotsDir)
      ? readdirSync(snapshotsDir, { withFileTypes: true }).filter((e) => e.isDirectory()).length
      : 0;
    const lastModifiedMs = stats.lastModifiedMs || lstatSync(repoDir).mtimeMs;
    repos.push({
      ...parsed,
      sizeBytes: stats.sizeBytes,
      nFiles: stats.nFiles,
      revisions,
      lastModified: new Date(lastModifiedMs).toISOString(),
    });
  }
  return repos;
}

/** Cache identity marker. Nodes sharing the cache over NFS read the same
 *  UUID, so the server can group their inventories into one cache group.
 *  `wx` write loses the create race gracefully on shared storage. */
export function readOrCreateCacheId(hfHome: string): string {
  if (!existsSync(hfHome)) throw new Error(`HF_HOME does not exist: ${hfHome}`);
  const marker = join(hfHome, ".dgx-cache-id");
  if (existsSync(marker)) {
    const existing = readFileSync(marker, "utf8").trim();
    if (existing) return existing;
  }
  const id = randomUUID();
  try {
    writeFileSync(marker, `${id}\n`, { flag: "wx" });
    return id;
  } catch {
    const existing = readFileSync(marker, "utf8").trim();
    if (existing) return existing;
    throw new Error(`cache-id marker exists but is unreadable/empty: ${marker}`);
  }
}

export function buildInventory(hfHome: string): HfCacheInventoryPayload {
  const cacheId = readOrCreateCacheId(hfHome);
  const repos = scanHfCache(hfHome);
  const fsStat = statfsSync(hfHome);
  return {
    cacheId,
    hfHome,
    scannedAt: new Date().toISOString(),
    totalBytes: repos.reduce((sum, r) => sum + r.sizeBytes, 0),
    diskFreeBytes: fsStat.bavail * fsStat.bsize,
    repos,
  };
}
