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
  rmSync(target, { recursive: true });
}
