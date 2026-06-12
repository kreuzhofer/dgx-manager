import { resolve, sep } from "node:path";

/** Resolve a user-supplied recipe path against SHARED_STORAGE, rejecting escape. Fail-fast. */
export function resolveRecipePath(rel: string, root: string): string {
  const base = resolve(root);
  const full = resolve(base, rel);
  if (full !== base && !full.startsWith(base + sep)) {
    throw new Error(`recipePath resolves outside shared storage: ${rel}`);
  }
  return full;
}
