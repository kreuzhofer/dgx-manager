import { readFileSync } from "node:fs";

export interface VerifyResult { ok: boolean; reason?: string; }

/** Verify an extracted bundle dir: package.json parses and version matches. */
export function verifyExtractedBundle(
  dir: string, version: string, readPkg: (p: string) => string = (p) => readFileSync(p, "utf-8"),
): VerifyResult {
  let pkg: { version?: string };
  try { pkg = JSON.parse(readPkg(`${dir}/package.json`)); }
  catch (e) { return { ok: false, reason: `package.json unreadable: ${(e as Error).message}` }; }
  if (pkg.version !== version) return { ok: false, reason: `version mismatch: got ${pkg.version}, want ${version}` };
  return { ok: true };
}

/** New agent is healthy iff it wrote the connected marker AFTER the restart. */
export function healthCheckPasses(markerMtimeMs: number | null, restartMs: number, _windowMs: number): boolean {
  return markerMtimeMs != null && markerMtimeMs >= restartMs;
}
