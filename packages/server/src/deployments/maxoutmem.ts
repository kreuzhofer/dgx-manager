/**
 * Declarative `maxoutmem` recipe flag support.
 *
 * A recipe may carry a top-level `maxoutmem: true` key. Plain `sparkrun run`
 * ignores it (unknown top-level key, tolerated by sparkrun's schema), but
 * dgx-manager reads it and, before launching a vLLM deploy, reclaims unified
 * memory on the target nodes (currently: stop the GNOME desktop `gdm` which
 * holds ~2 GiB of unified memory on these headless rack nodes, plus a
 * secondary page-cache drop). That headroom is what separates "128K KV cache
 * fits" from an OOM at KV-init on GLM-class models.
 *
 * The flag lives in the raw recipe YAML in the head node's sparkrun run-cache,
 * NOT in the (lossy) agent recipe catalog, so the server reads it over SSH.
 *
 * Split into pure builders (unit-tested) + one effectful orchestrator
 * ({@link maxOutMemoryForDeploy}) that is resilient by construction: it never
 * throws and never blocks the deploy — any failure degrades to `applied:false`
 * and the deploy proceeds normally.
 */

/** Registry + recipe basename parsed from a `@<registry>/<basename>` ref. */
export type RecipeRefParts = { registry: string; basename: string };

/** Safe token: no path separators, no shell metachars, no traversal. */
const SAFE_TOKEN = /^[A-Za-z0-9._-]+$/;

/**
 * Parse a recipe ref of the form `@<registry>/<basename>` into its parts.
 *
 * `@community-kreuzhofer/glm-5.2-awq-15pct-vllm-kreuzhofer`
 *   → { registry: "community-kreuzhofer", basename: "glm-5.2-awq-15pct-vllm-kreuzhofer" }
 *
 * Both parts must match {@link SAFE_TOKEN} (guards against path traversal and
 * shell injection). Throws a clear Error on a malformed or unsafe ref.
 */
export function parseRecipeRef(ref: string): RecipeRefParts {
  if (typeof ref !== "string" || !ref.startsWith("@")) {
    throw new Error(`Malformed recipe ref (expected '@<registry>/<basename>'): ${ref}`);
  }
  const rest = ref.slice(1);
  const slash = rest.indexOf("/");
  if (slash < 0) {
    throw new Error(`Malformed recipe ref (missing '/'): ${ref}`);
  }
  const registry = rest.slice(0, slash);
  const basename = rest.slice(slash + 1);
  if (!SAFE_TOKEN.test(registry)) {
    throw new Error(`Unsafe recipe registry in ref: ${ref}`);
  }
  if (!SAFE_TOKEN.test(basename)) {
    throw new Error(`Unsafe recipe basename in ref: ${ref}`);
  }
  return { registry, basename };
}

/**
 * Pure: build a shell command that, run on the head node, prints exactly
 * `true` or `false` depending on whether the recipe YAML declares
 * `maxoutmem: true`. Interpolates ONLY the validated registry/basename.
 */
export function readMaxOutMemCmd(ref: string): string {
  const { registry, basename } = parseRecipeRef(ref);
  const dir = `$HOME/.cache/sparkrun/registries/${registry}/recipes`;
  return (
    `f=$(find "${dir}" -name '${basename}.yaml' 2>/dev/null | head -1); ` +
    `if [ -n "$f" ] && grep -qiE '^[[:space:]]*maxoutmem[[:space:]]*:[[:space:]]*true([[:space:]]|$)' "$f"; ` +
    `then echo true; else echo false; fi`
  );
}

/** Pure: interpret the `readMaxOutMemCmd` stdout as a boolean. */
export function parseMaxOutMem(stdout: string): boolean {
  return stdout.trim() === "true";
}

/** Matches a `maxoutmem: true` line. Mirrors the grep in {@link readMaxOutMemCmd}. */
const MAXOUTMEM_LINE = /^[ \t]*maxoutmem[ \t]*:[ \t]*true[ \t]*$/im;

/**
 * Pure: read the `maxoutmem` flag straight from recipe YAML the manager already
 * holds — dgxrun (`@dgxrun/…`), inline `recipeYaml`, or a `recipePath` on shared
 * storage.
 *
 * `readMaxOutMemCmd` can only find recipes in the head node's *sparkrun registry
 * cache*. dgxrun recipes live in this repo's `recipes/dgxrun/`, never in that
 * cache, so the probe's `find` matched nothing and printed `false` — the
 * `maxoutmem: true` in our GLM recipes was silently ignored, and the unified
 * memory it was meant to reclaim had to be freed by hand (2026-07-10).
 */
export function parseMaxOutMemYaml(text: string): boolean {
  return MAXOUTMEM_LINE.test(text);
}

/**
 * Pure: best-effort shell to free memory on ONE node. Must never hang or
 * hard-fail — `sudo -n` avoids a password prompt, every step is `|| true`,
 * and no `set -e`. Stopping gdm frees the held unified memory; drop_caches is
 * a secondary free. Future reclamations get appended here.
 *
 * Measures MemAvailable before/after and reports the gdm state so the caller
 * can tell "freed 2 GiB" from "sudo denied / nothing freed" — the reclaim's
 * exit is best-effort, so observability lives in the printed line, not $?.
 * Output shape: `reclaimed freed_kib=<N> gdm=<inactive|active|unknown>`.
 */
export function reclaimMemoryCmd(): string {
  return (
    `before=$(awk '/MemAvailable/{print $2}' /proc/meminfo); ` +
    `sudo -n systemctl stop gdm 2>/dev/null || true; ` +
    `sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true; ` +
    `after=$(awk '/MemAvailable/{print $2}' /proc/meminfo); ` +
    `echo "reclaimed freed_kib=$((after-before)) gdm=$(systemctl is-active gdm 2>/dev/null || echo unknown)"`
  );
}

/**
 * Pure: parse the `reclaimMemoryCmd` stdout into a human `detail` string like
 * `"freed 2016 MiB, gdm=inactive"`. Falls back to the raw (trimmed) stdout
 * when the expected `freed_kib=`/`gdm=` tokens aren't present (e.g. an ssh
 * error message captured as detail).
 */
export function parseReclaimDetail(stdout: string): string {
  const kibMatch = stdout.match(/freed_kib=(-?\d+)/);
  const gdmMatch = stdout.match(/gdm=(\S+)/);
  if (!kibMatch && !gdmMatch) return stdout.trim();
  const parts: string[] = [];
  if (kibMatch) {
    const mib = Math.round(parseInt(kibMatch[1], 10) / 1024);
    parts.push(`freed ${mib} MiB`);
  }
  if (gdmMatch) parts.push(`gdm=${gdmMatch[1]}`);
  return parts.join(", ");
}

type SshExecResult = { code: number; stdout: string; stderr: string };

export type MaxOutMemoryOpts = {
  recipeRef: string;
  headIp: string;
  nodeIps: string[];
  sshExec: (
    host: string,
    command: string,
    options?: { timeout?: number },
  ) => Promise<SshExecResult>;
  log?: (msg: string) => void;
  /**
   * Pre-resolved `maxoutmem` flag. Pass it whenever the manager can read the
   * recipe YAML itself (see {@link parseMaxOutMemYaml}) — the SSH probe is then
   * skipped entirely, which is the only way the flag works for dgxrun recipes.
   * Omit to fall back to reading it from the head node's sparkrun cache.
   */
  enabled?: boolean;
};

export type MaxOutMemoryResult = {
  applied: boolean;
  perNode: Array<{ ip: string; ok: boolean; detail: string }>;
};

/**
 * Effectful orchestrator. Reads the `maxoutmem` flag from the head node; if
 * set, reclaims memory on every target node over SSH before the deploy.
 *
 * Resilient by construction: NEVER throws. Any error (ssh throw, bad ref) in
 * the read step degrades to `{ applied:false, perNode:[] }`. A per-node
 * reclaim failure is captured (`ok:false`) but does not abort the others or
 * throw. The deploy always proceeds regardless of outcome.
 */
export async function maxOutMemoryForDeploy(
  opts: MaxOutMemoryOpts,
): Promise<MaxOutMemoryResult> {
  const { recipeRef, headIp, nodeIps, sshExec } = opts;
  const log = opts.log ?? (() => {});

  // Prefer the caller's pre-resolved flag (manager read the YAML itself). Only
  // fall back to the head node's sparkrun recipe cache when we have nothing —
  // that probe cannot see dgxrun recipes. The head node is reachable during a
  // real deploy, so this is fast; the short cap just bounds the worst case so a
  // flaky node can't stall the deploy.
  let enabled: boolean;
  if (opts.enabled !== undefined) {
    enabled = opts.enabled;
  } else {
    try {
      const read = await sshExec(headIp, readMaxOutMemCmd(recipeRef), { timeout: 8_000 });
      enabled = parseMaxOutMem(read.stdout);
    } catch (e) {
      log(`[maxoutmem] flag read failed (${(e as Error).message}); skipping reclaim`);
      return { applied: false, perNode: [] };
    }
  }

  if (!enabled) {
    // Observable, per Principle 3: a recipe that meant to reclaim memory and
    // didn't must say so, rather than look identical to one that never asked.
    log(`[maxoutmem] ${recipeRef} does not request maxoutmem; skipping reclaim`);
    return { applied: false, perNode: [] };
  }

  const perNode: MaxOutMemoryResult["perNode"] = [];
  for (const ip of nodeIps) {
    try {
      const r = await sshExec(ip, reclaimMemoryCmd(), { timeout: 20_000 });
      const detail = parseReclaimDetail(r.stdout || r.stderr || "");
      perNode.push({ ip, ok: r.code === 0, detail });
      log(`[maxoutmem] node ${ip}: ${detail}`);
    } catch (e) {
      const detail = (e as Error).message;
      perNode.push({ ip, ok: false, detail });
      log(`[maxoutmem] node ${ip}: reclaim failed (${detail})`);
    }
  }

  const freed = perNode.filter((n) => n.ok).length;
  log(`[maxoutmem] freed memory on ${freed}/${nodeIps.length} nodes before deploy`);
  return { applied: true, perNode };
}
