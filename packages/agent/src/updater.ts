import { readFileSync, existsSync, mkdirSync, writeFileSync, statSync } from "node:fs";
import { execSync, execFileSync } from "node:child_process";
import { setTimeout as sleepP } from "node:timers/promises";
import { basename } from "node:path";

/** True when this process was spawned as the copied detached updater (/tmp/dgx-updater-*.js). */
export function isUpdaterEntrypoint(argv1: string | undefined): boolean {
  return !!argv1 && basename(argv1).startsWith("dgx-updater-");
}

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

/** Move current -> old, then new -> current. If installing the new dir fails,
 *  restore the previous agent from old so the node is never left without one. */
export function atomicSwap(run: (cmd: string) => void): void {
  run("sudo rm -rf /opt/dgx-agent-old");
  run("sudo mv /opt/dgx-agent /opt/dgx-agent-old");        // current moved away
  try {
    run("sudo mv /opt/dgx-agent-new /opt/dgx-agent");      // install new
  } catch (e) {
    let restored = false;
    try { run("sudo mv /opt/dgx-agent-old /opt/dgx-agent"); restored = true; } catch { /* double-failure: leave for manual recovery */ }
    throw new Error(restored
      ? `swap failed, restored previous agent: ${(e as Error).message}`
      : `swap failed AND restore failed — /opt/dgx-agent may be missing: ${(e as Error).message}`);
  }
}

/** Restore the previous agent: stash the bad current as -failed, move old back, restart. */
export function atomicRollback(run: (cmd: string) => void): void {
  run("sudo rm -rf /opt/dgx-agent-failed");
  run("sudo mv /opt/dgx-agent /opt/dgx-agent-failed");     // stash the bad new (keep for post-mortem)
  run("sudo mv /opt/dgx-agent-old /opt/dgx-agent");        // restore the good previous agent
  run("sudo systemctl restart dgx-agent");
}

export interface UpdateDeps {
  download(url: string, dest: string): Promise<void>;
  extract(tarball: string, destDir: string): Promise<void>;
  verify(dir: string, version: string): VerifyResult;
  preserveNodeId(): void;
  swap(): void;
  restart(): void;
  checkConnected(): number | null;
  rollback(): void;
  writeResult(r: { version: string; outcome: string; error?: string }): void;
  log(m: string): void;
  now(): number;
  sleep(ms: number): Promise<void>;
}

const NEW_DIR = "/opt/dgx-agent-new";
const HEALTH_WINDOW_MS = 90_000;
const POLL_MS = 3_000;

export async function runUpdate(args: { bundleUrl: string; version: string }, deps: UpdateDeps): Promise<void> {
  const { bundleUrl, version } = args;
  const tarball = `/tmp/agent-bundle-${version}.tar.gz`;
  let swapped = false;
  try {
    deps.log(`[updater] starting update to v${version}`);
    await deps.download(bundleUrl, tarball);
    await deps.extract(tarball, NEW_DIR);
    const v = deps.verify(NEW_DIR, version);
    if (!v.ok) { deps.log(`[updater] verify failed: ${v.reason}`); deps.writeResult({ version, outcome: "failed", error: `verify: ${v.reason}` }); return; }
    deps.preserveNodeId();
    deps.swap(); swapped = true;
    deps.restart();
    const restartMs = deps.now();
    while (deps.now() - restartMs < HEALTH_WINDOW_MS) {
      await deps.sleep(POLL_MS);
      if (healthCheckPasses(deps.checkConnected(), restartMs, HEALTH_WINDOW_MS)) {
        deps.log("[updater] new agent reconnected — update ok");
        deps.writeResult({ version, outcome: "success" });
        return;
      }
    }
    deps.log("[updater] new agent did not reconnect in 90s — rolling back");
    try {
      deps.rollback();
      deps.writeResult({ version, outcome: "rolled-back", error: "new agent did not reconnect within 90s" });
    } catch (rbErr) {
      deps.log(`[updater] ROLLBACK FAILED — node may be without an agent: ${(rbErr as Error).message}`);
      deps.writeResult({ version, outcome: "rollback-failed", error: `rollback failed after health timeout: ${(rbErr as Error).message}` });
    }
  } catch (e) {
    const err = (e as Error).message;
    if (swapped) {
      try {
        deps.rollback();
        deps.log(`[updater] post-swap failure, rolled back: ${err}`);
        deps.writeResult({ version, outcome: "rolled-back", error: err });
      } catch (rbErr) {
        deps.log(`[updater] post-swap failure AND rollback failed — node may be without an agent: ${(rbErr as Error).message}`);
        deps.writeResult({ version, outcome: "rollback-failed", error: `${err}; rollback also failed: ${(rbErr as Error).message}` });
      }
    } else {
      deps.log(`[updater] pre-swap failure (agent untouched): ${err}`); deps.writeResult({ version, outcome: "failed", error: err });
    }
  }
}

const RUN_DIR = "/run/dgx-agent";
const MARKER = `${RUN_DIR}/connected`;
const RESULT = `${RUN_DIR}/update-result.json`;
const LOCK = `${RUN_DIR}/updating`;

function realDeps(nodeIdFile: string): UpdateDeps {
  const run = (cmd: string) => { execSync(cmd, { timeout: 20_000 }); };
  return {
    download: async (url, dest) => { execFileSync("curl", ["-sfL", "-o", dest, url], { timeout: 600_000 }); },
    extract: async (tarball, destDir) => {
      execSync(`sudo rm -rf "${destDir}" && sudo mkdir -p "${destDir}"`, { timeout: 15_000 });
      execFileSync("sudo", ["tar", "-xzf", tarball, "-C", `${destDir}/`], { timeout: 300_000 });
    },
    verify: (dir, version) => verifyExtractedBundle(dir, version),
    preserveNodeId: () => { if (existsSync(nodeIdFile)) execSync(`sudo cp "${nodeIdFile}" ${NEW_DIR}/node-id`, { timeout: 5_000 }); },
    swap: () => atomicSwap(run),
    restart: () => { execSync("sudo systemctl restart dgx-agent", { timeout: 15_000 }); },
    checkConnected: () => { try { return statSync(MARKER).mtimeMs; } catch { return null; } },
    rollback: () => atomicRollback(run),
    writeResult: (r) => { try { mkdirSync(RUN_DIR, { recursive: true }); writeFileSync(RESULT, JSON.stringify(r)); } catch { /* */ } },
    log: (m) => { try { execSync(`logger -t dgx-updater ${JSON.stringify(m)}`); } catch { /* */ } console.log(m); },
    now: () => Date.now(),
    sleep: (ms) => sleepP(ms),
  };
}

// Entrypoint: `node <copied-updater> <bundleUrl> <version> <nodeIdFile>`
// The handler copies this module to /tmp/dgx-updater-<ts>.js before spawning it
// detached (so it survives the parent agent process being restarted); the guard
// must match THAT name, not the source `updater.js` filename.
if (isUpdaterEntrypoint(process.argv[1])) {
  const [, , bundleUrl, version, nodeIdFile] = process.argv;
  const releaseLock = () => { try { execSync(`rm -f ${LOCK}`); } catch { /* */ } };
  runUpdate({ bundleUrl, version }, realDeps(nodeIdFile || "/opt/dgx-agent/node-id"))
    .finally(releaseLock);
}
