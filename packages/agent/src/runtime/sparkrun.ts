import { spawn, execFileSync, spawnSync } from "node:child_process";
import type { ChildProcess } from "node:child_process";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { SPARKRUN_PKG } from "../recipes.js";
import { buildSparkrunArgs, type SparkrunLaunchOptions } from "./sparkrun-args.js";
import { parseClusterId } from "./sparkrun-parse.js";
import { saveDeployment, removeDeployment } from "./deployment-store.js";

export type Opts = Omit<SparkrunLaunchOptions, "recipeRef"> & { recipeName?: string };

const logFollowers = new Map<string, ChildProcess>();

export function launchSparkrun(
  deploymentId: string, recipeRef: string, opts: Opts,
  onLog: (line: string) => void, onExit: (code: number | null) => void,
): void {
  const argv = ["--from", SPARKRUN_PKG, "sparkrun", ...buildSparkrunArgs({ recipeRef, ...opts })];
  const child = spawn("uvx", argv, { detached: true });
  const hosts = opts.hosts;
  const tp = opts.tp ?? hosts.length;
  let buf = "";
  let clusterId: string | undefined;
  const persist = () => saveDeployment({
    deploymentId, recipeFile: recipeRef, recipeName: opts.recipeName ?? recipeRef,
    port: opts.port ?? 8000, startedAt: new Date().toISOString(),
    clusterNodes: hosts, clusterId, tp, kind: "sparkrun",
  });
  const onData = (b: Buffer) => {
    const s = b.toString();
    onLog(s);
    if (!clusterId) {
      buf += s;
      const id = parseClusterId(buf);
      if (id) {
        clusterId = id;
        persist();
        const follower = spawn(
          "uvx",
          ["--from", SPARKRUN_PKG, "sparkrun", "logs", clusterId, "-H", hosts.join(","), "--tp", String(tp), "--tail", "1000"],
          { detached: true },
        );
        follower.stdout?.on("data", (b: Buffer) => onLog(b.toString()));
        follower.stderr?.on("data", (b: Buffer) => onLog(b.toString()));
        follower.on("exit", () => { logFollowers.delete(deploymentId); });
        logFollowers.set(deploymentId, follower);
      }
    }
  };
  child.stdout?.on("data", onData);
  child.stderr?.on("data", onData);
  child.on("exit", (code) => onExit(code));
  persist();
}

export function stopSparkrun(deploymentId: string, target: string, hosts: string[], tp?: number): void {
  const f = logFollowers.get(deploymentId);
  if (f) { try { f.kill(); } catch { /* already gone */ } logFollowers.delete(deploymentId); }
  const args = ["--from", SPARKRUN_PKG, "sparkrun", "stop", target, "-H", hosts.join(",")];
  if (tp != null) args.push("--tp", String(tp));
  try { execFileSync("uvx", args, { timeout: 120_000 }); }
  finally { removeDeployment(deploymentId); }
}

/** Find the container name for a sparkrun cluster id (e.g. sparkrun_<hex>_solo). */
function containerNameFor(clusterId: string): string | null {
  const r = spawnSync("docker", ["ps", "-a", "--filter", `name=${clusterId}`, "--format", "{{.Names}}"],
    { encoding: "utf8", timeout: 10_000 });
  const name = (r.stdout || "").trim().split("\n")[0];
  return name || null;
}

export interface SparkrunContainerState { name: string; state: string; restartCount: number; }

/** Read-only: docker state + restart count for a sparkrun workload. null if not found. */
export function inspectSparkrunContainer(clusterId?: string): SparkrunContainerState | null {
  if (!clusterId) return null;
  const name = containerNameFor(clusterId);
  if (!name) return null;
  const r = spawnSync("docker", ["inspect", name, "--format", "{{.State.Status}} {{.RestartCount}}"],
    { encoding: "utf8", timeout: 10_000 });
  const out = (r.stdout || "").trim();
  if (!out) return null;
  const [state, rc] = out.split(/\s+/);
  return { name, state, restartCount: Number(rc) || 0 };
}

/** Read-only: snapshot a sparkrun container's FULL stdout+stderr (all restarts). */
export function snapshotContainerLogs(clusterId?: string): string {
  if (!clusterId) return "";
  const name = containerNameFor(clusterId);
  if (!name) return "";
  // Capture the FULL container log, not just the tail. The ROOT crash is at the
  // START of the log; docker's unless-stopped restarts append a re-mangled
  // command error at the END (the eugr entrypoint rebuilds the serve command
  // differently on restart). stderr (where vLLM errors go) is placed FIRST and
  // kept from its head, so `firstErrorLine` surfaces the root cause — not the
  // latest restart's masked error. Goal: an API consumer sees ALL of it.
  const r = spawnSync("docker", ["logs", name],
    { encoding: "utf8", timeout: 20_000, maxBuffer: 32 * 1024 * 1024 });
  const headCap = (s: string | null | undefined, n: number) => {
    const t = (s || "").trim();
    return t.length > n ? t.slice(0, n) + "\n…[truncated]" : t;
  };
  const stderr = headCap(r.stderr, 120_000); // errors incl. the root crash
  const stdout = headCap(r.stdout, 40_000);  // download/progress noise
  return [stderr, stdout].filter(Boolean).join("\n").trim();
}

export function isWorkloadRunning(target: string, hosts: string[]): boolean {
  try {
    execFileSync("uvx", ["--from", SPARKRUN_PKG, "sparkrun", "cluster", "check-job", target, "-H", hosts.join(",")],
      { timeout: 30_000, stdio: "ignore" });
    return true;
  } catch { return false; }
}

/**
 * Write an inline recipe YAML string to `<dir>/<deploymentId>.yaml` and return
 * the absolute path. Creates `dir` if it does not exist.
 *
 * Used by cmd:deploy when the server sends `inlineRecipeYaml` directly rather
 * than a registry name or shared-filesystem path.
 */
export function writeInlineRecipe(deploymentId: string, yaml: string, dir: string): string {
  mkdirSync(dir, { recursive: true });
  const filePath = join(dir, `${deploymentId}.yaml`);
  writeFileSync(filePath, yaml, "utf-8");
  return filePath;
}

/**
 * Remove a previously written inline recipe file (no-op if already gone).
 */
export function removeInlineRecipe(deploymentId: string, dir: string): void {
  rmSync(join(dir, `${deploymentId}.yaml`), { force: true });
}
