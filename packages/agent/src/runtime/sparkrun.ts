import { spawn, execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { SPARKRUN_PKG } from "../recipes.js";
import { buildSparkrunArgs, type SparkrunLaunchOptions } from "./sparkrun-args.js";
import { parseClusterId } from "./sparkrun-parse.js";
import { saveDeployment, removeDeployment } from "./deployment-store.js";

export type Opts = Omit<SparkrunLaunchOptions, "recipeRef"> & { recipeName?: string };

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
    if (!clusterId) { buf += s; const id = parseClusterId(buf); if (id) { clusterId = id; persist(); } }
  };
  child.stdout?.on("data", onData);
  child.stderr?.on("data", onData);
  child.on("exit", (code) => onExit(code));
  persist();
}

export function stopSparkrun(deploymentId: string, target: string, hosts: string[], tp?: number): void {
  const args = ["--from", SPARKRUN_PKG, "sparkrun", "stop", target, "-H", hosts.join(",")];
  if (tp != null) args.push("--tp", String(tp));
  try { execFileSync("uvx", args, { timeout: 120_000 }); }
  finally { removeDeployment(deploymentId); }
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
