import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { dirname } from "path";
import { homedir } from "os";

export interface TrackedDeployment {
  deploymentId: string;
  recipeFile: string;
  recipeName: string;
  port: number;
  startedAt: string;
  clusterNodes?: string[];
}

// Default to a user-writable location. The agent runs as a non-root systemd
// user (typically `daniel`), so /opt/dgx-agent/ is read-only and writes fail
// with EACCES. $HOME/.dgx-agent/ is always writable.
const STORE_PATH =
  process.env.DEPLOYMENT_STORE_PATH || `${homedir()}/.dgx-agent/deployments.json`;

function ensureDir() {
  try {
    const dir = dirname(STORE_PATH);
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  } catch (err) {
    console.warn(`[deployment-store] Failed to create dir for ${STORE_PATH}:`, err);
  }
}

export function loadDeployments(): TrackedDeployment[] {
  try {
    if (!existsSync(STORE_PATH)) return [];
    return JSON.parse(readFileSync(STORE_PATH, "utf-8"));
  } catch {
    return [];
  }
}

export function saveDeployment(d: TrackedDeployment) {
  ensureDir();
  try {
    const list = loadDeployments().filter((x) => x.deploymentId !== d.deploymentId);
    list.push(d);
    writeFileSync(STORE_PATH, JSON.stringify(list, null, 2));
  } catch (err) {
    // Non-fatal: deployment still starts, but agent can't reattach across
    // restarts. Log once and continue rather than crashing the flow.
    console.warn(`[deployment-store] Failed to persist deployment ${d.deploymentId}:`, err);
  }
}

export function removeDeployment(deploymentId: string) {
  ensureDir();
  try {
    const list = loadDeployments().filter((x) => x.deploymentId !== deploymentId);
    writeFileSync(STORE_PATH, JSON.stringify(list, null, 2));
  } catch (err) {
    console.warn(`[deployment-store] Failed to remove deployment ${deploymentId}:`, err);
  }
}

export function clearDeployments() {
  ensureDir();
  try {
    writeFileSync(STORE_PATH, "[]");
  } catch (err) {
    console.warn(`[deployment-store] Failed to clear deployments:`, err);
  }
}
