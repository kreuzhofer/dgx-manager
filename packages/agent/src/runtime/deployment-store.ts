import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { dirname } from "path";

export interface TrackedDeployment {
  deploymentId: string;
  recipeFile: string;
  recipeName: string;
  port: number;
  startedAt: string;
}

const STORE_PATH =
  process.env.DEPLOYMENT_STORE_PATH || "/opt/dgx-agent/deployments.json";

function ensureDir() {
  const dir = dirname(STORE_PATH);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
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
  const list = loadDeployments().filter((x) => x.deploymentId !== d.deploymentId);
  list.push(d);
  writeFileSync(STORE_PATH, JSON.stringify(list, null, 2));
}

export function removeDeployment(deploymentId: string) {
  ensureDir();
  const list = loadDeployments().filter((x) => x.deploymentId !== deploymentId);
  writeFileSync(STORE_PATH, JSON.stringify(list, null, 2));
}

export function clearDeployments() {
  ensureDir();
  writeFileSync(STORE_PATH, "[]");
}
