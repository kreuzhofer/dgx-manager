import { ChildProcess } from "child_process";
import { removeDeployment } from "./deployment-store.js";

export interface VllmStatus {
  deploymentId: string;
  recipeName: string;
  port: number;
  alive: boolean;
  containerRunning: boolean;
  requestsRunning: number | null;
  requestsWaiting: number | null;
  kvCacheUsage: number | null;
  tps: number | null;
  error?: string;
}

interface VllmInstance {
  process: ChildProcess;
  recipeName: string;
  port: number;
  stopping?: boolean;
}

const running = new Map<string, VllmInstance>();

/** Check if a deployment is being stopped (suppress onExit status reports). */
export function isStopping(deploymentId: string): boolean {
  return running.get(deploymentId)?.stopping === true;
}

/**
 * Drop a deployment from both the in-memory `running` map and the on-disk
 * tracking store. Use this in failure paths where the deployment died on its
 * own (not via an intentional stop) — clearing only the disk store leaves a
 * stale in-memory entry that the health loop keeps re-detecting every tick,
 * producing an endless "Container stopped unexpectedly" stream.
 */
export function untrackDeployment(deploymentId: string): void {
  running.delete(deploymentId);
  removeDeployment(deploymentId);
}
