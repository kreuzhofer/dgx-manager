import { prisma } from "../prisma.js";
import { sshExec as defaultSshExec } from "./executor.js";
import { broadcast } from "../sse.js";
import { isValidIpv4 } from "../ws/node-ip.js";
import { reseedClusterKnownHosts, type NodeForSeed, type ReseedReport } from "./known-hosts.js";

/** Load online nodes that have a usable mgmt IP, shaped for the orchestrator. */
export async function loadOnlineSeedNodes(): Promise<NodeForSeed[]> {
  const nodes = await prisma.node.findMany({
    where: { status: "online" },
    select: { id: true, ipAddress: true, fastIpAddress: true },
  });
  return nodes
    .filter((n) => isValidIpv4(n.ipAddress))
    .map((n) => ({ id: n.id, host: n.ipAddress as string, ipAddress: n.ipAddress, fastIpAddress: n.fastIpAddress }));
}

/** Load online nodes and reseed. Used by route (force) and provision trigger. */
export async function triggerClusterReseed(opts: { force?: boolean } = {}): Promise<ReseedReport> {
  const nodes = await loadOnlineSeedNodes();
  return reseedClusterKnownHosts(nodes, { sshExec: defaultSshExec, broadcast, logger: console }, opts);
}

let debounceTimer: ReturnType<typeof setTimeout> | null = null;

/**
 * Coalesce a burst of agent-reconnect events into a single reseed ~delayMs
 * later. Subject to the orchestrator's single-flight + 5-minute throttle.
 */
export function scheduleDebouncedReseed(delayMs = 10_000): void {
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    debounceTimer = null;
    triggerClusterReseed().catch((e) => console.error(`known-hosts reseed failed: ${(e as Error).message}`));
  }, delayMs);
  debounceTimer.unref?.();
}
