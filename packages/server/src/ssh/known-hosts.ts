import { isValidIpv4 } from "../ws/node-ip.js";

/** A node the orchestrator can SSH into and seed. `host` is the SSH target (mgmt IP). */
export interface NodeForSeed {
  id: string;
  host: string;
  ipAddress: string | null;
  fastIpAddress: string | null;
}

export interface PerNodeResult {
  nodeId: string;
  host: string;
  ipsSeeded: number;
  ok: boolean;
  error?: string;
}

export interface ReseedReport {
  trustedIps: string[];
  perNode: PerNodeResult[];
  /** Set when the guard short-circuited instead of running. */
  skipped?: "throttled";
}

export interface ReseedDeps {
  sshExec: (host: string, command: string, options?: { timeout?: number }) => Promise<{ code: number; stdout: string; stderr: string }>;
  logger?: Pick<Console, "info" | "warn" | "error">;
  broadcast?: (event: { type: string; payload: unknown }) => void;
  /** Injectable clock for throttle testing. Defaults to Date.now. */
  now?: () => number;
}

/**
 * Extract the global IPv4 addresses from `ip -o -4 addr show scope global`
 * output. Each interface line carries `inet <ip>/<prefix>`; we return the bare
 * dotted-quads in document order, prefixes stripped.
 */
export function parseGlobalIpv4s(ipAddrOutput: string): string[] {
  const out: string[] = [];
  const re = /\binet\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\/\d+/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(ipAddrOutput)) !== null) out.push(m[1]);
  return out;
}

function slash24(ip: string): string {
  return ip.split(".").slice(0, 3).join(".");
}

/**
 * Keep only IPs whose /24 matches the /24 of any node's mgmt or fast IP. This
 * drops docker bridge / veth noise (e.g. 172.17.x) while retaining every mgmt
 * and IB-fabric address that node↔node SSH might use. Result is deduped + sorted.
 */
export function filterToClusterSubnets(ips: string[], nodes: NodeForSeed[]): string[] {
  const prefixes = new Set<string>();
  for (const n of nodes) {
    if (isValidIpv4(n.ipAddress)) prefixes.add(slash24(n.ipAddress));
    if (isValidIpv4(n.fastIpAddress)) prefixes.add(slash24(n.fastIpAddress));
  }
  const kept = new Set<string>();
  for (const ip of ips) {
    if (isValidIpv4(ip) && prefixes.has(slash24(ip))) kept.add(ip);
  }
  return [...kept].sort();
}

/**
 * Build the remote shell script (run over SSH, on each node) that refreshes
 * `/etc/ssh/ssh_known_hosts` for the given cluster IPs:
 *   1. remove any existing entry per IP (clears a rotated key after a re-image),
 *   2. ssh-keyscan all IPs and append,
 *   3. sort -u to dedup.
 *
 * Every IP is interpolated into an executed command, so each is validated as a
 * strict IPv4 literal first — this is the shell-injection boundary. Throws on
 * any non-IPv4 element or empty input (callers guard against empty before calling).
 */
export function buildKeyscanSeedScript(ips: string[]): string {
  if (ips.length === 0) throw new Error("buildKeyscanSeedScript: no IPs");
  for (const ip of ips) {
    if (!isValidIpv4(ip)) throw new Error(`buildKeyscanSeedScript: invalid IPv4 ${ip}`);
  }
  const removals = ips
    .map((ip) => `sudo ssh-keygen -f /etc/ssh/ssh_known_hosts -R ${ip} >/dev/null 2>&1 || true`)
    .join("\n");
  return [
    "sudo touch /etc/ssh/ssh_known_hosts",
    removals,
    `ssh-keyscan -T 5 ${ips.join(" ")} 2>/dev/null | sudo tee -a /etc/ssh/ssh_known_hosts >/dev/null`,
    "sudo sort -u /etc/ssh/ssh_known_hosts -o /etc/ssh/ssh_known_hosts",
  ].join("\n");
}

const THROTTLE_MS = 5 * 60 * 1000;
// Covers SSH setup overhead + a fast local `ip addr` command.
const GATHER_TIMEOUT_MS = 10_000;
const SEED_TIMEOUT_MS = 30_000;

let inFlight: Promise<ReseedReport> | null = null;
let lastRunAt: number | null = null;

/** Reset module-level guard state. For tests only. */
export function resetKnownHostsGuard(): void {
  inFlight = null;
  lastRunAt = null;
}

async function doReseed(nodes: NodeForSeed[], deps: ReseedDeps): Promise<ReseedReport> {
  const log = deps.logger ?? console;

  // Phase 1 — gather live IPs from each reachable node.
  const discovered: string[] = [];
  for (const node of nodes) {
    try {
      const r = await deps.sshExec(node.host, "ip -o -4 addr show scope global", { timeout: GATHER_TIMEOUT_MS });
      if (r.code === 0) discovered.push(...parseGlobalIpv4s(r.stdout));
      else log.warn(`known-hosts gather: ${node.host} exited ${r.code}`);
    } catch (e) {
      log.warn(`known-hosts gather: ${node.host} unreachable: ${(e as Error).message}`);
    }
  }

  const trustedIps = filterToClusterSubnets(discovered, nodes);
  const perNode: PerNodeResult[] = [];

  // Phase 2 — seed each node with the unioned trusted set.
  if (trustedIps.length > 0) {
    const script = buildKeyscanSeedScript(trustedIps);
    for (const node of nodes) {
      try {
        const r = await deps.sshExec(node.host, script, { timeout: SEED_TIMEOUT_MS });
        const ok = r.code === 0;
        perNode.push({ nodeId: node.id, host: node.host, ipsSeeded: ok ? trustedIps.length : 0, ok, error: ok ? undefined : r.stderr || `exit ${r.code}` });
      } catch (e) {
        perNode.push({ nodeId: node.id, host: node.host, ipsSeeded: 0, ok: false, error: (e as Error).message });
      }
    }
  }

  const report: ReseedReport = { trustedIps, perNode };
  deps.broadcast?.({ type: "cluster:reseed", payload: report });
  return report;
}

/**
 * Reseed the whole cluster's known_hosts mesh. Single-flight (concurrent calls
 * share one run) and rate-limited (any completed run arms the throttle for 5
 * minutes; `force: true` bypasses it). The throttle fires regardless of per-node
 * success so a persistently-unreachable node cannot cause a reseed storm on every
 * agent reconnect.
 */
export async function reseedClusterKnownHosts(
  nodes: NodeForSeed[],
  deps: ReseedDeps,
  opts: { force?: boolean } = {},
): Promise<ReseedReport> {
  const now = deps.now ?? Date.now;
  if (inFlight) return inFlight;
  if (!opts.force && lastRunAt !== null && now() - lastRunAt < THROTTLE_MS) {
    return { trustedIps: [], perNode: [], skipped: "throttled" };
  }
  inFlight = doReseed(nodes, deps);
  try {
    const report = await inFlight;
    // Arm the throttle after ANY completed run, win or lose — a failed node must
    // not bypass the rate-limit and trigger a reseed storm on every reconnect.
    lastRunAt = now();
    return report;
  } finally {
    inFlight = null;
  }
}
