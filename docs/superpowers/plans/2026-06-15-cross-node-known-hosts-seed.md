# Cross-Node `known_hosts` Auto-Seeding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the cross-node SSH `known_hosts` trust mesh durable and automatic, so neither a node re-image nor onboarding a new node ever reintroduces the `rc=255 Host key verification failed` blocker that stalls multi-node (TP=2) deploys.

**Architecture:** A new server-side module `packages/server/src/ssh/known-hosts.ts` splits pure helpers (IP parsing, subnet filtering, remote-script building) from an IO-coupled orchestrator (`reseedClusterKnownHosts`) wrapped in a single-flight + 5-minute-throttle guard. The orchestrator SSHes each online node to discover its live IPs (`ip -o -4 addr`), unions them into a trusted set, then re-scans them into each node's system-wide `/etc/ssh/ssh_known_hosts`. A thin prisma-coupled trigger module fires it from three places: provision-complete, debounced agent-reconnect, and a manual `POST /api/cluster/reseed-known-hosts` (plus a dashboard button). No Prisma schema change, no `packages/agent/src` change.

**Tech Stack:** TypeScript (ES modules, strict), Express 5, Prisma (SQLite), `ssh2` via the existing `sshExec`, Vitest + fast-check + supertest. Next.js 15 dashboard.

**Spec:** `docs/superpowers/specs/2026-06-15-cross-node-known-hosts-seed-design.md`

**No agent version bump:** every change is in `packages/server` and `packages/dashboard`. Nothing under `packages/agent/src/` is touched, so `./scripts/bump-agent-version.sh` is NOT run for this work.

---

## File Structure

| File | Responsibility | Change |
| --- | --- | --- |
| `packages/server/src/ssh/known-hosts.ts` | Types + pure helpers (`parseGlobalIpv4s`, `filterToClusterSubnets`, `buildKeyscanSeedScript`) + orchestrator `reseedClusterKnownHosts` + single-flight/throttle guard + `resetKnownHostsGuard` (test seam) | **Create** |
| `packages/server/src/ssh/known-hosts.test.ts` | Property/unit tests for pure helpers + guard behaviour | **Create** |
| `packages/server/src/ssh/known-hosts-trigger.ts` | Prisma-coupled wrappers: `loadOnlineSeedNodes`, `triggerClusterReseed`, `scheduleDebouncedReseed` | **Create** |
| `packages/server/src/routes/cluster.ts` | `POST /reseed-known-hosts` endpoint | **Create** |
| `packages/server/src/__tests__/integration/cluster.reseed.test.ts` | Orchestrator integration (injected `sshExec`) + endpoint (supertest) | **Create** |
| `packages/server/src/index.ts` | Mount `clusterRouter` at `/api/cluster` | Modify |
| `packages/server/src/routes/nodes.ts` | Fire reseed in the provision-complete callback | Modify |
| `packages/server/src/ws/agent-hub.ts` | Schedule debounced reseed at the two agent-online sites | Modify |
| `packages/dashboard/lib/api.ts` | `reseedKnownHosts()` fetch helper | Modify |
| `packages/dashboard/app/nodes/page.tsx` | "Reseed SSH trust" button calling the helper | Modify |

**Reused existing code:**
- `isValidIpv4` from `packages/server/src/ws/node-ip.ts` (injection guard).
- `sshExec` (default) from `packages/server/src/ssh/executor.ts`, signature `(host: string, command: string, options?: { timeout?: number }) => Promise<{ code: number; stdout: string; stderr: string }>`.
- `broadcast` from `packages/server/src/sse.ts`, signature `(event: { type: string; payload: unknown }) => void`.
- Injection pattern in routes: `const sshExec = (req.app.get("sshExec") || defaultSshExec) as typeof defaultSshExec;`.
- Integration-test pattern: per-suite SQLite via `mkdtempSync` + `DATABASE_URL` set BEFORE dynamic `import` of prisma + `npx prisma db push --force-reset` with `PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION`. Canonical example: `packages/server/src/__tests__/integration/nodes.power.test.ts`.

**Reference — real `ip -o -4 addr show scope global` output (a GB10 node):**
```
2: enP7s7    inet 192.168.44.36/24 brd 192.168.44.255 scope global dynamic noprefixroute enP7s7\       valid_lft 6132sec preferred_lft 6132sec
3: enp1s0f0np0    inet 192.168.100.10/24 brd 192.168.100.255 scope global noprefixroute enp1s0f0np0\       valid_lft forever preferred_lft forever
5: enP2p1s0f0np0    inet 192.168.100.30/24 brd 192.168.100.255 scope global noprefixroute enP2p1s0f0np0\       valid_lft forever preferred_lft forever
```

---

## Task 1: Types + `parseGlobalIpv4s` pure helper

**Files:**
- Create: `packages/server/src/ssh/known-hosts.ts`
- Create (test): `packages/server/src/ssh/known-hosts.test.ts`

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/ssh/known-hosts.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { fc, test as propTest } from "@fast-check/vitest";
import { parseGlobalIpv4s } from "./known-hosts.js";

describe("parseGlobalIpv4s", () => {
  it("extracts the inet IPv4s from `ip -o -4 addr` output, without prefixes", () => {
    const out = [
      "2: enP7s7    inet 192.168.44.36/24 brd 192.168.44.255 scope global dynamic noprefixroute enP7s7\\       valid_lft 6132sec preferred_lft 6132sec",
      "3: enp1s0f0np0    inet 192.168.100.10/24 brd 192.168.100.255 scope global noprefixroute enp1s0f0np0\\       valid_lft forever preferred_lft forever",
      "5: enP2p1s0f0np0    inet 192.168.100.30/24 brd 192.168.100.255 scope global noprefixroute enP2p1s0f0np0",
    ].join("\n");
    expect(parseGlobalIpv4s(out)).toEqual(["192.168.44.36", "192.168.100.10", "192.168.100.30"]);
  });

  it("returns [] for empty input", () => {
    expect(parseGlobalIpv4s("")).toEqual([]);
  });

  // Invariant: every returned value is a bare dotted-quad with no slash/prefix.
  propTest.prop([fc.array(fc.tuple(fc.integer({ min: 0, max: 255 }), fc.integer({ min: 0, max: 255 }), fc.integer({ min: 0, max: 255 }), fc.integer({ min: 0, max: 255 }), fc.integer({ min: 0, max: 32 })))])(
    "never returns a token containing a slash",
    (quads) => {
      const lines = quads.map(([a, b, c, d, p], i) => `${i}: if${i}    inet ${a}.${b}.${c}.${d}/${p} scope global`);
      const result = parseGlobalIpv4s(lines.join("\n"));
      expect(result.every((ip) => !ip.includes("/"))).toBe(true);
      expect(result.length).toBe(quads.length);
    },
  );
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/server/src/ssh/known-hosts.test.ts`
Expected: FAIL — `parseGlobalIpv4s` is not exported / module missing.

- [ ] **Step 3: Write minimal implementation**

Create `packages/server/src/ssh/known-hosts.ts`:

```ts
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/server/src/ssh/known-hosts.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/ssh/known-hosts.ts packages/server/src/ssh/known-hosts.test.ts
git commit -m "feat(server): parseGlobalIpv4s + known-hosts types"
```

---

## Task 2: `filterToClusterSubnets` pure helper

**Files:**
- Modify: `packages/server/src/ssh/known-hosts.ts`
- Modify (test): `packages/server/src/ssh/known-hosts.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `packages/server/src/ssh/known-hosts.test.ts`:

```ts
import { filterToClusterSubnets, type NodeForSeed } from "./known-hosts.js";

const NODES: NodeForSeed[] = [
  { id: "a", host: "192.168.44.36", ipAddress: "192.168.44.36", fastIpAddress: "192.168.100.10" },
  { id: "b", host: "192.168.44.37", ipAddress: "192.168.44.37", fastIpAddress: "192.168.100.11" },
];

describe("filterToClusterSubnets", () => {
  it("keeps mgmt + IB IPs and drops docker/veth noise", () => {
    const ips = ["192.168.44.36", "192.168.100.10", "192.168.100.30", "172.17.0.1", "10.0.0.5"];
    expect(filterToClusterSubnets(ips, NODES)).toEqual(["192.168.100.10", "192.168.100.30", "192.168.44.36"]);
  });

  it("dedupes and sorts", () => {
    const ips = ["192.168.44.37", "192.168.44.36", "192.168.44.36"];
    expect(filterToClusterSubnets(ips, NODES)).toEqual(["192.168.44.36", "192.168.44.37"]);
  });

  // Invariant: output is a subset of input, and every output IP shares a /24 with some node.
  propTest.prop([
    fc.array(fc.tuple(fc.integer({ min: 0, max: 255 }), fc.integer({ min: 0, max: 255 }), fc.integer({ min: 0, max: 255 }), fc.integer({ min: 0, max: 255 }))),
  ])("output ⊆ input ∧ all in a node /24", (quads) => {
    const ips = quads.map(([a, b, c, d]) => `${a}.${b}.${c}.${d}`);
    const result = filterToClusterSubnets(ips, NODES);
    const clusterPrefixes = new Set(["192.168.44", "192.168.100"]);
    for (const ip of result) {
      expect(ips).toContain(ip);
      expect(clusterPrefixes.has(ip.split(".").slice(0, 3).join("."))).toBe(true);
    }
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/server/src/ssh/known-hosts.test.ts`
Expected: FAIL — `filterToClusterSubnets` not exported.

- [ ] **Step 3: Write minimal implementation**

Append to `packages/server/src/ssh/known-hosts.ts`:

```ts
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/server/src/ssh/known-hosts.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/ssh/known-hosts.ts packages/server/src/ssh/known-hosts.test.ts
git commit -m "feat(server): filterToClusterSubnets known-hosts helper"
```

---

## Task 3: `buildKeyscanSeedScript` pure helper (injection guard)

**Files:**
- Modify: `packages/server/src/ssh/known-hosts.ts`
- Modify (test): `packages/server/src/ssh/known-hosts.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `packages/server/src/ssh/known-hosts.test.ts`:

```ts
import { buildKeyscanSeedScript } from "./known-hosts.js";

describe("buildKeyscanSeedScript", () => {
  const ips = ["192.168.44.36", "192.168.100.10"];

  it("emits an ssh-keygen -R for each IP, a keyscan of all IPs, and a dedup sort", () => {
    const script = buildKeyscanSeedScript(ips);
    expect(script).toContain("ssh-keygen -f /etc/ssh/ssh_known_hosts -R 192.168.44.36");
    expect(script).toContain("ssh-keygen -f /etc/ssh/ssh_known_hosts -R 192.168.100.10");
    expect(script).toContain("ssh-keyscan -T 5 192.168.44.36 192.168.100.10");
    expect(script).toContain("sort -u /etc/ssh/ssh_known_hosts -o /etc/ssh/ssh_known_hosts");
  });

  it("throws on a non-IPv4 element (shell-injection guard)", () => {
    expect(() => buildKeyscanSeedScript(["192.168.44.36", "1.2.3.4; rm -rf /"])).toThrow();
  });

  it("throws on empty input", () => {
    expect(() => buildKeyscanSeedScript([])).toThrow();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/server/src/ssh/known-hosts.test.ts`
Expected: FAIL — `buildKeyscanSeedScript` not exported.

- [ ] **Step 3: Write minimal implementation**

Append to `packages/server/src/ssh/known-hosts.ts`:

```ts
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/server/src/ssh/known-hosts.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/ssh/known-hosts.ts packages/server/src/ssh/known-hosts.test.ts
git commit -m "feat(server): buildKeyscanSeedScript with injection guard"
```

---

## Task 4: Orchestrator `reseedClusterKnownHosts` + single-flight/throttle guard

**Files:**
- Modify: `packages/server/src/ssh/known-hosts.ts`
- Modify (test): `packages/server/src/ssh/known-hosts.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `packages/server/src/ssh/known-hosts.test.ts`:

```ts
import { beforeEach, vi } from "vitest";
import { reseedClusterKnownHosts, resetKnownHostsGuard, type ReseedDeps } from "./known-hosts.js";

const IP_OUT_A = "2: e0 inet 192.168.44.36/24 scope global\n3: e1 inet 192.168.100.10/24 scope global";
const IP_OUT_B = "2: e0 inet 192.168.44.37/24 scope global\n3: e1 inet 192.168.100.11/24 scope global";

function gatherSeedStub() {
  // Returns ip output for the gather command, code 0 for the seed script.
  return vi.fn(async (host: string, command: string) => {
    if (command.includes("ip -o -4 addr")) {
      return { code: 0, stdout: host === "192.168.44.36" ? IP_OUT_A : IP_OUT_B, stderr: "" };
    }
    return { code: 0, stdout: "", stderr: "" };
  });
}

describe("reseedClusterKnownHosts", () => {
  beforeEach(() => resetKnownHostsGuard());

  it("gathers IPs from all nodes then seeds each with the unioned trusted set", async () => {
    const sshExec = gatherSeedStub();
    const report = await reseedClusterKnownHosts(NODES, { sshExec });
    expect(report.trustedIps).toEqual(["192.168.100.10", "192.168.100.11", "192.168.44.36", "192.168.44.37"]);
    expect(report.perNode.every((p) => p.ok)).toBe(true);
    expect(report.perNode.map((p) => p.ipsSeeded)).toEqual([4, 4]);
  });

  it("skips an unreachable node in gather but still seeds the reachable ones", async () => {
    const sshExec = vi.fn(async (host: string, command: string) => {
      if (host === "192.168.44.37" && command.includes("ip -o -4 addr")) throw new Error("unreachable");
      if (command.includes("ip -o -4 addr")) return { code: 0, stdout: IP_OUT_A, stderr: "" };
      return { code: 0, stdout: "", stderr: "" };
    });
    const report = await reseedClusterKnownHosts(NODES, { sshExec });
    expect(report.trustedIps).toEqual(["192.168.100.10", "192.168.44.36"]);
    // node b's seed still runs (it just contributed no IPs); both seed calls succeed
    expect(report.perNode.find((p) => p.host === "192.168.44.37")?.ok).toBe(true);
  });

  it("is single-flight: concurrent calls share one underlying run", async () => {
    const sshExec = gatherSeedStub();
    const [r1, r2] = await Promise.all([
      reseedClusterKnownHosts(NODES, { sshExec }),
      reseedClusterKnownHosts(NODES, { sshExec }),
    ]);
    expect(r1).toBe(r2);
    // 2 nodes × (1 gather + 1 seed) = 4 sshExec calls for exactly one run
    expect(sshExec).toHaveBeenCalledTimes(4);
  });

  it("throttles a second automatic run within 5 minutes, but force bypasses it", async () => {
    let t = 1_000_000;
    const now = () => t;
    const deps: ReseedDeps = { sshExec: gatherSeedStub(), now };
    await reseedClusterKnownHosts(NODES, deps);
    t += 60_000; // +1 min
    const throttled = await reseedClusterKnownHosts(NODES, deps);
    expect(throttled.skipped).toBe("throttled");
    const forced = await reseedClusterKnownHosts(NODES, deps, { force: true });
    expect(forced.skipped).toBeUndefined();
    expect(forced.perNode.every((p) => p.ok)).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/server/src/ssh/known-hosts.test.ts`
Expected: FAIL — `reseedClusterKnownHosts` / `resetKnownHostsGuard` not exported.

- [ ] **Step 3: Write minimal implementation**

Append to `packages/server/src/ssh/known-hosts.ts`:

```ts
const THROTTLE_MS = 5 * 60 * 1000;
const GATHER_TIMEOUT_MS = 10_000;
const SEED_TIMEOUT_MS = 30_000;

let inFlight: Promise<ReseedReport> | null = null;
let lastSuccessAt: number | null = null;

/** Reset module-level guard state. For tests only. */
export function resetKnownHostsGuard(): void {
  inFlight = null;
  lastSuccessAt = null;
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
 * share one run) and throttled (a successful automatic run suppresses further
 * automatic runs for 5 minutes). `force: true` bypasses the throttle (manual op).
 */
export async function reseedClusterKnownHosts(
  nodes: NodeForSeed[],
  deps: ReseedDeps,
  opts: { force?: boolean } = {},
): Promise<ReseedReport> {
  const now = deps.now ?? Date.now;
  if (inFlight) return inFlight;
  if (!opts.force && lastSuccessAt !== null && now() - lastSuccessAt < THROTTLE_MS) {
    return { trustedIps: [], perNode: [], skipped: "throttled" };
  }
  inFlight = doReseed(nodes, deps);
  try {
    const report = await inFlight;
    if (report.perNode.some((p) => p.ok)) lastSuccessAt = now();
    return report;
  } finally {
    inFlight = null;
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/server/src/ssh/known-hosts.test.ts`
Expected: PASS (all describe blocks).

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/ssh/known-hosts.ts packages/server/src/ssh/known-hosts.test.ts
git commit -m "feat(server): reseedClusterKnownHosts orchestrator + single-flight/throttle guard"
```

---

## Task 5: Trigger wrappers (`known-hosts-trigger.ts`)

**Files:**
- Create: `packages/server/src/ssh/known-hosts-trigger.ts`

No dedicated unit test: this is thin prisma/IO glue exercised by the endpoint integration test (Task 6) and the trigger call-sites. (Risk-tier note: low-risk IO glue — covered indirectly.)

- [ ] **Step 1: Write the implementation**

Create `packages/server/src/ssh/known-hosts-trigger.ts`:

```ts
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
```

- [ ] **Step 2: Typecheck**

Run: `npm run build -w packages/server` (or `npx tsc -p packages/server`)
Expected: no type errors.

- [ ] **Step 3: Commit**

```bash
git add packages/server/src/ssh/known-hosts-trigger.ts
git commit -m "feat(server): known-hosts trigger wrappers (load/trigger/debounce)"
```

---

## Task 6: Manual endpoint `POST /api/cluster/reseed-known-hosts` + mount

**Files:**
- Create: `packages/server/src/routes/cluster.ts`
- Modify: `packages/server/src/index.ts`
- Create (test): `packages/server/src/__tests__/integration/cluster.reseed.test.ts`

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/__tests__/integration/cluster.reseed.test.ts`:

```ts
import { beforeAll, afterAll, afterEach, describe, it, expect, vi } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "cluster-reseed-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;
let clusterRouter: typeof import("../../routes/cluster.js").clusterRouter;
let resetKnownHostsGuard: typeof import("../../ssh/known-hosts.js").resetKnownHostsGuard;

beforeAll(async () => {
  execSync("npx prisma db push --force-reset", {
    cwd: process.cwd().replace(/\/packages\/server.*$/, ""),
    env: {
      ...process.env,
      DATABASE_URL: `file:${DB_PATH}`,
      PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION:
        "User consented to db push --force-reset against per-suite SQLite test databases in /tmp on 2026-05-03 (option #1)",
    },
    stdio: "pipe",
  });
  ({ prisma } = await import("../../prisma.js"));
  ({ clusterRouter } = await import("../../routes/cluster.js"));
  ({ resetKnownHostsGuard } = await import("../../ssh/known-hosts.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

afterEach(async () => {
  resetKnownHostsGuard();
  await prisma.node.deleteMany();
});

function makeApp(sshExec: any) {
  const app = express();
  app.use(express.json());
  app.set("sshExec", sshExec);
  app.use("/api/cluster", clusterRouter);
  return app;
}

const IP_OUT = "2: e0 inet 192.168.44.41/24 scope global\n3: e1 inet 192.168.100.41/24 scope global";

describe("POST /api/cluster/reseed-known-hosts", () => {
  it("200 with a per-node report when nodes are seeded", async () => {
    await prisma.node.create({ data: { name: "n1", ipAddress: "192.168.44.41", status: "online" } });
    const sshExec = vi.fn(async (_host: string, command: string) =>
      command.includes("ip -o -4 addr") ? { code: 0, stdout: IP_OUT, stderr: "" } : { code: 0, stdout: "", stderr: "" },
    );
    const res = await request(makeApp(sshExec)).post("/api/cluster/reseed-known-hosts").send({});
    expect(res.status).toBe(200);
    expect(res.body.perNode).toHaveLength(1);
    expect(res.body.perNode[0].ok).toBe(true);
    expect(res.body.trustedIps).toContain("192.168.44.41");
  });

  it("502 when every node is unreachable", async () => {
    await prisma.node.create({ data: { name: "n1", ipAddress: "192.168.44.41", status: "online" } });
    const sshExec = vi.fn(async () => {
      throw new Error("unreachable");
    });
    const res = await request(makeApp(sshExec)).post("/api/cluster/reseed-known-hosts").send({});
    expect(res.status).toBe(502);
    expect(res.body.perNode[0].ok).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/server/src/__tests__/integration/cluster.reseed.test.ts`
Expected: FAIL — `routes/cluster.js` does not exist.

- [ ] **Step 3: Write the route**

Create `packages/server/src/routes/cluster.ts`:

```ts
import { Router } from "express";
import { sshExec as defaultSshExec } from "../ssh/executor.js";
import { broadcast } from "../sse.js";
import { reseedClusterKnownHosts } from "../ssh/known-hosts.js";
import { loadOnlineSeedNodes } from "../ssh/known-hosts-trigger.js";

export const clusterRouter = Router();

// Manually reseed the cross-node known_hosts mesh. Bypasses the throttle
// (operator intent is explicit). 200 if any node seeded, 502 if none.
clusterRouter.post("/reseed-known-hosts", async (req, res) => {
  const sshExec = (req.app.get("sshExec") || defaultSshExec) as typeof defaultSshExec;
  try {
    const nodes = await loadOnlineSeedNodes();
    const report = await reseedClusterKnownHosts(nodes, { sshExec, broadcast, logger: console }, { force: true });
    const anyOk = report.perNode.some((p) => p.ok);
    res.status(anyOk ? 200 : 502).json(report);
  } catch (e) {
    res.status(500).json({ error: (e as Error).message });
  }
});
```

- [ ] **Step 4: Mount the router**

In `packages/server/src/index.ts`, add the import alongside the other route imports (near line 20):

```ts
import { clusterRouter } from "./routes/cluster.js";
```

And mount it alongside the other `app.use("/api/...")` lines (near line 69):

```ts
app.use("/api/cluster", clusterRouter);
```

- [ ] **Step 5: Run test to verify it passes**

Run: `npx vitest run packages/server/src/__tests__/integration/cluster.reseed.test.ts`
Expected: PASS (both cases).

- [ ] **Step 6: Commit**

```bash
git add packages/server/src/routes/cluster.ts packages/server/src/index.ts packages/server/src/__tests__/integration/cluster.reseed.test.ts
git commit -m "feat(server): POST /api/cluster/reseed-known-hosts endpoint"
```

---

## Task 7: Provision-complete trigger

**Files:**
- Modify: `packages/server/src/routes/nodes.ts`

- [ ] **Step 1: Add the import**

In `packages/server/src/routes/nodes.ts`, alongside the existing imports add:

```ts
import { triggerClusterReseed } from "../ssh/known-hosts-trigger.js";
```

- [ ] **Step 2: Fire the reseed in the provision-complete callback**

In the `provisionNode(...).then(async (log) => { ... })` block (around line 350), after the post-provision `prisma.node.update(...)` and its `sseBroadcast({ type: "node:provision", ... })`, add a fire-and-forget reseed so a freshly provisioned node is trusted cluster-wide:

```ts
    // A newly provisioned node must trust (and be trusted by) the rest of the
    // cluster, else node→node SSH for multi-node deploys fails with rc=255.
    // force:true — onboarding is deliberate and must not be suppressed by the throttle.
    triggerClusterReseed({ force: true }).catch((e) => console.error(`known-hosts reseed after provision failed: ${(e as Error).message}`));
```

- [ ] **Step 3: Typecheck + run the existing nodes tests**

Run: `npx tsc -p packages/server --noEmit && npx vitest run packages/server/src/__tests__/integration/nodes.power.test.ts`
Expected: no type errors; nodes.power tests still PASS (no behavioural change to power routes).

- [ ] **Step 4: Commit**

```bash
git add packages/server/src/routes/nodes.ts
git commit -m "feat(server): reseed known_hosts after node provisioning"
```

---

## Task 8: Debounced agent-reconnect trigger

**Files:**
- Modify: `packages/server/src/ws/agent-hub.ts`

- [ ] **Step 1: Add the import**

In `packages/server/src/ws/agent-hub.ts`, alongside the existing imports add:

```ts
import { scheduleDebouncedReseed } from "../ssh/known-hosts-trigger.js";
```

- [ ] **Step 2: Schedule a reseed at both agent-online sites**

There are two places where a node is set online: the `agent:register` handler (the `sseBroadcast({ type: "node:status", ..., status: "online", agentVersion })` near line 191) and the `agent:register-token` handler (the `sseBroadcast({ type: "node:status", ..., status: "online", agentVersion: tokenAgentVersion })` near line 279). Immediately after EACH of those two `sseBroadcast(...)` calls, add:

```ts
            // A (re)connected agent may be a re-imaged node whose host key
            // changed — refresh the cluster known_hosts mesh (debounced + throttled).
            scheduleDebouncedReseed();
```

- [ ] **Step 3: Typecheck + run the agent-hub tests**

Run: `npx tsc -p packages/server --noEmit && npx vitest run packages/server/src/ws/agent-hub.pull-progress.test.ts`
Expected: no type errors; existing agent-hub test PASSES (debounce timer is `unref`'d and fire-and-forget, so it does not affect the test).

- [ ] **Step 4: Commit**

```bash
git add packages/server/src/ws/agent-hub.ts
git commit -m "feat(server): debounced known_hosts reseed on agent (re)connect"
```

---

## Task 9: Dashboard "Reseed SSH trust" button

**Files:**
- Modify: `packages/dashboard/lib/api.ts`
- Modify: `packages/dashboard/app/nodes/page.tsx`

- [ ] **Step 1: Add the API helper**

In `packages/dashboard/lib/api.ts`, follow the existing fetch-wrapper style used by the other POST helpers in that file and add:

```ts
export async function reseedKnownHosts(): Promise<{
  trustedIps: string[];
  perNode: Array<{ nodeId: string; host: string; ipsSeeded: number; ok: boolean; error?: string }>;
}> {
  const res = await fetch(`${API_URL}/api/cluster/reseed-known-hosts`, { method: "POST" });
  if (!res.ok && res.status !== 502) throw new Error(`Reseed failed: ${res.status}`);
  return res.json();
}
```

(Use whatever the file already names its base URL constant — match the existing helpers; this plan assumes `API_URL`.)

- [ ] **Step 2: Add the button to the nodes page**

In `packages/dashboard/app/nodes/page.tsx`, import the helper and add a button in the page header/toolbar (next to existing controls), following the page's existing button styling. Example wiring:

```tsx
import { reseedKnownHosts } from "@/lib/api";
// ...inside the component:
const [reseeding, setReseeding] = useState(false);
async function handleReseed() {
  setReseeding(true);
  try {
    const report = await reseedKnownHosts();
    const okCount = report.perNode.filter((p) => p.ok).length;
    alert(`SSH trust reseeded: ${okCount}/${report.perNode.length} nodes, ${report.trustedIps.length} IPs.`);
  } catch (e) {
    alert(`Reseed failed: ${(e as Error).message}`);
  } finally {
    setReseeding(false);
  }
}
// ...in the toolbar JSX:
<button onClick={handleReseed} disabled={reseeding} className="<match existing button classes>">
  {reseeding ? "Reseeding…" : "Reseed SSH trust"}
</button>
```

(Match the page's actual component structure, button classes, and any existing `useState` import. Keep it minimal — a header button + an `alert` summary is sufficient for this operator action.)

- [ ] **Step 3: Build the dashboard to verify it compiles**

Run: `npm run build -w packages/dashboard`
Expected: build succeeds (no type errors).

- [ ] **Step 4: Commit**

```bash
git add packages/dashboard/lib/api.ts packages/dashboard/app/nodes/page.tsx
git commit -m "feat(dashboard): Reseed SSH trust button on nodes page"
```

---

## Task 10: Full suite + final verification

- [ ] **Step 1: Run the whole test suite**

Run: `npm test`
Expected: all tests green, including the new `known-hosts.test.ts` and `cluster.reseed.test.ts`.

- [ ] **Step 2: Typecheck both packages**

Run: `npx tsc -p packages/server --noEmit && npm run build -w packages/dashboard`
Expected: no type errors in either package.

- [ ] **Step 3: Confirm no agent change**

Run: `git diff --name-only main...HEAD -- packages/agent/src`
Expected: empty output (no agent source touched → no version bump needed).

---

## Self-Review (completed by plan author)

**Spec coverage:**
- Live SSH IP discovery (dual IB IPs, no schema change) → Tasks 1, 4 (`parseGlobalIpv4s`, gather phase).
- Subnet filtering drops docker/veth → Task 2.
- Remove-then-scan for key rotation + system-wide file → Task 3.
- Single-flight + 5-min throttle, idempotent whole-mesh → Task 4.
- Trigger: provision-complete → Task 7. Agent-reconnect (debounced) → Task 8. Manual endpoint + button → Tasks 6, 9.
- Injection guard via `isValidIpv4` → Task 3 (+ used in Task 2, trigger).
- Error handling: unreachable skip, per-node report, 200/502, `cluster:reseed` SSE → Tasks 4, 6.
- Test matrix (property/unit pure + orchestrator integration + endpoint) → Tasks 1–4, 6.
- No agent bump → Task 10 Step 3 asserts it.

**Placeholder scan:** none — every code step is complete. (Task 9 intentionally defers to the page's existing button classes; flagged inline, low-risk UI.)

**Type consistency:** `NodeForSeed`, `PerNodeResult`, `ReseedReport`, `ReseedDeps` defined in Task 1 and used unchanged in Tasks 4–6. `reseedClusterKnownHosts(nodes, deps, opts)` signature is identical across Tasks 4, 6, and the trigger wrapper. `sshExec` signature matches the real `executor.ts` export.
