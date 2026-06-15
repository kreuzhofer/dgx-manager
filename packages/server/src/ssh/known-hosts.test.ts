import { describe, it, expect, beforeEach, vi } from "vitest";
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
  // The generator mixes cluster-subnet IPs with random ones so the keep-path is actually exercised.
  propTest.prop([
    fc.array(
      fc.oneof(
        fc
          .tuple(fc.constantFrom("192.168.44", "192.168.100"), fc.integer({ min: 0, max: 255 }))
          .map(([p, d]) => `${p}.${d}`),
        fc
          .tuple(
            fc.integer({ min: 0, max: 255 }),
            fc.integer({ min: 0, max: 255 }),
            fc.integer({ min: 0, max: 255 }),
            fc.integer({ min: 0, max: 255 }),
          )
          .map((q) => q.join(".")),
      ),
    ),
  ])("output ⊆ input ∧ all in a node /24", (ips) => {
    const result = filterToClusterSubnets(ips, NODES);
    const clusterPrefixes = new Set(["192.168.44", "192.168.100"]);
    for (const ip of result) {
      expect(ips).toContain(ip);
      expect(clusterPrefixes.has(ip.split(".").slice(0, 3).join("."))).toBe(true);
    }
  });
});

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
    expect(() => buildKeyscanSeedScript(["1.2.3.4\n5.6.7.8"])).toThrow();
  });

  it("throws on empty input", () => {
    expect(() => buildKeyscanSeedScript([])).toThrow();
  });
});

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
    // Referential equality proves both callers received the same Promise result.
    expect(r1).toBe(r2);
    // 4 sshExec calls == 2 nodes × (1 gather + 1 seed) = exactly one underlying run (single-flight proof).
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

  it("returns trustedIps:[] and perNode:[] when every node's gather returns empty stdout", async () => {
    // All nodes respond with code 0 but no IP lines — trustedIps stays empty, seed phase is skipped.
    const sshExec = vi.fn(async (_host: string, _command: string) => ({ code: 0, stdout: "", stderr: "" }));
    const report = await reseedClusterKnownHosts(NODES, { sshExec });
    expect(report.trustedIps).toEqual([]);
    expect(report.perNode).toEqual([]);
    // buildKeyscanSeedScript is never invoked — only gather calls are made (one per node).
    expect(sshExec).toHaveBeenCalledTimes(NODES.length);
  });
});
