import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  parseManagerHost,
  buildFirewallSteps,
  applyOllamaFirewall,
  type FirewallStep,
  type FirewallExec,
} from "./firewall.js";

describe("parseManagerHost", () => {
  it("extracts an IPv4 host from a ws:// URL with port and path", () => {
    expect(parseManagerHost("ws://192.168.44.14:4000/ws/agent")).toBe("192.168.44.14");
  });

  it("extracts the host from a wss:// URL", () => {
    expect(parseManagerHost("wss://manager.example.com/ws/agent")).toBe("manager.example.com");
  });

  it("extracts a bare hostname without port", () => {
    expect(parseManagerHost("ws://dgx-manager/ws/agent")).toBe("dgx-manager");
  });

  it("throws on an empty string", () => {
    expect(() => parseManagerHost("")).toThrow();
  });

  it("throws on a non-URL string", () => {
    expect(() => parseManagerHost("not a url at all")).toThrow();
  });

  it("throws on a URL with no host", () => {
    // "ws://" fails WHATWG parsing (special scheme needs a host);
    // a relative path is not a URL at all.
    expect(() => parseManagerHost("ws://")).toThrow();
    expect(() => parseManagerHost("/ws/agent")).toThrow();
  });
});

describe("buildFirewallSteps", () => {
  it("builds the exact ordered v4 + v6 step list for a manager IP", () => {
    const steps = buildFirewallSteps("192.168.44.14");
    const expected: FirewallStep[] = [
      // IPv4: manager + loopback allowed, everything else dropped
      { kind: "tolerate", argv: ["iptables", "-N", "DGX_AGENT_OLLAMA"] },
      { kind: "run", argv: ["iptables", "-F", "DGX_AGENT_OLLAMA"] },
      { kind: "run", argv: ["iptables", "-A", "DGX_AGENT_OLLAMA", "-s", "192.168.44.14", "-j", "ACCEPT"] },
      { kind: "run", argv: ["iptables", "-A", "DGX_AGENT_OLLAMA", "-i", "lo", "-j", "ACCEPT"] },
      { kind: "run", argv: ["iptables", "-A", "DGX_AGENT_OLLAMA", "-j", "DROP"] },
      {
        kind: "check-then",
        check: ["iptables", "-C", "INPUT", "-p", "tcp", "--dport", "11434", "-j", "DGX_AGENT_OLLAMA"],
        then: ["iptables", "-I", "INPUT", "-p", "tcp", "--dport", "11434", "-j", "DGX_AGENT_OLLAMA"],
      },
      // IPv6: loopback-only (the manager is IPv4-only)
      { kind: "tolerate", argv: ["ip6tables", "-N", "DGX_AGENT_OLLAMA"] },
      { kind: "run", argv: ["ip6tables", "-F", "DGX_AGENT_OLLAMA"] },
      { kind: "run", argv: ["ip6tables", "-A", "DGX_AGENT_OLLAMA", "-i", "lo", "-j", "ACCEPT"] },
      { kind: "run", argv: ["ip6tables", "-A", "DGX_AGENT_OLLAMA", "-j", "DROP"] },
      {
        kind: "check-then",
        check: ["ip6tables", "-C", "INPUT", "-p", "tcp", "--dport", "11434", "-j", "DGX_AGENT_OLLAMA"],
        then: ["ip6tables", "-I", "INPUT", "-p", "tcp", "--dport", "11434", "-j", "DGX_AGENT_OLLAMA"],
      },
    ];
    expect(steps).toEqual(expected);
  });

  it("rejects shell-metacharacter injection attempts in the host", () => {
    expect(() => buildFirewallSteps("1.2.3.4; rm -rf /")).toThrow(/host/i);
    expect(() => buildFirewallSteps("$(reboot)")).toThrow(/host/i);
    expect(() => buildFirewallSteps("1.2.3.4 -j ACCEPT")).toThrow(/host/i);
  });

  it("rejects an empty host", () => {
    expect(() => buildFirewallSteps("")).toThrow(/host/i);
  });

  it("accepts a plain hostname", () => {
    const steps = buildFirewallSteps("dgx-manager.local");
    expect(steps.some((s) => s.kind === "run" && s.argv.includes("dgx-manager.local"))).toBe(true);
  });
});

describe("applyOllamaFirewall", () => {
  /** Fake exec that records sudo invocations and fails for argv sequences
   *  matched by `failWhen`. */
  function fakeExec(failWhen: (args: string[]) => boolean = () => false) {
    const calls: string[][] = [];
    const exec: FirewallExec = async (cmd, args) => {
      calls.push([cmd, ...args]);
      if (failWhen(args)) {
        const err = new Error("exit 1") as Error & { stderr?: string };
        err.stderr = "iptables: simulated failure.";
        throw err;
      }
    };
    return { calls, exec };
  }

  let errorSpy: ReturnType<typeof vi.spyOn>;
  let logSpy: ReturnType<typeof vi.spyOn>;
  beforeEach(() => {
    errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
  });
  afterEach(() => {
    errorSpy.mockRestore();
    logSpy.mockRestore();
  });

  it("runs every command via `sudo -n` and skips -I when -C succeeds", async () => {
    const { calls, exec } = fakeExec();
    const ok = await applyOllamaFirewall("ws://192.168.44.14:4000/ws/agent", exec);
    expect(ok).toBe(true);
    // Every invocation goes through sudo -n
    for (const call of calls) {
      expect(call[0]).toBe("sudo");
      expect(call[1]).toBe("-n");
    }
    // -C succeeded for both families → no -I anywhere
    expect(calls.some((c) => c.includes("-I"))).toBe(false);
    // Both -C checks ran
    expect(calls.filter((c) => c.includes("-C")).length).toBe(2);
  });

  it("inserts the INPUT jump when the -C check fails", async () => {
    const { calls, exec } = fakeExec((args) => args.includes("-C"));
    const ok = await applyOllamaFirewall("ws://192.168.44.14:4000/ws/agent", exec);
    expect(ok).toBe(true);
    const inserts = calls.filter((c) => c.includes("-I"));
    expect(inserts).toEqual([
      ["sudo", "-n", "iptables", "-I", "INPUT", "-p", "tcp", "--dport", "11434", "-j", "DGX_AGENT_OLLAMA"],
      ["sudo", "-n", "ip6tables", "-I", "INPUT", "-p", "tcp", "--dport", "11434", "-j", "DGX_AGENT_OLLAMA"],
    ]);
  });

  it("tolerates -N failing (chain already exists) and continues", async () => {
    const { calls, exec } = fakeExec((args) => args.includes("-N"));
    const ok = await applyOllamaFirewall("ws://192.168.44.14:4000/ws/agent", exec);
    expect(ok).toBe(true);
    // -F still ran for both families after the tolerated -N failure
    expect(calls.filter((c) => c.includes("-F")).length).toBe(2);
    expect(errorSpy).not.toHaveBeenCalled();
  });

  it("logs loudly and returns false — without throwing — when a required command fails", async () => {
    // args as seen by exec: ["-n", "iptables", "-F", ...] — fail only v4 -F
    const { calls, exec } = fakeExec((args) => args.includes("-F") && args[1] === "iptables");
    const ok = await applyOllamaFirewall("ws://192.168.44.14:4000/ws/agent", exec);
    expect(ok).toBe(false);
    expect(errorSpy).toHaveBeenCalled();
    const msg = errorSpy.mock.calls.map((c) => c.join(" ")).join("\n");
    expect(msg).toContain("[firewall]");
    expect(msg).toContain("iptables -F DGX_AGENT_OLLAMA");
    expect(msg).toContain("simulated failure");
    // Aborts after the hard failure — no further rule commands
    const last = calls[calls.length - 1];
    expect(last).toEqual(["sudo", "-n", "iptables", "-F", "DGX_AGENT_OLLAMA"]);
  });

  it("logs loudly and returns false on an unparseable MANAGER_URL, running nothing", async () => {
    const { calls, exec } = fakeExec();
    const ok = await applyOllamaFirewall("not a url", exec);
    expect(ok).toBe(false);
    expect(calls).toEqual([]);
    expect(errorSpy).toHaveBeenCalled();
    expect(errorSpy.mock.calls[0][0]).toContain("[firewall]");
  });
});
