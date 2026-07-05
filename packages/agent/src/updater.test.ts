import { describe, it, expect } from "vitest";
import { verifyExtractedBundle, healthCheckPasses, runUpdate, atomicSwap, atomicRollback, type UpdateDeps } from "./updater.js";

describe("verifyExtractedBundle", () => {
  it("ok when package.json version matches", () => {
    const r = verifyExtractedBundle("/opt/dgx-agent-new", "0.5.720", () => JSON.stringify({ version: "0.5.720" }));
    expect(r.ok).toBe(true);
  });
  it("fails on version mismatch", () => {
    const r = verifyExtractedBundle("/d", "0.5.720", () => JSON.stringify({ version: "0.5.719" }));
    expect(r.ok).toBe(false);
    expect(r.reason).toMatch(/version/i);
  });
  it("fails when package.json unreadable/unparseable", () => {
    expect(verifyExtractedBundle("/d", "0.5.720", () => { throw new Error("ENOENT"); }).ok).toBe(false);
    expect(verifyExtractedBundle("/d", "0.5.720", () => "not json").ok).toBe(false);
  });
});

describe("healthCheckPasses", () => {
  it("true when marker written after restart", () => {
    expect(healthCheckPasses(2000, 1000, 90000)).toBe(true);
  });
  it("false when marker missing", () => {
    expect(healthCheckPasses(null, 1000, 90000)).toBe(false);
  });
  it("false when marker is stale (pre-restart)", () => {
    expect(healthCheckPasses(500, 1000, 90000)).toBe(false);
  });
});

describe("atomicSwap", () => {
  it("happy path: runs the 3 commands in order, no restore", () => {
    const calls: string[] = [];
    atomicSwap((cmd) => { calls.push(cmd); });
    expect(calls).toEqual([
      "sudo rm -rf /opt/dgx-agent-old",
      "sudo mv /opt/dgx-agent /opt/dgx-agent-old",
      "sudo mv /opt/dgx-agent-new /opt/dgx-agent",
    ]);
  });

  it("when installing the new dir fails, restores old -> current and rethrows", () => {
    const calls: string[] = [];
    const run = (cmd: string) => {
      calls.push(cmd);
      if (cmd === "sudo mv /opt/dgx-agent-new /opt/dgx-agent") throw new Error("mv: no such file");
    };
    expect(() => atomicSwap(run)).toThrow(/swap failed, restored previous agent/);
    expect(calls).toEqual([
      "sudo rm -rf /opt/dgx-agent-old",
      "sudo mv /opt/dgx-agent /opt/dgx-agent-old",
      "sudo mv /opt/dgx-agent-new /opt/dgx-agent",
      "sudo mv /opt/dgx-agent-old /opt/dgx-agent",
    ]);
  });

  it("when both the install mv AND the restore mv fail, rethrows a truthful double-failure message", () => {
    const calls: string[] = [];
    const run = (cmd: string) => {
      calls.push(cmd);
      if (cmd === "sudo mv /opt/dgx-agent-new /opt/dgx-agent") throw new Error("mv: no such file");
      if (cmd === "sudo mv /opt/dgx-agent-old /opt/dgx-agent") throw new Error("mv: old is gone too");
    };
    expect(() => atomicSwap(run)).toThrow(/swap failed AND restore failed.*may be missing/);
  });
});

describe("atomicRollback", () => {
  it("stashes the bad current, restores old, then restarts, in order", () => {
    const calls: string[] = [];
    atomicRollback((cmd) => { calls.push(cmd); });
    expect(calls).toEqual([
      "sudo rm -rf /opt/dgx-agent-failed",
      "sudo mv /opt/dgx-agent /opt/dgx-agent-failed",
      "sudo mv /opt/dgx-agent-old /opt/dgx-agent",
      "sudo systemctl restart dgx-agent",
    ]);
  });
});

function makeDeps(over: Partial<UpdateDeps> & { connectAt?: number }): { deps: UpdateDeps; calls: string[]; result: any } {
  const calls: string[] = []; let t = 1000; let result: any = null;
  const base: UpdateDeps = {
    download: async () => { calls.push("download"); },
    extract: async () => { calls.push("extract"); },
    verify: () => { calls.push("verify"); return { ok: true }; },
    preserveNodeId: () => calls.push("preserveNodeId"),
    swap: () => calls.push("swap"),
    restart: () => calls.push("restart"),
    checkConnected: () => (over.connectAt != null && t >= over.connectAt ? t : null),
    rollback: () => calls.push("rollback"),
    writeResult: (r) => { result = r; },
    log: () => {},
    now: () => t,
    sleep: async (ms) => { t += ms; },  // advance fake clock
  };
  return { deps: { ...base, ...over }, calls, get result() { return result; } } as any;
}

describe("runUpdate", () => {
  it("happy path: swap+restart, marker fresh -> success (no rollback)", async () => {
    const h = makeDeps({ connectAt: 1000 }); // marker present immediately >= restart time
    await runUpdate({ bundleUrl: "u", version: "1.0.0" }, h.deps);
    expect(h.calls).toEqual(["download", "extract", "verify", "preserveNodeId", "swap", "restart"]);
    expect(h.result.outcome).toBe("success");
    expect(h.calls).not.toContain("rollback");
  });
  it("verify fail: aborts BEFORE swap (old agent untouched)", async () => {
    const h = makeDeps({ verify: () => ({ ok: false, reason: "bad" }) });
    await runUpdate({ bundleUrl: "u", version: "1.0.0" }, h.deps);
    expect(h.calls).not.toContain("swap");
    expect(h.calls).not.toContain("restart");
    expect(h.result.outcome).toBe("failed");
  });
  it("health fail: swap+restart then no reconnect -> rollback", async () => {
    const h = makeDeps({ checkConnected: () => null }); // never connects
    await runUpdate({ bundleUrl: "u", version: "1.0.0" }, h.deps);
    expect(h.calls).toContain("swap");
    expect(h.calls).toContain("rollback");
    expect(h.result.outcome).toBe("rolled-back");
  });
  it("download fail: aborts, no swap, failed result", async () => {
    const h = makeDeps({ download: async () => { throw new Error("net"); } });
    await runUpdate({ bundleUrl: "u", version: "1.0.0" }, h.deps);
    expect(h.calls).not.toContain("swap");
    expect(h.result.outcome).toBe("failed");
  });
  it("health fail + rollback itself throws -> outcome is rollback-failed (not rolled-back), writeResult called once", async () => {
    let writeResultCalls = 0;
    const h = makeDeps({
      checkConnected: () => null, // never connects
      rollback: () => { throw new Error("mv: old agent gone"); },
    });
    const origWriteResult = h.deps.writeResult;
    h.deps.writeResult = (r) => { writeResultCalls++; origWriteResult(r); };
    await runUpdate({ bundleUrl: "u", version: "1.0.0" }, h.deps);
    expect(h.result.outcome).toBe("rollback-failed");
    expect(h.result.outcome).not.toBe("rolled-back");
    expect(writeResultCalls).toBe(1);
  });
});
