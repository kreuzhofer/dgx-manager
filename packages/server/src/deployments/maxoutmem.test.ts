import { describe, it, expect, vi } from "vitest";
import {
  parseRecipeRef,
  readMaxOutMemCmd,
  parseMaxOutMem,
  reclaimMemoryCmd,
  parseReclaimDetail,
  maxOutMemoryForDeploy,
} from "./maxoutmem.js";

describe("parseRecipeRef", () => {
  it("parses a valid @registry/basename ref", () => {
    expect(parseRecipeRef("@community-kreuzhofer/glm-5.2-awq-15pct-vllm-kreuzhofer")).toEqual({
      registry: "community-kreuzhofer",
      basename: "glm-5.2-awq-15pct-vllm-kreuzhofer",
    });
  });

  it.each([
    ["no-at-sign", "no-at-sign/basename"],
    ["missing slash", "@only"],
    ["shell metachar in basename", "@reg/bad;name"],
    ["path traversal in basename", "@reg/../evil"],
    ["path traversal in registry", "@../evil/basename"],
    ["empty registry", "@/basename"],
  ])("throws on malformed/unsafe ref: %s", (_label, ref) => {
    expect(() => parseRecipeRef(ref)).toThrow();
  });
});

describe("readMaxOutMemCmd", () => {
  it("interpolates the validated registry and basename.yaml", () => {
    const cmd = readMaxOutMemCmd("@community-kreuzhofer/glm-5.2-awq");
    expect(cmd).toContain("community-kreuzhofer");
    expect(cmd).toContain("glm-5.2-awq.yaml");
    expect(cmd).toContain("maxoutmem");
  });

  it("rejects an unsafe ref before building (no metachars leak into the command)", () => {
    expect(() => readMaxOutMemCmd("@reg/bad;name")).toThrow();
  });
});

describe("parseMaxOutMem", () => {
  it.each([
    ["true\n", true],
    [" true ", true],
    ["true", true],
    ["false", false],
    ["false\n", false],
    ["blah", false],
    ["", false],
  ])("%j → %s", (stdout, expected) => {
    expect(parseMaxOutMem(stdout)).toBe(expected);
  });
});

describe("reclaimMemoryCmd", () => {
  const cmd = reclaimMemoryCmd();
  it("stops gdm with non-interactive sudo", () => {
    expect(cmd).toContain("systemctl stop gdm");
    expect(cmd).toContain("sudo -n");
  });
  it("uses best-effort || true semantics (no set -e)", () => {
    expect(cmd).toContain("|| true");
    expect(cmd).not.toContain("set -e");
  });
  it("measures MemAvailable before/after and reports freed_kib + gdm state", () => {
    expect(cmd).toContain("MemAvailable");
    expect(cmd).toContain("freed_kib=");
    expect(cmd).toContain("gdm=");
    expect(cmd).toContain("is-active gdm");
  });
});

describe("parseReclaimDetail", () => {
  it("formats freed_kib as MiB and includes gdm state", () => {
    expect(parseReclaimDetail("reclaimed freed_kib=2064384 gdm=inactive")).toBe(
      "freed 2016 MiB, gdm=inactive",
    );
  });
  it("handles a negative delta (memory pressure rose during reclaim)", () => {
    expect(parseReclaimDetail("reclaimed freed_kib=-1024 gdm=active")).toBe(
      "freed -1 MiB, gdm=active",
    );
  });
  it("reports gdm=unknown when systemctl was unavailable", () => {
    expect(parseReclaimDetail("reclaimed freed_kib=0 gdm=unknown")).toBe(
      "freed 0 MiB, gdm=unknown",
    );
  });
  it("falls back to raw stdout when tokens are absent (e.g. an error msg)", () => {
    expect(parseReclaimDetail("  connection refused  ")).toBe("connection refused");
  });
});

describe("maxOutMemoryForDeploy", () => {
  const ref = "@reg/basename";

  it("flag false → applied:false and reclaim is never called", async () => {
    const seen: string[] = [];
    const sshExec = vi.fn(async (_host: string, command: string) => {
      seen.push(command);
      return { code: 0, stdout: "false", stderr: "" };
    });
    const res = await maxOutMemoryForDeploy({
      recipeRef: ref,
      headIp: "10.0.0.1",
      nodeIps: ["10.0.0.1", "10.0.0.2"],
      sshExec,
    });
    expect(res).toEqual({ applied: false, perNode: [] });
    expect(seen).toHaveLength(1);
    expect(seen[0]).toContain("maxoutmem");
    expect(seen.some((c) => c.includes("systemctl stop gdm"))).toBe(false);
  });

  it("flag true → reclaim called once per node, applied:true, detail parsed", async () => {
    const reclaimHosts: string[] = [];
    const sshExec = vi.fn(async (host: string, command: string) => {
      if (command.includes("maxoutmem")) return { code: 0, stdout: "true", stderr: "" };
      reclaimHosts.push(host);
      return { code: 0, stdout: "reclaimed freed_kib=2064384 gdm=inactive", stderr: "" };
    });
    const res = await maxOutMemoryForDeploy({
      recipeRef: ref,
      headIp: "10.0.0.1",
      nodeIps: ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
      sshExec,
    });
    expect(res.applied).toBe(true);
    expect(res.perNode).toHaveLength(3);
    expect(res.perNode.every((n) => n.ok)).toBe(true);
    expect(res.perNode.every((n) => n.detail === "freed 2016 MiB, gdm=inactive")).toBe(true);
    expect(reclaimHosts).toEqual(["10.0.0.1", "10.0.0.2", "10.0.0.3"]);
  });

  it("a node's sshExec rejecting → that node ok:false, others still attempted, no throw", async () => {
    const reclaimHosts: string[] = [];
    const sshExec = vi.fn(async (host: string, command: string) => {
      if (command.includes("maxoutmem")) return { code: 0, stdout: "true", stderr: "" };
      reclaimHosts.push(host);
      if (host === "10.0.0.2") throw new Error("connection refused");
      return { code: 0, stdout: "reclaimed", stderr: "" };
    });
    const res = await maxOutMemoryForDeploy({
      recipeRef: ref,
      headIp: "10.0.0.1",
      nodeIps: ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
      sshExec,
    });
    expect(res.applied).toBe(true);
    expect(res.perNode).toHaveLength(3);
    const bad = res.perNode.find((n) => n.ip === "10.0.0.2");
    expect(bad?.ok).toBe(false);
    expect(bad?.detail).toContain("connection refused");
    // all three nodes were attempted despite the middle one failing
    expect(reclaimHosts).toEqual(["10.0.0.1", "10.0.0.2", "10.0.0.3"]);
  });

  it("read sshExec rejecting → applied:false, no reclaim calls, no throw", async () => {
    let reclaimCalls = 0;
    const sshExec = vi.fn(async (_host: string, command: string) => {
      if (command.includes("maxoutmem")) throw new Error("ssh unreachable");
      reclaimCalls++;
      return { code: 0, stdout: "reclaimed", stderr: "" };
    });
    const res = await maxOutMemoryForDeploy({
      recipeRef: ref,
      headIp: "10.0.0.1",
      nodeIps: ["10.0.0.1", "10.0.0.2"],
      sshExec,
    });
    expect(res).toEqual({ applied: false, perNode: [] });
    expect(reclaimCalls).toBe(0);
  });

  it("non-zero exit code on a node → ok:false but still applied", async () => {
    const sshExec = vi.fn(async (_host: string, command: string) => {
      if (command.includes("maxoutmem")) return { code: 0, stdout: "true", stderr: "" };
      return { code: 1, stdout: "", stderr: "boom" };
    });
    const res = await maxOutMemoryForDeploy({
      recipeRef: ref,
      headIp: "10.0.0.1",
      nodeIps: ["10.0.0.1"],
      sshExec,
    });
    expect(res.applied).toBe(true);
    expect(res.perNode[0].ok).toBe(false);
    expect(res.perNode[0].detail).toBe("boom");
  });
});
