import { describe, it, expect } from "vitest";
import { sparkrunAuditCmd, sparkrunSetupCmds, sparkrunPrewarmCmd, sparkrunMeshCmd, SPARKRUN_PKG } from "./provisioner.js";

describe("sparkrun provisioning commands", () => {
  it("pins the sparkrun version", () => {
    expect(SPARKRUN_PKG).toBe("sparkrun==0.2.38");
  });

  it("audit checks sparkrun is runnable via uvx", () => {
    const c = sparkrunAuditCmd();
    expect(c).toContain("uvx");
    expect(c).toContain("sparkrun");
    expect(c).toContain("--version");
  });

  it("setup cmds for a host use non-interactive subcommands with -H", () => {
    const cmds = sparkrunSetupCmds("10.0.0.1");
    const joined = cmds.join("\n");
    expect(joined).toContain("setup install");
    expect(joined).toContain("setup earlyoom");
    expect(joined).toContain("setup docker-group");
    expect(joined).toContain("-H 10.0.0.1");
    expect(joined).not.toContain("wizard"); // never interactive
  });

  it("prewarm builds the image on the host (opt-in, expensive)", () => {
    const c = sparkrunPrewarmCmd("qwen3-1.7b-vllm", "10.0.0.1");
    expect(c).toContain("sparkrun");
    expect(c).toContain("run");
    expect(c).toContain("-H 10.0.0.1");
  });

  it("mesh cmd creates a cluster-wide SSH setup command", () => {
    const c = sparkrunMeshCmd(["10.0.0.1", "10.0.0.2"]);
    expect(c).toContain("setup ssh");
    expect(c).toContain("10.0.0.1");
    expect(c).toContain("10.0.0.2");
  });
});
