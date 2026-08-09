import { describe, it, expect } from "vitest";
import { sparkrunAuditCmd, sparkrunSetupCmds, sparkrunPrewarmCmd, sparkrunMeshCmd, SPARKRUN_PKG } from "./provisioner.js";

describe("sparkrun provisioning commands", () => {
  // A floor rather than an equality pin: `==0.2.38` went stale and cost us a
  // misfiled upstream bug (recipe-registry#20). The floor must stay at or above
  // 0.3.3, the first version that unescapes `{{...}}` in a recipe command and
  // resolves placeholders nested inside JSON-valued flags.
  it("constrains sparkrun to a floor of at least 0.3.3, not an exact pin", () => {
    const m = /^sparkrun>=(\d+)\.(\d+)\.(\d+)$/.exec(SPARKRUN_PKG);
    expect(m, `expected a >= constraint, got ${SPARKRUN_PKG}`).not.toBeNull();

    const [major, minor, patch] = m!.slice(1).map(Number);
    const floor = major > 0 || (major === 0 && (minor > 3 || (minor === 3 && patch >= 3)));

    expect(floor, `floor ${major}.${minor}.${patch} is below 0.3.3`).toBe(true);
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
