import { describe, it, expect } from "vitest";
import { ollamaAuditCmd, evalOllamaAudit, ollamaInstallCmd } from "./provisioner.js";

describe("ollama audit command", () => {
  it("detects the binary, not the HTTP API (must work with a stopped service)", () => {
    const c = ollamaAuditCmd();
    expect(c).toContain("command -v ollama");
    expect(c).toContain("ollama --version");
    expect(c).not.toContain("11434"); // never require a running server
    expect(c).not.toContain("curl");
  });

  it("also reports the systemd run-state for the detail string", () => {
    expect(ollamaAuditCmd()).toContain("systemctl is-active ollama");
  });
});

describe("evalOllamaAudit", () => {
  it("installed + running -> green with version and running detail", () => {
    const check = evalOllamaAudit("0.20.3\nactive\n");
    expect(check).toEqual({
      name: "Ollama",
      status: "green",
      detail: "v0.20.3 (service running)",
    });
  });

  it("installed + stopped -> still green (fleet policy: autostart disabled)", () => {
    const check = evalOllamaAudit("0.20.3\ninactive\n");
    expect(check.status).toBe("green");
    expect(check.detail).toBe("v0.20.3 (service stopped)");
  });

  it("not installed (empty output) -> yellow, auto-installable", () => {
    const check = evalOllamaAudit("");
    expect(check).toEqual({
      name: "Ollama",
      status: "yellow",
      detail: "Not installed — can auto-install",
    });
  });

  it("whitespace-only output counts as not installed", () => {
    expect(evalOllamaAudit("  \n\n").status).toBe("yellow");
  });

  it("binary present but version unparseable -> green with unknown version", () => {
    const check = evalOllamaAudit("inactive\n");
    expect(check.status).toBe("green");
    expect(check.detail).toBe("installed (version unknown) (service stopped)");
  });
});

/**
 * What the emitted command *does* — which drop-in a node ends up with, and
 * whether the service is bounced — is covered by executing it against stubs in
 * provisioner.ollama-dropin.test.ts. What remains here is the handful of
 * properties worth pinning at the string level.
 */
describe("ollamaInstallCmd", () => {
  const cmd = ollamaInstallCmd("daniel");

  it("targets the real systemd drop-in directory by default", () => {
    expect(cmd).toContain("/etc/systemd/system/ollama.service.d");
  });

  it("does not stop the service (run-state for this boot is left as-is)", () => {
    expect(cmd).not.toContain("systemctl stop");
  });

  it("disable runs after restart so the final state is disabled-for-boot", () => {
    const restartIdx = cmd.indexOf("systemctl restart ollama");
    expect(restartIdx).toBeGreaterThan(-1);
    expect(cmd.indexOf("systemctl disable ollama")).toBeGreaterThan(restartIdx);
  });

  it("keeps the install source unchanged", () => {
    expect(cmd).toContain("https://ollama.ai/install.sh");
  });

  // A failure mid-script must not leave the node half-configured with the
  // remaining steps still running.
  it("aborts on the first failing step", () => {
    expect(cmd.startsWith("set -eu")).toBe(true);
  });
});
