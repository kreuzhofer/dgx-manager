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

describe("ollamaInstallCmd", () => {
  const cmd = ollamaInstallCmd("daniel");

  it("disables boot autostart and never enables it", () => {
    expect(cmd).toContain("sudo systemctl disable ollama");
    expect(cmd).not.toContain("systemctl enable ollama");
  });

  it("does not stop the service (run-state for this boot is left as-is)", () => {
    expect(cmd).not.toContain("systemctl stop");
  });

  it("disable runs after restart so the final state is disabled-for-boot", () => {
    const restartIdx = cmd.indexOf("systemctl restart ollama");
    expect(restartIdx).toBeGreaterThan(-1);
    expect(cmd.indexOf("systemctl disable ollama")).toBeGreaterThan(restartIdx);
  });

  it("rejects an sshUser that would break single-quoting in the root tee pipeline", () => {
    expect(() => ollamaInstallCmd("bad'user")).toThrow(/invalid sshUser/);
  });

  it("accepts a normal unix username", () => {
    expect(() => ollamaInstallCmd("ubuntu")).not.toThrow();
  });

  it("keeps the install source and systemd drop-in config unchanged", () => {
    expect(cmd).toContain("https://ollama.ai/install.sh");
    expect(cmd).toContain("OLLAMA_HOST=0.0.0.0");
    expect(cmd).toContain("OLLAMA_MAX_LOADED_MODELS=0");
    expect(cmd).toContain("/etc/systemd/system/ollama.service.d/override.conf");
    expect(cmd).toContain("User=daniel");
    expect(cmd).toContain("Environment=HOME=/home/daniel");
  });
});
