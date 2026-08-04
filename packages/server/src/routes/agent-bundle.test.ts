/**
 * Tests for the generated agent install script's sudoers whitelist.
 *
 * Background: the agent needs passwordless sudo for exactly the commands it
 * runs — self-update (restart/stop dgx-agent), the on-demand Ollama start
 * that Ollama deploys perform (fleet policy disables Ollama autostart), and
 * the :11434 firewall applied at agent boot (iptables/ip6tables). This used
 * to work only because fleet nodes happened to grant broader NOPASSWD; the
 * provisioned sudoers file itself must guarantee it.
 */
import { describe, it, expect } from "vitest";
import { execFileSync } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { generateInstallScript } from "./agent-bundle.js";

describe("generateInstallScript — sudoers whitelist", () => {
  const script = generateInstallScript("http://192.168.44.36:4000");

  /** The single sudoers line (comma-separated command list) written to
   *  /etc/sudoers.d/dgx-agent. */
  function sudoersLine(): string {
    const line = script.split("\n").find((l) => l.includes("NOPASSWD:"));
    expect(line, "install script must write a NOPASSWD sudoers line").toBeDefined();
    return line!;
  }

  it("whitelists every privileged command the agent runs", () => {
    const line = sudoersLine();
    for (const cmd of [
      "/usr/bin/systemctl restart dgx-agent",
      "/usr/bin/systemctl stop dgx-agent",
      "/usr/bin/systemctl start ollama",
      "/usr/sbin/iptables",
      "/usr/sbin/ip6tables",
    ]) {
      expect(line).toContain(cmd);
    }
  });

  /** Least privilege: the systemctl grants must stay verb+unit-scoped —
   *  a bare "/usr/bin/systemctl" entry would allow ANY systemctl action. */
  it("does not grant unrestricted systemctl", () => {
    const line = sudoersLine();
    expect(line).not.toMatch(/\/usr\/bin\/systemctl\s*(,|"|$)/);
  });
});

/**
 * Tests for the Ollama systemd drop-in the install script writes.
 *
 * Incident (2026-07-28): agenthost was unreachable on :11434 because the
 * drop-in that sets OLLAMA_HOST=0.0.0.0 was written ONLY in the branch that
 * installs Ollama. agenthost already had Ollama, so the script logged
 * "already installed" and wrote nothing — leaving Ollama on its default
 * loopback bind. The deployment still reported "running" (the model loads
 * fine); it was simply unreachable by the manager.
 *
 * These run the real shell emitted by generateInstallScript, sliced out
 * between its `# >>> ollama-dropin` markers, against stubbed systemctl /
 * curl / mountpoint — so they test behaviour, not the wording of the script.
 */
describe("generateInstallScript — Ollama drop-in", () => {
  const script = generateInstallScript("http://192.168.44.36:4000");

  function ollamaSection(): string {
    const m = script.match(/# >>> ollama-dropin\n([\s\S]*?)# <<< ollama-dropin/);
    expect(m, "install script must delimit the ollama drop-in section").not.toBeNull();
    return m![1];
  }

  interface RunResult { override: string | null; calls: string[] }

  /** Execute the drop-in section with stubbed system commands. */
  function run(opts: { preexisting: boolean; tankMounted: boolean; dir?: string }): RunResult {
    const tmp = opts.dir ?? mkdtempSync(join(tmpdir(), "ollama-dropin-"));
    const bin = join(tmp, "bin");
    mkdirSync(bin, { recursive: true });
    const calls = join(tmp, "calls.log");
    const dropin = join(tmp, "dropin");
    writeFileSync(calls, ""); // truncate: the idempotency test reuses the dir

    const stub = (name: string, body: string) => {
      const p = join(bin, name);
      writeFileSync(p, `#!/bin/sh\necho "${name} $*" >> "${calls}"\n${body}\n`);
      chmodSync(p, 0o755);
    };
    stub("systemctl", opts.preexisting ? "exit 0" : `case "$1" in list-unit-files) exit 1 ;; esac\nexit 0`);
    if (opts.preexisting) stub("ollama", "exit 0");
    stub("curl", "exit 0");
    stub("mountpoint", opts.tankMounted ? "exit 0" : "exit 1");

    writeFileSync(join(tmp, "section.sh"), ollamaSection());
    execFileSync("bash", ["-c", `set -eu; log() { :; }; . "${join(tmp, "section.sh")}"`], {
      // Closed PATH: the host has its own ollama, which would make every
      // node look already-installed.
      env: { ...process.env, PATH: `${bin}:/usr/bin:/bin`, AGENT_USER: "testuser", OLLAMA_DROPIN_DIR: dropin },
      stdio: "pipe",
    });

    const overridePath = join(dropin, "override.conf");
    return {
      override: existsSync(overridePath) ? readFileSync(overridePath, "utf8") : null,
      calls: existsSync(calls) ? readFileSync(calls, "utf8").trim().split("\n") : [],
    };
  }

  // THE REGRESSION: this is the exact agenthost case.
  it("writes the drop-in even when Ollama is already installed", () => {
    const r = run({ preexisting: true, tankMounted: false });
    expect(r.override).not.toBeNull();
    expect(r.override).toContain("Environment=OLLAMA_HOST=0.0.0.0");
  });

  /** Adopting an existing Ollama's identity would repoint it at a different
   *  HOME and orphan its model store — agenthost runs as User=ollama with
   *  4.4 GB under /usr/share/ollama. Only claim the service we installed. */
  it("does not hijack a pre-existing Ollama's user or model store", () => {
    const r = run({ preexisting: true, tankMounted: true });
    expect(r.override).not.toContain("User=");
    expect(r.override).not.toContain("OLLAMA_MODELS");
  });

  it("claims user and shared model store when it installs Ollama itself", () => {
    const r = run({ preexisting: false, tankMounted: false });
    expect(r.override).toContain("User=testuser");
    expect(r.override).toContain("Environment=HOME=/home/testuser");
    expect(r.override).toContain("Environment=OLLAMA_HOST=0.0.0.0");
  });

  /** Re-running the installer must not bounce a live Ollama serving a
   *  deployment — restart only when the drop-in actually changed. */
  it("does not restart Ollama when the drop-in is already current", () => {
    const tmp = mkdtempSync(join(tmpdir(), "ollama-dropin-idem-"));
    const first = run({ preexisting: true, tankMounted: false, dir: tmp });
    expect(first.calls.some((c) => c.startsWith("systemctl restart"))).toBe(true);
    const second = run({ preexisting: true, tankMounted: false, dir: tmp });
    expect(second.override).toBe(first.override);
    expect(second.calls.some((c) => c.startsWith("systemctl restart"))).toBe(false);
  });
});
