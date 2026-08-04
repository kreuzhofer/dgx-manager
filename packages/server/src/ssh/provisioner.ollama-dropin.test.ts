import { describe, it, expect } from "vitest";
import { execFileSync } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ollamaInstallCmd } from "./provisioner.js";

/**
 * Tests for the Ollama systemd drop-in the SSH provisioner writes.
 *
 * The bug (#11): this path wrote `User=`, `HOME=` and `OLLAMA_MODELS=` on
 * EVERY run. Provisioning a node that already had a hand-installed Ollama
 * therefore repointed it at a different model store — on agenthost that meant
 * a 4.4 GB store at /usr/share/ollama becoming invisible, and the embedding
 * deployment serving from it going dark. The token install-script path had
 * guarded against exactly this for some time; the lesson was never carried
 * across.
 *
 * These run the real shell the provisioner emits, against stubbed systemctl /
 * curl / mountpoint / tee, so they test what the node ends up with rather than
 * the wording of the command string.
 */
describe("ollamaInstallCmd — drop-in behaviour", () => {
  interface RunResult {
    override: string | null;
    calls: string[];
  }

  /**
   * Execute the emitted command with stubbed system commands.
   *
   * `sudo` is stubbed to exec its arguments, so the command runs unprivileged
   * while still exercising the real control flow; `systemctl` reports whether
   * a pre-existing Ollama is present, and `mountpoint` whether shared storage
   * is mounted.
   */
  function run(opts: {
    preexisting: boolean;
    tankMounted: boolean;
    dir?: string;
    sshUser?: string;
  }): RunResult {
    const tmp = opts.dir ?? mkdtempSync(join(tmpdir(), "prov-ollama-"));
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
    stub("systemctl", opts.preexisting ? "exit 0" : `case "$1" in list-unit-files|is-enabled) exit 1 ;; esac\nexit 0`);
    stub("curl", "exit 0");
    stub("sh", 'exit 0'); // the piped installer
    stub("mountpoint", opts.tankMounted ? "exit 0" : "exit 1");
    // Run the privileged part as ourselves so the test needs no root.
    stub("sudo", 'exec "$@"');
    stub("chown", "exit 0");

    const cmd = ollamaInstallCmd(opts.sshUser ?? "testuser");
    writeFileSync(join(tmp, "cmd.sh"), cmd);
    execFileSync("bash", [join(tmp, "cmd.sh")], {
      env: {
        ...process.env,
        PATH: `${bin}:${process.env.PATH}`,
        OLLAMA_DROPIN_DIR: dropin,
        SHARED_STORAGE_DIR: join(tmp, "tank"),
      },
      stdio: "pipe",
    });

    const overridePath = join(dropin, "override.conf");
    return {
      override: existsSync(overridePath) ? readFileSync(overridePath, "utf8") : null,
      calls: existsSync(calls) ? readFileSync(calls, "utf8").trim().split("\n").filter(Boolean) : [],
    };
  }

  // THE REGRESSION: the agenthost case. Its Ollama runs as the `ollama` system
  // user with a 4.4 GB store in the default location; claiming User/HOME or
  // repointing OLLAMA_MODELS orphans that store.
  it("never claims User, HOME or OLLAMA_MODELS on a pre-existing Ollama", () => {
    const r = run({ preexisting: true, tankMounted: true });
    expect(r.override).not.toBeNull();
    expect(r.override).not.toContain("User=");
    expect(r.override).not.toContain("HOME=");
    expect(r.override).not.toContain("OLLAMA_MODELS");
  });

  // The settings that are safe to impose on someone else's service: they change
  // how it is reached and how it manages memory, not where its data lives.
  it("still applies the reachability and memory settings to a pre-existing Ollama", () => {
    const r = run({ preexisting: true, tankMounted: false });
    expect(r.override).toContain("Environment=OLLAMA_HOST=0.0.0.0");
    expect(r.override).toContain("Environment=OLLAMA_MAX_LOADED_MODELS=0");
    // An evicted model cannot be woken through the gateway, so never unload one.
    expect(r.override).toContain("Environment=OLLAMA_KEEP_ALIVE=-1");
  });

  it("does not run the installer when Ollama is already present", () => {
    const r = run({ preexisting: true, tankMounted: false });
    expect(r.calls.some((c) => c.startsWith("curl"))).toBe(false);
  });

  it("claims User, HOME and the shared model store for an Ollama it installs itself", () => {
    const r = run({ preexisting: false, tankMounted: true, sshUser: "daniel" });
    expect(r.override).toContain("User=daniel");
    expect(r.override).toContain("Environment=HOME=/home/daniel");
    expect(r.override).toContain("Environment=OLLAMA_MODELS=");
    expect(r.override).toContain("Environment=OLLAMA_HOST=0.0.0.0");
    expect(r.override).toContain("Environment=OLLAMA_KEEP_ALIVE=-1");
  });

  it("installs Ollama when it is absent", () => {
    const r = run({ preexisting: false, tankMounted: false });
    expect(r.calls.some((c) => c.startsWith("curl"))).toBe(true);
  });

  // Pointing OLLAMA_MODELS at a path that is not mounted makes every model on
  // the node invisible — the failure this ticket is about, in its other form.
  it("omits OLLAMA_MODELS when shared storage is not mounted", () => {
    const r = run({ preexisting: false, tankMounted: false });
    expect(r.override).not.toContain("OLLAMA_MODELS");
    // …but still configures everything that does not depend on the mount.
    expect(r.override).toContain("User=testuser");
    expect(r.override).toContain("Environment=OLLAMA_HOST=0.0.0.0");
  });

  // Fleet policy: Ollama's API is unauthenticated, so it must never come back
  // on its own at boot.
  it("leaves boot autostart disabled", () => {
    const r = run({ preexisting: false, tankMounted: true });
    expect(r.calls).toContain("systemctl disable ollama");
    expect(r.calls.some((c) => c === "systemctl enable ollama")).toBe(false);
  });

  it("does not stop the service — this boot's run-state is left alone", () => {
    const r = run({ preexisting: true, tankMounted: false });
    expect(r.calls.some((c) => c.startsWith("systemctl stop"))).toBe(false);
  });

  // Re-provisioning must not bounce an Ollama that is currently serving a
  // deployment, so a restart happens only when the drop-in actually changes.
  it("does not restart when the drop-in is already current", () => {
    const dir = mkdtempSync(join(tmpdir(), "prov-ollama-idem-"));
    const first = run({ preexisting: true, tankMounted: false, dir });
    expect(first.calls).toContain("systemctl restart ollama");

    const second = run({ preexisting: true, tankMounted: false, dir });
    expect(second.override).toBe(first.override);
    expect(second.calls.some((c) => c === "systemctl restart ollama")).toBe(false);
  });

  it("restarts when the drop-in content changes", () => {
    const dir = mkdtempSync(join(tmpdir(), "prov-ollama-change-"));
    run({ preexisting: true, tankMounted: false, dir });
    // A previously-adopted Ollama that we now install ourselves gains
    // User/HOME, so the file differs and the service must pick it up.
    const second = run({ preexisting: false, tankMounted: false, dir });
    expect(second.calls).toContain("systemctl restart ollama");
  });
});

describe("ollamaInstallCmd — argument safety", () => {
  it("rejects an sshUser that would break quoting in the emitted script", () => {
    expect(() => ollamaInstallCmd("bad'user")).toThrow(/invalid sshUser/);
    expect(() => ollamaInstallCmd("a b")).toThrow(/invalid sshUser/);
    expect(() => ollamaInstallCmd("$(whoami)")).toThrow(/invalid sshUser/);
  });

  it("accepts a normal unix username", () => {
    expect(() => ollamaInstallCmd("ubuntu")).not.toThrow();
    expect(() => ollamaInstallCmd("daniel")).not.toThrow();
  });
});
