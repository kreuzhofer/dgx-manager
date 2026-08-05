import { describe, it, expect } from "vitest";
import { execFileSync } from "node:child_process";
import { chmodSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  ollamaDropInBody,
  OLLAMA_MANAGED_MARKER,
  OLLAMA_OWNERSHIP_SETTINGS,
  OLLAMA_UNIVERSAL_SETTINGS,
} from "./dropin.js";
import { ollamaInstallCmd } from "../ssh/provisioner.js";
import { generateInstallScript } from "../routes/agent-bundle.js";

/** Render the body by executing it, so the assertions are about what a node gets. */
function render(opts: {
  preexisting: boolean;
  mounted: boolean;
  userExpr?: string;
  storageExpr?: string;
}): string {
  const tmp = mkdtempSync(join(tmpdir(), "dropin-"));
  const bin = join(tmp, "bin");
  mkdirSync(bin, { recursive: true });
  const stub = (name: string, body: string) => {
    const p = join(bin, name);
    writeFileSync(p, `#!/bin/sh\n${body}\n`);
    chmodSync(p, 0o755);
  };
  stub("mountpoint", opts.mounted ? "exit 0" : "exit 1");
  stub("mkdir", "exit 0");
  stub("chown", "exit 0");

  const body = ollamaDropInBody({
    userExpr: opts.userExpr ?? "someuser",
    storageExpr: opts.storageExpr ?? "/mnt/tank",
  });
  writeFileSync(join(tmp, "body.sh"), body);
  return execFileSync("bash", ["-c", `. "${join(tmp, "body.sh")}"`], {
    env: { ...process.env, PATH: `${bin}:${process.env.PATH}`, OLLAMA_MANAGED: opts.preexisting ? "0" : "1" },
    encoding: "utf8",
  });
}

describe("ollamaDropInBody", () => {
  it("applies the universal settings whoever installed Ollama", () => {
    for (const preexisting of [true, false]) {
      const out = render({ preexisting, mounted: true });
      for (const setting of OLLAMA_UNIVERSAL_SETTINGS) {
        expect(out, `preexisting=${preexisting}`).toContain(setting);
      }
    }
  });

  // The heart of #11: adopting someone else's Ollama must not move its data.
  it("claims nothing that decides where data lives from a pre-existing Ollama", () => {
    const out = render({ preexisting: true, mounted: true });
    for (const setting of OLLAMA_OWNERSHIP_SETTINGS) {
      expect(out).not.toContain(setting);
    }
  });

  it("claims ownership settings for an Ollama the manager installed", () => {
    const out = render({ preexisting: false, mounted: true, userExpr: "daniel" });
    expect(out).toContain("User=daniel");
    expect(out).toContain("Environment=HOME=/home/daniel");
    expect(out).toContain("Environment=OLLAMA_MODELS=/mnt/tank/models/ollama");
  });

  it("omits the model store when shared storage is not mounted", () => {
    const out = render({ preexisting: false, mounted: false });
    expect(out).not.toContain("OLLAMA_MODELS");
    expect(out).toContain("User=someuser"); // everything else still applied
  });

  // The renderer owns the quoting, so a storage path containing a space must
  // survive intact rather than word-splitting into a broken Environment= line.
  it("quotes the storage path so a space cannot split it", () => {
    const out = render({
      preexisting: false,
      mounted: true,
      storageExpr: "/mnt/my tank",
    });
    expect(out).toContain("Environment=OLLAMA_MODELS=/mnt/my tank/models/ollama");
  });

  it("always opens with the systemd section header", () => {
    expect(render({ preexisting: true, mounted: false }).trimStart()).toMatch(/^\[Service\]/);
  });
});

/**
 * The reason this module exists. The SSH provisioner and the token install
 * script must impose the same policy; expressed separately they drifted twice.
 * This asserts they still render from the one source — a third drift fails here
 * rather than on a node.
 */
describe("both provisioning paths render the same policy", () => {
  const provisioner = ollamaInstallCmd("daniel");
  const installScript = generateInstallScript("http://192.168.44.14:4000");
  const paths: Array<[string, string]> = [
    ["SSH provisioner", provisioner],
    ["install script", installScript],
  ];

  /**
   * Extract each path's drop-in body and execute it, so the comparison is
   * between what the two paths actually WRITE. An earlier version of this test
   * matched the generated text with a regex, which passed while the paths still
   * disagreed — the failure mode it existed to prevent.
   */
  function bodyOutput(script: string, managed: boolean): string {
    const m = script.match(/\{\n\s*echo "\[Service\]"[\s\S]*?\n\}/);
    expect(m, "each path must contain a drop-in body").not.toBeNull();
    const tmp = mkdtempSync(join(tmpdir(), "policy-"));
    const bin = join(tmp, "bin");
    mkdirSync(bin, { recursive: true });
    for (const [name, body] of [
      ["mountpoint", "exit 0"],
      ["mkdir", "exit 0"],
      ["chown", "exit 0"],
      ["sudo", 'exec "$@"'],
    ] as const) {
      const p = join(bin, name);
      writeFileSync(p, `#!/bin/sh\n${body}\n`);
      chmodSync(p, 0o755);
    }
    writeFileSync(join(tmp, "b.sh"), m![0]);
    return execFileSync("bash", ["-c", `. "${join(tmp, "b.sh")}"`], {
      env: {
        ...process.env,
        PATH: `${bin}:/usr/bin:/bin`,
        OLLAMA_MANAGED: managed ? "1" : "0",
        AGENT_USER: "daniel",
        // Each path names the storage root its own way (the provisioner from
        // SHARED_STORAGE, the install script hardcoded). That is configuration,
        // not policy — normalise it so this compares the policy.
        STORAGE: "/mnt/tank",
      },
      encoding: "utf8",
    });
  }

  it("write the same settings when adopting someone else's Ollama", () => {
    const [a, b] = paths.map(([, s]) => bodyOutput(s, false));
    expect(a).toBe(b);
    for (const setting of OLLAMA_UNIVERSAL_SETTINGS) expect(a).toContain(setting);
    for (const setting of OLLAMA_OWNERSHIP_SETTINGS) expect(a).not.toContain(setting);
  });

  it("write the same settings for an Ollama they installed themselves", () => {
    const [a, b] = paths.map(([, s]) => bodyOutput(s, true));
    expect(a).toBe(b);
    expect(a).toContain("User=daniel");
    expect(a).toContain("Environment=OLLAMA_MODELS=");
  });

  // Fleet policy is autostart off: the :11434 API is unauthenticated and the
  // agent's firewall is applied when the agent starts, so an Ollama that comes
  // up at boot is briefly unfiltered. The two paths disagreed on this for a
  // long time — the install script enabled it, the provisioner disabled it —
  // and the fleet split accordingly (#12).
  it("both leave Ollama disabled for boot", () => {
    for (const [label, script] of paths) {
      expect(script, label).toMatch(/systemctl disable ollama/);
      expect(script, label).not.toMatch(/systemctl enable ollama/);
    }
  });

  // Neither may stop the service: a deployment may be serving from it right now.
  it("neither stops the service", () => {
    for (const [label, script] of paths) {
      expect(script, label).not.toMatch(/systemctl stop ollama/);
    }
  });

  // The predicate matters as much as the body: deciding ownership from whether
  // the unit is *enabled* misreads every node the manager provisioned, since
  // fleet policy disables Ollama.
  it("both decide ownership from the recorded marker, not the unit's state", () => {
    for (const [label, script] of paths) {
      expect(script, label).toContain(OLLAMA_MANAGED_MARKER);
      expect(script, label).not.toMatch(/is-enabled ollama/);
    }
  });
});
