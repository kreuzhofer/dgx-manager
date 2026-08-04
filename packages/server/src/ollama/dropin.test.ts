import { describe, it, expect } from "vitest";
import { execFileSync } from "node:child_process";
import { chmodSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  ollamaDropInBody,
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
    env: { ...process.env, PATH: `${bin}:${process.env.PATH}`, OLLAMA_PREEXISTING: opts.preexisting ? "1" : "0" },
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

  it("both apply every universal setting", () => {
    for (const setting of OLLAMA_UNIVERSAL_SETTINGS) {
      expect(provisioner, "SSH provisioner").toContain(setting);
      expect(installScript, "install script").toContain(setting);
    }
  });

  it("both gate the ownership settings on having installed Ollama themselves", () => {
    for (const path of [provisioner, installScript]) {
      // The ownership block is emitted only inside the preexisting=0 branch.
      const guard = path.match(/if \[ "\$OLLAMA_PREEXISTING" = "0" \]; then\n\s*echo "User=/);
      expect(guard).not.toBeNull();
    }
  });

  it("both gate the model store on the mount", () => {
    for (const path of [provisioner, installScript]) {
      expect(path).toMatch(/OLLAMA_PREEXISTING" = "0" \] && mountpoint -q/);
    }
  });
});
