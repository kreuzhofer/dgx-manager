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
