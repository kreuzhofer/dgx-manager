import { describe, it, expect } from "vitest";
import {
  powerCommand,
  macCaptureCmd,
  wolArmCmd,
  normalizeMac,
  versionGte,
  agentSupportsPower,
  MIN_AGENT_POWER_VERSION,
} from "./power.js";

describe("powerCommand", () => {
  // Uses systemd --no-block so the SSH exec returns BEFORE the node drops,
  // giving the caller a clean exit code instead of a dropped-connection error.
  it("maps reboot to a non-blocking systemd reboot", () => {
    expect(powerCommand("reboot")).toBe("sudo systemctl --no-block reboot");
  });
  it("maps shutdown to a non-blocking systemd poweroff", () => {
    expect(powerCommand("shutdown")).toBe("sudo systemctl --no-block poweroff");
  });
  it("maps sleep to systemd suspend", () => {
    expect(powerCommand("sleep")).toBe("sudo systemctl suspend");
  });
  it("throws on an unknown action", () => {
    // @ts-expect-error invalid action
    expect(() => powerCommand("explode")).toThrow();
  });
  // A hung node may never complete a graceful --no-block reboot (systemd itself
  // can be stuck), so force reboot issues reboot(2) immediately without stopping
  // services or unmounting (double --force).
  it("maps a forced reboot to an immediate --force --force reboot", () => {
    expect(powerCommand("reboot", { force: true })).toBe(
      "sudo systemctl --force --force reboot",
    );
  });
  it("maps a forced shutdown to an immediate --force --force poweroff", () => {
    expect(powerCommand("shutdown", { force: true })).toBe(
      "sudo systemctl --force --force poweroff",
    );
  });
  it("ignores force for suspend", () => {
    expect(powerCommand("sleep", { force: true })).toBe("sudo systemctl suspend");
  });
  it("reboot without force stays graceful", () => {
    expect(powerCommand("reboot", { force: false })).toBe(
      "sudo systemctl --no-block reboot",
    );
  });
});

describe("macCaptureCmd", () => {
  it("builds an ssh command that finds the iface for an IP then reads its MAC", () => {
    const cmd = macCaptureCmd("192.168.44.41");
    expect(cmd).toContain("192.168.44.41");
    expect(cmd).toContain("/sys/class/net/");
  });
  // The ip is interpolated into a remote shell command, so it must be a strict
  // IPv4 literal — reject anything that could carry shell metacharacters.
  it("throws on a non-IPv4 / injection-bearing ip", () => {
    expect(() => macCaptureCmd('1.2.3.4"; reboot; "')).toThrow();
    expect(() => macCaptureCmd("not-an-ip")).toThrow();
    expect(() => macCaptureCmd("1.2.3.4 && rm -rf /")).toThrow();
    expect(() => macCaptureCmd("999.999.999.999")).toThrow();
  });
});

describe("wolArmCmd", () => {
  // Resolves the iface bound to `ip`, then arms magic-packet Wake-on-LAN on it.
  // Run best-effort before poweroff so a later /wake can actually reach the NIC.
  it("builds an ssh command that finds the iface for an IP then arms WOL via ethtool", () => {
    const cmd = wolArmCmd("192.168.44.41");
    expect(cmd).toContain("192.168.44.41");
    expect(cmd).toContain("ethtool -s");
    expect(cmd).toContain("wol g");
  });
  // Same injection surface as macCaptureCmd: the ip reaches a remote shell, so
  // it must be a strict IPv4 literal.
  it("throws on a non-IPv4 / injection-bearing ip", () => {
    expect(() => wolArmCmd('1.2.3.4"; reboot; "')).toThrow();
    expect(() => wolArmCmd("not-an-ip")).toThrow();
    expect(() => wolArmCmd("1.2.3.4 && rm -rf /")).toThrow();
  });
});

describe("versionGte / agentSupportsPower", () => {
  it("compares dotted-numeric versions component-wise", () => {
    expect(versionGte("0.5.645", "0.5.645")).toBe(true); // equal
    expect(versionGte("0.5.646", "0.5.645")).toBe(true);
    expect(versionGte("0.6.0", "0.5.645")).toBe(true);
    expect(versionGte("0.5.608", "0.5.645")).toBe(false);
    expect(versionGte("0.5.9", "0.5.645")).toBe(false); // 9 < 645, not string compare
  });
  it("treats missing/garbage versions as not-supported", () => {
    expect(versionGte(null, "0.5.645")).toBe(false);
    expect(versionGte(undefined, "0.5.645")).toBe(false);
    expect(versionGte("", "0.5.645")).toBe(false);
  });
  it("gates cmd:power support on the minimum version", () => {
    expect(agentSupportsPower(MIN_AGENT_POWER_VERSION)).toBe(true);
    expect(agentSupportsPower("0.5.608")).toBe(false);
    expect(agentSupportsPower(null)).toBe(false);
  });
});

describe("normalizeMac", () => {
  it("lowercases and trims a captured MAC", () => {
    expect(normalizeMac("  AA:BB:CC:DD:EE:FF\n")).toBe("aa:bb:cc:dd:ee:ff");
  });
  it("returns null for empty or junk output", () => {
    expect(normalizeMac("")).toBeNull();
    expect(normalizeMac("device not found")).toBeNull();
  });
});
