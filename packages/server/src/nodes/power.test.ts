import { describe, it, expect } from "vitest";
import { powerCommand, macCaptureCmd, normalizeMac } from "./power.js";

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

describe("normalizeMac", () => {
  it("lowercases and trims a captured MAC", () => {
    expect(normalizeMac("  AA:BB:CC:DD:EE:FF\n")).toBe("aa:bb:cc:dd:ee:ff");
  });
  it("returns null for empty or junk output", () => {
    expect(normalizeMac("")).toBeNull();
    expect(normalizeMac("device not found")).toBeNull();
  });
});
