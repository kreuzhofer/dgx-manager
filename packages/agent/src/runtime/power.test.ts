import { describe, it, expect } from "vitest";
import { powerCommand, powerUnitName, powerLaunchCommand } from "./power.js";

describe("agent powerCommand", () => {
  it("maps graceful reboot/shutdown/sleep to the systemd variants", () => {
    expect(powerCommand("reboot")).toBe("sudo systemctl --no-block reboot");
    expect(powerCommand("shutdown")).toBe("sudo systemctl --no-block poweroff");
    expect(powerCommand("sleep")).toBe("sudo systemctl suspend");
  });
  // Force = immediate hard reset for a wedged node; only reboot/shutdown.
  it("maps forced reboot/shutdown to --force --force", () => {
    expect(powerCommand("reboot", { force: true })).toBe(
      "sudo systemctl --force --force reboot",
    );
    expect(powerCommand("shutdown", { force: true })).toBe(
      "sudo systemctl --force --force poweroff",
    );
  });
  it("ignores force for suspend", () => {
    expect(powerCommand("sleep", { force: true })).toBe("sudo systemctl suspend");
  });
  // Must match the server's command strings exactly so agent/SSH paths agree.
  it("matches the server's graceful reboot string", () => {
    expect(powerCommand("reboot")).toBe("sudo systemctl --no-block reboot");
  });
});

describe("powerUnitName", () => {
  // Bug fix: a fixed unit name collided ("Unit dgx-power.service already exists")
  // when a retry landed while a prior attempt's unit still lingered.
  it("produces a distinct unit name per invocation", () => {
    expect(powerUnitName(1000)).toBe("dgx-power-1000");
    expect(powerUnitName(1000)).not.toBe(powerUnitName(1001));
  });
});

describe("powerLaunchCommand", () => {
  it("launches the script via systemd-run in system.slice with the given unit + --collect", () => {
    const c = powerLaunchCommand("dgx-power-42", "/tmp/dgx-power.sh");
    expect(c).toContain("systemd-run");
    expect(c).toContain("--unit=dgx-power-42");
    expect(c).toContain("--slice=system.slice");
    expect(c).toContain("--collect");
    expect(c).toContain("bash /tmp/dgx-power.sh");
  });
});
